"""Turn a `Sample` + a `ModelConfig` into chat messages, doing media prep.

This is where the "native video vs. uniform frame sampling" decision is made.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from . import prompts
from .config import ImageConfig, ModelConfig
from .dataset import Sample
from .media import MediaCache, encode_image_block, sample_frames, video_file_block
from .media.video import frames_tag, probe_video

log = logging.getLogger(__name__)

Block = Dict[str, Any]


@dataclass
class MediaPayload:
    """Content blocks for one sample's media, plus what we log about them."""

    blocks: List[Block]
    preamble: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_media_blocks(self) -> int:
        return sum(1 for b in self.blocks if b.get("type") in {"image_url", "file"})


@dataclass
class PreparedRequest:
    messages: List[Dict[str, Any]]
    media: MediaPayload
    media_sent: bool = True

    @property
    def media_meta(self) -> Dict[str, Any]:
        meta = dict(self.media.meta)
        if not self.media_sent:
            meta["media_sent"] = False
        return meta

    @property
    def num_media_blocks(self) -> int:
        return self.media.num_media_blocks if self.media_sent else 0


def effective_system_prompt(
    system_prompt: Optional[str], include_media: bool
) -> Optional[str]:
    """The system message a request will carry, or None for no message.

    `system_prompt=None` takes the arm's default: no system message with media,
    and `BLIND_SYSTEM_PROMPT` on the text-only control. An explicit empty string
    suppresses the message on either arm.
    """
    if system_prompt is not None:
        return system_prompt or None
    return None if include_media else prompts.BLIND_SYSTEM_PROMPT


def prepare_media(sample: Sample, cfg: ModelConfig, cache: MediaCache) -> MediaPayload:
    if sample.is_video:
        return _video_payload(sample, cfg, cache)
    return _image_payload(sample, cfg, cache)


def build_request(
    sample: Sample,
    cfg: ModelConfig,
    cache: MediaCache,
    system_prompt: Optional[str] = None,
    include_media: bool = True,
) -> PreparedRequest:
    """The request for one sample.

    `include_media=False` is the blind control: the question text alone, with no
    media and no preamble describing it (a preamble such as "386 frames over
    277s" is itself information about the media).
    """
    if include_media:
        media = prepare_media(sample, cfg, cache)
        content: List[Block] = [
            {"type": "text", "text": media.preamble},
            *media.blocks,
            {"type": "text", "text": prompts.render_answer_instruction(sample)},
        ]
    else:
        media = MediaPayload(blocks=[], preamble="", meta={"kind": sample.media_kind})
        content = [{"type": "text", "text": prompts.render_answer_instruction(sample)}]
    system = effective_system_prompt(system_prompt, include_media)
    messages = [
        *([{"role": "system", "content": system}] if system else []),
        {"role": "user", "content": content},
    ]
    return PreparedRequest(messages=messages, media=media, media_sent=include_media)


# --------------------------------------------------------------------------
def _image_payload(sample: Sample, cfg: ModelConfig, cache: MediaCache) -> MediaPayload:
    if not cfg.supports_images:
        raise RuntimeError(f"Model '{cfg.name}' is configured without image support.")
    media = cache.fetch(sample.media_file, media_kind="image")
    block, meta = encode_image_block(media.path, cfg.image)
    meta.update(media_kind="image")
    return MediaPayload(blocks=[block], preamble=prompts.IMAGE_PREAMBLE, meta=meta)


def _video_payload(sample: Sample, cfg: ModelConfig, cache: MediaCache) -> MediaPayload:
    media = cache.fetch(sample.media_file, media_kind="video")

    if cfg.native_video:
        payload = _native_video_payload(sample, cfg, media)
        if payload is not None:
            return payload

    frames = sample_frames(
        media.path,
        cache.frames_dir(media.path, frames_tag(cfg.video)),
        cfg.video,
    )
    limit = cfg.video.max_images_per_request
    if limit and frames.num_frames > limit:
        log.info(
            "%s: %s has %d frames but the endpoint allows %d images/request; thinning uniformly",
            cfg.name, sample.sample_id, frames.num_frames, limit,
        )
        frames = frames.limit_to(limit)

    frame_cfg, budget_meta = _frame_image_config(sample, cfg, frames)

    blocks: List[Block] = []
    total_bytes = 0
    reencoded = 0
    for path, timestamp in zip(frames.paths, frames.timestamps):
        blocks.append({"type": "text", "text": f"[frame t={timestamp:.2f}s]"})
        block, frame_meta = encode_image_block(path, frame_cfg)
        blocks.append(block)
        total_bytes += frame_meta.get("encoded_bytes", 0)
        reencoded += bool(frame_meta.get("reencoded"))

    budget = cfg.video.max_total_image_bytes
    if budget and total_bytes > budget:
        log.warning(
            "%s: %s frames total %.1f MB, still above the %.1f MB budget after re-encoding",
            cfg.name, sample.sample_id, total_bytes / 1e6, budget / 1e6,
        )

    preamble = prompts.VIDEO_FRAMES_PREAMBLE.format(
        num_frames=frames.num_frames,
        fps=frames.sampled_fps,
        duration=frames.info.duration,
        timestamps=_format_timestamps(frames.timestamps),
    )
    meta = {
        "media_kind": "video",
        "video_mode": "frames",
        "total_image_bytes": total_bytes,
        "frames_reencoded_for_budget": reencoded,
        "size_budget_exceeded": bool(budget and total_bytes > budget),
        **budget_meta,
        **frames.to_dict(),
    }
    meta.pop("frames_dir", None)
    return MediaPayload(blocks=blocks, preamble=preamble, meta=meta)


def _frame_image_config(sample: Sample, cfg: ModelConfig, frames) -> tuple[ImageConfig, Dict[str, Any]]:
    """Per-frame encoding config that keeps the whole request under budget.

    Frames are only shrunk when their combined size would exceed
    `video.max_total_image_bytes`; capping each at `budget / n` guarantees the
    total fits while leaving already-small frames untouched.
    """
    budget = cfg.video.max_total_image_bytes
    source_total = sum(p.stat().st_size for p in frames.paths)
    meta: Dict[str, Any] = {"source_frame_bytes": source_total}

    if not budget or source_total <= budget or not frames.num_frames:
        return cfg.image, meta

    per_frame = max(1, budget // frames.num_frames)
    existing = cfg.image.max_bytes
    per_frame = min(per_frame, existing) if existing else per_frame
    log.info(
        "%s: %s frames total %.1f MB (> %.1f MB budget); capping each frame at %.0f KB",
        cfg.name, sample.sample_id, source_total / 1e6, budget / 1e6, per_frame / 1e3,
    )
    meta["per_frame_byte_budget"] = per_frame
    # Frames may shrink below the single-image floor: many small frames carry
    # more signal for a video question than a few large ones.
    floor = min(cfg.image.min_side, 128)
    return replace(cfg.image, max_bytes=per_frame, min_side=floor), meta


def _native_video_payload(sample: Sample, cfg: ModelConfig, media) -> Optional[MediaPayload]:
    """Whole-file video block, or None if the file is too big to upload inline."""
    size = media.size_bytes
    if cfg.video.native_max_bytes and size > cfg.video.native_max_bytes:
        log.warning(
            "%s: video %s is %.1f MiB (> native_max_bytes); falling back to frame sampling",
            cfg.name, sample.sample_id, size / (1024 * 1024),
        )
        return None
    info = probe_video(media.path)
    block, meta = video_file_block(media.path)
    meta.update(media_kind="video", video_mode="native", **info.to_dict())
    return MediaPayload(
        blocks=[block],
        preamble=prompts.VIDEO_NATIVE_PREAMBLE.format(duration=info.duration),
        meta=meta,
    )


def _format_timestamps(timestamps: List[float], max_listed: int = 24) -> str:
    if not timestamps:
        return "(none)"
    if len(timestamps) <= max_listed:
        return ", ".join(f"{t:.2f}" for t in timestamps)
    head = ", ".join(f"{t:.2f}" for t in timestamps[:8])
    tail = ", ".join(f"{t:.2f}" for t in timestamps[-4:])
    return f"{head}, ... (evenly spaced), {tail}"
