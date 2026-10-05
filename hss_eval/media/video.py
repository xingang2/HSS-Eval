"""Video handling.

Two paths:

* Models that accept whole videos (`native_video: true`) get the file inline as a
  base64 data URI.
* Everything else gets **uniformly sampled frames**: target rate is `fps` (2.0 by
  default) with a hard cap of `max_frames` (500). If a 2 fps sample would exceed
  the cap, the rate is lowered to `max_frames / duration` so the frames still
  span the whole clip uniformly instead of truncating it.

Frame extraction uses ffmpeg's `fps` filter, which resamples the decoded stream
onto an evenly spaced grid.
"""

from __future__ import annotations

import base64
import contextlib
import errno
import fcntl
import json
import math
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..config import VideoConfig
from .cache import mime_for_path

FRAME_PATTERN = "frame_%05d.jpg"
FRAME_GLOB = "frame_*.jpg"
DONE_MARKER = "_sampling.json"


class VideoToolMissing(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise VideoToolMissing(
            f"`{tool}` not found on PATH. Install ffmpeg (e.g. `apt-get install ffmpeg`) "
            "to enable video frame sampling."
        )
    return path


@dataclass
class VideoInfo:
    duration: float
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    nb_frames: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_sec": round(self.duration, 3),
            "source_fps": self.fps,
            "width": self.width,
            "height": self.height,
            "source_nb_frames": self.nb_frames,
        }


@dataclass
class FrameSet:
    """The frames handed to a model, plus the metadata we log for the run."""

    paths: List[Path]
    timestamps: List[float]
    sampled_fps: float
    requested_fps: float
    max_frames: int
    info: VideoInfo
    truncated_by_cap: bool = False
    provider_limited: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.paths)

    def limit_to(self, count: int) -> "FrameSet":
        """Uniformly thin the frames to `count` (for provider image limits)."""
        if count >= len(self.paths):
            return self
        keep = _uniform_indices(len(self.paths), count)
        return FrameSet(
            paths=[self.paths[i] for i in keep],
            timestamps=[self.timestamps[i] for i in keep],
            sampled_fps=count / self.info.duration if self.info.duration > 0 else self.sampled_fps,
            requested_fps=self.requested_fps,
            max_frames=self.max_frames,
            info=self.info,
            truncated_by_cap=self.truncated_by_cap,
            provider_limited=True,
            meta={**self.meta, "frames_before_provider_limit": len(self.paths), "provider_image_limit": count},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "sampled_fps": round(self.sampled_fps, 6),
            "requested_fps": self.requested_fps,
            "max_frames": self.max_frames,
            "capped": self.truncated_by_cap,
            "provider_limited": self.provider_limited,
            "first_timestamp": round(self.timestamps[0], 3) if self.timestamps else None,
            "last_timestamp": round(self.timestamps[-1], 3) if self.timestamps else None,
            **self.info.to_dict(),
            **self.meta,
        }


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------
def probe_video(path: Path) -> VideoInfo:
    ffprobe = _require("ffprobe")
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path.name}: {proc.stderr.strip()[:400]}")
    doc = json.loads(proc.stdout or "{}")
    streams = doc.get("streams") or []
    stream = streams[0] if streams else {}

    duration = _to_float(doc.get("format", {}).get("duration")) or _to_float(stream.get("duration")) or 0.0
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    nb_frames = _to_int(stream.get("nb_frames"))
    if duration <= 0 and nb_frames and fps:
        duration = nb_frames / fps
    return VideoInfo(
        duration=duration,
        fps=fps,
        width=_to_int(stream.get("width")),
        height=_to_int(stream.get("height")),
        nb_frames=nb_frames,
    )


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rate(rate: Any) -> Optional[float]:
    if not rate or not isinstance(rate, str) or "/" not in rate:
        return _to_float(rate)
    num, _, den = rate.partition("/")
    num_f, den_f = _to_float(num), _to_float(den)
    if not num_f or not den_f:
        return None
    return num_f / den_f


# --------------------------------------------------------------------------
# uniform sampling
# --------------------------------------------------------------------------
def plan_sampling(duration: float, fps: float, max_frames: int) -> Tuple[float, int, bool]:
    """Return `(effective_fps, expected_frames, capped)`.

    At `fps` frames/second a `duration`-second clip yields ~`duration * fps`
    frames. When that exceeds `max_frames`, the rate is reduced so the cap is met
    while coverage stays uniform across the full clip.
    """
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if duration <= 0:
        return fps, 1, False
    target = max(1, int(math.floor(duration * fps)))
    if target <= max_frames:
        return fps, target, False
    return max_frames / duration, max_frames, True


def _scale_filter(max_side: Optional[int]) -> Optional[str]:
    if not max_side:
        return None
    # Never upscale: cap each dimension at max_side, preserving aspect ratio.
    return (
        f"scale=w='min({max_side},iw)':h='min({max_side},ih)'"
        ":force_original_aspect_ratio=decrease:flags=lanczos"
    )


def _sampling_tag(cfg: VideoConfig) -> str:
    side = cfg.max_side or "orig"
    return f"fps{cfg.fps:g}_max{cfg.max_frames}_side{side}_q{cfg.jpeg_quality}"


# Frame directories are keyed by (media file, sampling tag), so two models with
# the same video settings share one directory. Extraction deletes and rewrites
# that directory's contents, so it must not run twice at once: a concurrent
# reader would otherwise see half-written JPEGs (UnidentifiedImageError) or
# frames deleted out from under it (FileNotFoundError).
_EXTRACT_LOCKS: Dict[str, threading.Lock] = {}
_EXTRACT_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def _extraction_lock(out_dir: Path) -> Iterator[None]:
    """Serialise extraction into `out_dir`, across threads and processes."""
    key = str(out_dir.resolve())
    with _EXTRACT_LOCKS_GUARD:
        lock = _EXTRACT_LOCKS.setdefault(key, threading.Lock())
    with lock:
        # Also guard against a second `hss_eval` process sharing the cache. An
        # unsupported filesystem (some network mounts) is not fatal: the thread
        # lock above still covers the common single-process case.
        out_dir.mkdir(parents=True, exist_ok=True)
        handle = None
        try:
            handle = open(out_dir / ".extract.lock", "w")
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError as exc:
            if handle is not None:
                handle.close()
                handle = None
            if exc.errno not in {errno.ENOSYS, errno.ENOLCK, errno.EACCES, errno.EROFS}:
                raise
        try:
            yield
        finally:
            if handle is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()


def sample_frames(
    path: Path,
    out_dir: Path,
    cfg: Optional[VideoConfig] = None,
    force: bool = False,
) -> FrameSet:
    """Extract uniformly spaced frames into `out_dir` (cached across runs)."""
    cfg = cfg or VideoConfig()
    info = probe_video(path)
    effective_fps, expected, capped = plan_sampling(info.duration, cfg.fps, cfg.max_frames)

    marker = out_dir / DONE_MARKER
    existing = sorted(out_dir.glob(FRAME_GLOB))
    if force or not marker.exists() or not existing:
        with _extraction_lock(out_dir):
            # Re-check under the lock: another worker may have extracted this
            # same video while we were queued behind it, in which case reusing
            # its frames is both correct and free.
            existing = sorted(out_dir.glob(FRAME_GLOB))
            if force or not marker.exists() or not existing:
                for stale in existing:
                    stale.unlink(missing_ok=True)
                _run_ffmpeg(path, out_dir, effective_fps, cfg)
                existing = sorted(out_dir.glob(FRAME_GLOB))
                if not existing:
                    raise RuntimeError(f"ffmpeg produced no frames for {path.name}")
                marker.write_text(
                    json.dumps(
                        {
                            "effective_fps": effective_fps,
                            "requested_fps": cfg.fps,
                            "max_frames": cfg.max_frames,
                            "extracted": len(existing),
                            "expected": expected,
                            **info.to_dict(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    # ffmpeg's rounding can emit one extra frame; keep a uniform subset.
    paths = existing
    if len(paths) > cfg.max_frames:
        paths = _uniform_subset(paths, cfg.max_frames)
        capped = True

    indices = [int(p.stem.split("_")[-1]) - 1 for p in paths]
    timestamps = [idx / effective_fps for idx in indices]
    return FrameSet(
        paths=paths,
        timestamps=timestamps,
        sampled_fps=effective_fps,
        requested_fps=cfg.fps,
        max_frames=cfg.max_frames,
        info=info,
        truncated_by_cap=capped,
        meta={"extracted_frames": len(existing), "frames_dir": str(out_dir)},
    )


def _run_ffmpeg(path: Path, out_dir: Path, effective_fps: float, cfg: VideoConfig) -> None:
    ffmpeg = _require("ffmpeg")
    filters = [f"fps={effective_fps:.10g}"]
    scale = _scale_filter(cfg.max_side)
    if scale:
        filters.append(scale)
    cmd = [
        ffmpeg, "-nostdin", "-v", "error", "-y",
        "-i", str(path),
        "-vf", ",".join(filters),
        "-vsync", "0",
        # Belt-and-braces cap so rounding in the fps filter can never write more
        # than max_frames files.
        "-frames:v", str(cfg.max_frames),
        "-q:v", str(cfg.jpeg_quality),
        str(out_dir / FRAME_PATTERN),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path.name}: {proc.stderr.strip()[:600]}")


def _uniform_indices(total: int, count: int) -> List[int]:
    """`count` indices spread evenly over `range(total)`, endpoints included."""
    if count >= total:
        return list(range(total))
    if count == 1:
        return [total // 2]
    step = (total - 1) / float(count - 1)
    return sorted({min(total - 1, int(round(i * step))) for i in range(count)})


def _uniform_subset(items: List[Path], count: int) -> List[Path]:
    return [items[i] for i in _uniform_indices(len(items), count)]


def frames_tag(cfg: VideoConfig) -> str:
    return _sampling_tag(cfg)


# --------------------------------------------------------------------------
# native video block
# --------------------------------------------------------------------------
def video_file_block(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Content block for models that ingest video natively (Gemini-style `file`)."""
    raw = path.read_bytes()
    mime = mime_for_path(path, "video")
    uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    block = {"type": "file", "file": {"filename": path.name, "file_data": uri}}
    return block, {"native_video_bytes": len(raw), "mime_type": mime}
