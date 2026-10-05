"""Image -> chat content block (base64 data URI)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from ..config import ImageConfig
from .cache import mime_for_path

Image.MAX_IMAGE_PIXELS = None  # large stills are downscaled below


def _downscale(img: Image.Image, max_side: int) -> Image.Image:
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return img.resize(new_size, Image.LANCZOS)


def _encode_jpeg(img: Image.Image, quality: int) -> bytes:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def image_data_uri(path: Path, cfg: Optional[ImageConfig] = None) -> Tuple[str, Dict[str, Any]]:
    """Return `(data_uri, meta)`, re-encoding only when limits require it.

    Original bytes are preferred so fine-grained visual detail is preserved;
    we fall back to a downscaled JPEG when the image is too large for provider
    payload limits.
    """
    cfg = cfg or ImageConfig()
    raw = path.read_bytes()
    mime = mime_for_path(path, "image")
    meta: Dict[str, Any] = {"source_bytes": len(raw), "reencoded": False}

    with Image.open(io.BytesIO(raw)) as probe:
        width, height = probe.size
    meta["source_size"] = [width, height]

    too_wide = cfg.max_side is not None and max(width, height) > cfg.max_side
    too_big = cfg.max_bytes is not None and len(raw) > cfg.max_bytes

    if too_wide or too_big:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            if cfg.max_side is not None:
                img = _downscale(img, cfg.max_side)
            data = _encode_jpeg(img, quality=92)
            # Shrink further if still over the byte budget.
            for quality in (85, 75, 65):
                if cfg.max_bytes is None or len(data) <= cfg.max_bytes:
                    break
                data = _encode_jpeg(img, quality=quality)
            floor = max(1, cfg.min_side)
            while cfg.max_bytes is not None and len(data) > cfg.max_bytes and max(img.size) > floor:
                img = _downscale(img, max(floor, int(max(img.size) * 0.8)))
                data = _encode_jpeg(img, quality=80)
            raw, mime = data, "image/jpeg"
            meta.update(reencoded=True, encoded_size=list(img.size))

    meta["encoded_bytes"] = len(raw)
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", meta


def encode_image_block(
    path: Path, cfg: Optional[ImageConfig] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build an OpenAI-style `image_url` content block from a local file."""
    cfg = cfg or ImageConfig()
    uri, meta = image_data_uri(path, cfg)
    meta["image_mode"] = "base64"
    return _block(uri, cfg), meta


def _block(url: str, cfg: ImageConfig) -> Dict[str, Any]:
    image_url: Dict[str, Any] = {"url": url}
    if cfg.detail:
        image_url["detail"] = cfg.detail
    return {"type": "image_url", "image_url": image_url}
