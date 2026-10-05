"""Local media access and the on-disk cache for extracted video frames."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

_EXT_MIME: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


def mime_for_path(path: Path, media_kind: str = "image") -> str:
    return _EXT_MIME.get(path.suffix.lower(), "image/jpeg" if media_kind == "image" else "video/mp4")


@dataclass
class MediaFile:
    path: Path
    media_kind: str

    @property
    def mime_type(self) -> str:
        return mime_for_path(self.path, self.media_kind)

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size


class MediaCache:
    """Resolves a sample's media file and owns `<root>/frames/` for video frames."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def fetch(self, path: Path | str, media_kind: str = "image") -> MediaFile:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Media file not found: {path}. Is the dataset fully downloaded?"
            )
        return MediaFile(path=path, media_kind=media_kind)

    def frames_dir(self, path: Path | str, tag: str) -> Path:
        """Directory for the frames of one video under one sampling config."""
        key = hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()
        out = self.root / "frames" / key / tag
        out.mkdir(parents=True, exist_ok=True)
        return out

    def disk_usage_mb(self) -> float:
        total = 0
        for dirpath, _, filenames in os.walk(self.root):
            for name in filenames:
                try:
                    total += (Path(dirpath) / name).stat().st_size
                except OSError:
                    pass
        return total / (1024 * 1024)
