"""Configuration: .env loading and the model registry (configs/models.yaml)."""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_YAML = REPO_ROOT / "configs" / "models.yaml"
DEFAULT_CACHE_DIR = REPO_ROOT / "cache"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

_API_KEY_NAMES = ("LITELLM_API_KEY", "OPENAI_API_KEY")
_BASE_URL_NAMES = ("LITELLM_BASE_URL", "OPENAI_BASE_URL")

# Marks the text-only control arm in result labels and filenames.
NO_MEDIA_SUFFIX = "+nomedia"

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# .env
# --------------------------------------------------------------------------
def load_dotenv(path: Path | str | None = None, override: bool = False) -> Dict[str, str]:
    """Minimal .env parser (no dependency on python-dotenv).

    Tolerates `KEY = value`, quoted values, `export KEY=value` and comments.
    Values are injected into os.environ.
    """
    path = Path(path) if path else REPO_ROOT / ".env"
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return values


@dataclass
class Credentials:
    base_url: str
    api_key: str

    @property
    def openai_base_url(self) -> str:
        """The configured endpoint, used verbatim (bar a trailing slash)."""
        return self.base_url.rstrip("/")


def load_credentials(env_file: Path | str | None = None) -> Credentials:
    """Endpoint and key from `.env`, falling back to the environment."""
    values = load_dotenv(env_file)
    base_url = _first_value(_BASE_URL_NAMES, values)
    api_key = _first_value(_API_KEY_NAMES, values)
    if not base_url or not api_key:
        raise RuntimeError(
            "Missing API credentials. Create a .env in the repo root with:\n"
            "  LITELLM_BASE_URL=https://your-endpoint/\n  LITELLM_API_KEY=sk-...\n"
            "(see .env.example)"
        )
    return Credentials(base_url=base_url, api_key=api_key)


def _first_value(names: tuple[str, ...], dotenv: Dict[str, str]) -> Optional[str]:
    for source in (dotenv, os.environ):
        for name in names:
            value = source.get(name)
            if value and value.strip():
                return value.strip()
    return None


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
@dataclass
class VideoConfig:
    fps: float = 2.0
    max_frames: int = 500
    max_side: Optional[int] = 768
    jpeg_quality: int = 3
    # Largest video sent whole to a `native_video` model; bigger files fall back
    # to frame sampling.
    native_max_bytes: int = 60 * 1024 * 1024
    # Per-request image limit imposed by some endpoints. Frames are thinned
    # uniformly to fit. None = no extra cap beyond max_frames.
    max_images_per_request: Optional[int] = None
    # Budget for the combined size of all frames in one request; oversized
    # frames are re-encoded smaller to fit. None = no budget.
    max_total_image_bytes: Optional[int] = 32_000_000


@dataclass
class ImageConfig:
    max_side: Optional[int] = 4096
    max_bytes: Optional[int] = 4_500_000
    # Downscaling to meet `max_bytes` stops at this size.
    min_side: int = 512
    detail: Optional[str] = None


@dataclass
class RequestConfig:
    timeout: float = 1200.0
    max_retries: int = 5
    initial_backoff: float = 4.0
    max_backoff: float = 90.0


@dataclass
class ModelConfig:
    """One row of the registry: an alias plus everything needed to call it."""

    name: str
    model_id: Optional[str]
    display_name: str = ""
    vendor: str = ""
    enabled: bool = True
    supports_images: bool = True
    supports_reasoning_effort: bool = True
    # Request shape: "chat" (/chat/completions, the default), "responses"
    # (OpenAI Responses API, which returns reasoning summaries) or "messages"
    # (Anthropic Messages API, which reports thinking-token counts).
    api: str = "chat"
    # `reasoning.summary` on the Responses API.
    reasoning_summary: str = "auto"
    native_video: bool = False
    reasoning_effort: Optional[str] = "high"
    # Effort levels the endpoint accepts. `--effort` skips models that do not
    # list the requested level. Empty = unknown, so nothing is skipped.
    supported_efforts: List[str] = field(default_factory=list)
    max_output_tokens: Optional[int] = 16384
    # If a call ends with finish_reason=length and no visible answer (all budget
    # spent on reasoning), retry with the budget doubled, up to this ceiling.
    max_output_tokens_ceiling: Optional[int] = 65536
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)
    video: VideoConfig = field(default_factory=VideoConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    # False = text-only control arm (set by --no-media, not by the registry).
    send_media: bool = True

    def label(self) -> str:
        return self.display_name or self.name

    @property
    def result_label(self) -> str:
        """Filename stem for this model's results: `<alias>@<effort>[+nomedia]`.

        Effort is part of the condition, so one alias run at two efforts writes
        two files rather than mixing them.
        """
        label = self.name
        if self.supports_reasoning_effort and self.reasoning_effort:
            label = f"{label}@{self.reasoning_effort}"
        if not self.send_media:
            label = f"{label}{NO_MEDIA_SUFFIX}"
        return label

    @staticmethod
    def alias_of(result_label: str) -> str:
        """Registry alias behind a result label (`gpt-6-astra@max` -> alias)."""
        stem = result_label
        if stem.endswith(NO_MEDIA_SUFFIX):
            stem = stem[: -len(NO_MEDIA_SUFFIX)]
        return stem.split("@", 1)[0]

    def require_ready(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                f"Model '{self.name}' is disabled in the registry. Set `enabled: true` "
                "in configs/models.yaml to use it."
            )
        if not self.model_id:
            raise RuntimeError(f"Model '{self.name}' has no `model_id` in configs/models.yaml.")


@dataclass
class JudgeSettings:
    default: str
    include_media: bool = False


@dataclass
class Registry:
    models: Dict[str, ModelConfig]
    judges: Dict[str, ModelConfig]
    judge_settings: JudgeSettings
    source: Path

    def get(self, name: str) -> ModelConfig:
        if name in self.models:
            return self.models[name]
        if name in self.judges:
            return self.judges[name]
        known = ", ".join(sorted({*self.models, *self.judges}))
        raise KeyError(f"Unknown model alias '{name}'. Known aliases: {known}")

    def enabled_models(self) -> List[ModelConfig]:
        return [m for m in self.models.values() if m.enabled and m.model_id]

    def resolve_selection(self, selection: Optional[str]) -> List[ModelConfig]:
        """`None`/'all' -> every enabled model; otherwise a comma-separated list."""
        if selection is None or selection.strip().lower() in {"all", "*"}:
            chosen = self.enabled_models()
            if not chosen:
                raise RuntimeError("No enabled models with a model_id in the registry.")
            return chosen
        out: List[ModelConfig] = []
        for name in (p.strip() for p in selection.split(",")):
            if not name:
                continue
            cfg = self.get(name)
            cfg.require_ready()
            out.append(cfg)
        if not out:
            raise RuntimeError(f"No models selected from '{selection}'.")
        return out


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build_model(name: str, raw: Dict[str, Any]) -> ModelConfig:
    raw = dict(raw or {})
    video = VideoConfig(**(raw.pop("video", None) or {}))
    image = ImageConfig(**(raw.pop("image", None) or {}))
    request = RequestConfig(**(raw.pop("request", None) or {}))
    known = {f for f in ModelConfig.__dataclass_fields__ if f not in {"name", "video", "image", "request"}}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Model '{name}': unknown config key(s) {sorted(unknown)}")
    return ModelConfig(name=name, video=video, image=image, request=request, **raw)


def load_registry(path: Path | str | None = None) -> Registry:
    path = Path(path) if path else DEFAULT_MODELS_YAML
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = doc.get("defaults") or {}

    models: Dict[str, ModelConfig] = {}
    for name, raw in (doc.get("models") or {}).items():
        models[name] = _build_model(name, _deep_merge(defaults, raw or {}))

    judge_doc = doc.get("judge") or {}
    judges: Dict[str, ModelConfig] = {}
    for name, raw in (judge_doc.get("models") or {}).items():
        judges[name] = _build_model(name, _deep_merge(defaults, raw or {}))

    default_judge = judge_doc.get("default") or next(iter(judges), next(iter(models), ""))
    settings = JudgeSettings(
        default=default_judge,
        include_media=bool(judge_doc.get("include_media", False)),
    )
    return Registry(models=models, judges=judges, judge_settings=settings, source=path)
