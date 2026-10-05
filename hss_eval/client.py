"""Thin OpenAI-compatible client, with retries.

Every model is called through one OpenAI-compatible endpoint (typically a
LiteLLM proxy), so one client covers the whole registry. Parameters an endpoint
rejects (e.g. `reasoning_effort`, `temperature`) are dropped and the call is
retried.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from .config import Credentials, ModelConfig

log = logging.getLogger(__name__)

# 529 is Anthropic's transient "overloaded_error".
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524, 529}
_UNSUPPORTED_PARAM = re.compile(
    r"(unsupported|unrecognized|unknown|invalid|not supported|does not support)[^.]{0,80}?"
    r"['\"`]?(reasoning_effort|temperature|top_p|max_completion_tokens|max_tokens|thinking)['\"`]?",
    re.IGNORECASE,
)

DROPPABLE = ("reasoning_effort", "temperature", "top_p")
# Anthropic's Messages API is versioned by header, not by URL.
_ANTHROPIC_VERSION = "2023-06-01"
_DATA_URL = re.compile(r"^data:([^;]+);base64,(.*)$", re.S)


@dataclass
class ModelResponse:
    text: str
    model: str
    latency_sec: float
    attempts: int
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    dropped_params: List[str] = field(default_factory=list)
    reasoning_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "latency_sec": round(self.latency_sec, 3),
            "attempts": self.attempts,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "dropped_params": self.dropped_params,
            # The model's reasoning trace, when the provider returns one. `null`
            # means no trace was exposed, not that the model did not think.
            "reasoning_text": self.reasoning_text,
        }


class LiteLLMClient:
    def __init__(self, credentials: Credentials, timeout: float = 900.0):
        self._client = OpenAI(
            base_url=credentials.openai_base_url,
            api_key=credentials.api_key,
            timeout=timeout,
            max_retries=0,  # retries are handled here so we can mutate params
        )
        self.base_url = credentials.openai_base_url

    # -- request construction ---------------------------------------------
    def build_kwargs(self, cfg: ModelConfig, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"model": cfg.model_id, "messages": messages}
        if cfg.max_output_tokens:
            kwargs["max_completion_tokens"] = int(cfg.max_output_tokens)
        if cfg.supports_reasoning_effort and cfg.reasoning_effort:
            kwargs["reasoning_effort"] = cfg.reasoning_effort
        if cfg.temperature is not None:
            kwargs["temperature"] = float(cfg.temperature)
        if cfg.top_p is not None:
            kwargs["top_p"] = float(cfg.top_p)
        if cfg.extra_body:
            # Provider-specific fields (e.g. Anthropic `thinking`) are not named
            # parameters in the OpenAI SDK, so they must ride in extra_body.
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **cfg.extra_body}
        return kwargs

    def build_responses_kwargs(
        self, cfg: ModelConfig, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """The same request, in the Responses API's shape.

        Only the envelope differs: same model, same content, same effort. The
        reason to use it at all is `reasoning.summary`, which /chat/completions
        does not offer on any OpenAI model.
        """
        kwargs: Dict[str, Any] = {
            "model": cfg.model_id,
            "input": [_to_responses_message(m) for m in messages],
            # Do not have the provider persist the request: large multi-frame
            # requests exceed the size it will store, which fails the call.
            "store": False,
        }
        if cfg.max_output_tokens:
            kwargs["max_output_tokens"] = int(cfg.max_output_tokens)
        reasoning: Dict[str, Any] = {"summary": cfg.reasoning_summary or "auto"}
        if cfg.supports_reasoning_effort and cfg.reasoning_effort:
            reasoning["effort"] = cfg.reasoning_effort
        kwargs["reasoning"] = reasoning
        if cfg.temperature is not None:
            kwargs["temperature"] = float(cfg.temperature)
        if cfg.top_p is not None:
            kwargs["top_p"] = float(cfg.top_p)
        if cfg.extra_body:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **cfg.extra_body}
        return kwargs

    def build_anthropic_kwargs(
        self, cfg: ModelConfig, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """The same request in Anthropic's Messages shape.

        Used because this route reports
        `usage.output_tokens_details.thinking_tokens`, which the OpenAI-compatible
        shape does not carry.

        Anthropic differences the converter has to absorb: the system prompt is
        a top-level field rather than a message, effort lives in
        `output_config`, and the token cap is `max_tokens`.
        """
        system = "\n\n".join(
            m["content"] for m in messages
            if m.get("role") == "system" and isinstance(m.get("content"), str)
        )
        body: Dict[str, Any] = {
            "model": ModelConfig.alias_of(cfg.model_id or cfg.name).split("/")[-1],
            "max_tokens": int(cfg.max_output_tokens or 16384),
            "messages": [
                {"role": m.get("role", "user"), "content": _to_anthropic_content(m.get("content"))}
                for m in messages if m.get("role") != "system"
            ],
        }
        if system:
            body["system"] = system
        if cfg.supports_reasoning_effort and cfg.reasoning_effort:
            body["output_config"] = {"effort": cfg.reasoning_effort}
        if cfg.temperature is not None:
            body["temperature"] = float(cfg.temperature)
        if cfg.top_p is not None:
            body["top_p"] = float(cfg.top_p)
        for key, value in (cfg.extra_body or {}).items():
            body[key] = value
        return body

    # -- main entry point --------------------------------------------------
    def complete(
        self,
        cfg: ModelConfig,
        messages: List[Dict[str, Any]],
        request_timeout: Optional[float] = None,
    ) -> ModelResponse:
        cfg.require_ready()
        builder = {
            "responses": self.build_responses_kwargs,
            "messages": self.build_anthropic_kwargs,
        }.get(cfg.api, self.build_kwargs)
        kwargs = builder(cfg, messages)
        timeout = request_timeout or cfg.request.timeout
        dropped: List[str] = []
        last_error: Optional[Exception] = None
        started = time.time()

        for attempt in range(1, cfg.request.max_retries + 1):
            try:
                if cfg.api == "messages":
                    # Not an SDK namespace -- the low-level post still goes
                    # through the SDK's error handling, so non-2xx raises the
                    # same APIStatusError the retry loop below already expects.
                    raw = self._client.post(
                        "/v1/messages", body=kwargs, cast_to=object,
                        options={"headers": {"anthropic-version": _ANTHROPIC_VERSION},
                                 "timeout": timeout},
                    )
                    response = self._parse_anthropic(raw, cfg, started, attempt, dropped)
                elif cfg.api == "responses":
                    raw = self._client.responses.create(timeout=timeout, **kwargs)
                    response = self._parse_responses(raw, cfg, started, attempt, dropped)
                else:
                    raw = self._client.chat.completions.create(timeout=timeout, **kwargs)
                    response = self._parse(raw, cfg, started, attempt, dropped)
                if self._raise_budget_for_length(cfg, kwargs, response):
                    log.warning(
                        "%s: no answer within %s output tokens (all spent reasoning); "
                        "retrying with %s",
                        cfg.name, response.usage.get("completion_tokens"),
                        kwargs.get("max_completion_tokens") or kwargs.get("max_tokens"),
                    )
                    continue
                return response
            except APIStatusError as exc:
                last_error = exc
                message = _error_message(exc)
                dropped_now = self._drop_unsupported(kwargs, message)
                if dropped_now:
                    dropped.extend(dropped_now)
                    log.warning("%s: dropping unsupported param(s) %s and retrying", cfg.name, dropped_now)
                    continue
                if exc.status_code == 400 and "max_completion_tokens" in kwargs and "max_completion_tokens" in message:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    dropped.append("max_completion_tokens->max_tokens")
                    continue
                if exc.status_code not in RETRYABLE_STATUS or attempt == cfg.request.max_retries:
                    raise
            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                last_error = exc
                log.warning(
                    "%s: %s after %.0fs (attempt %d/%d); retrying",
                    cfg.name, type(exc).__name__, time.time() - started,
                    attempt, cfg.request.max_retries,
                )
                if attempt == cfg.request.max_retries:
                    raise
            self._sleep(attempt, cfg)

        raise RuntimeError(f"{cfg.name}: request failed after {cfg.request.max_retries} attempts") from last_error

    def _parse_responses(
        self, raw: Any, cfg: ModelConfig, started: float, attempt: int, dropped: List[str]
    ) -> "ModelResponse":
        """Read a Responses-API reply into the same shape as a chat reply.

        Everything downstream -- scoring, the token columns, the length-retry --
        reads the chat vocabulary, so the mapping happens here rather than
        spreading a second set of field names through the codebase.
        """
        data = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        text_parts: List[str] = []
        summary_parts: List[str] = []
        for item in data.get("output") or []:
            kind = item.get("type")
            if kind == "message":
                for part in item.get("content") or []:
                    if part.get("type") in {"output_text", "text"} and part.get("text"):
                        text_parts.append(part["text"])
            elif kind == "reasoning":
                for part in item.get("summary") or []:
                    if part.get("text"):
                        summary_parts.append(part["text"])

        raw_usage = data.get("usage") or {}
        out_details = raw_usage.get("output_tokens_details") or {}
        usage: Dict[str, Any] = {}
        if raw_usage:
            usage = {
                "prompt_tokens": raw_usage.get("input_tokens"),
                "completion_tokens": raw_usage.get("output_tokens"),
                "total_tokens": raw_usage.get("total_tokens"),
                "completion_tokens_details": {
                    "reasoning_tokens": out_details.get("reasoning_tokens"),
                },
                "prompt_tokens_details": {
                    "cached_tokens": (raw_usage.get("input_tokens_details") or {}).get("cached_tokens"),
                },
            }
            usage = {k: v for k, v in usage.items() if v is not None}

        # `incomplete` + `max_output_tokens` is this API's spelling of the chat
        # API's finish_reason="length", which is what triggers the budget retry.
        reason = (data.get("incomplete_details") or {}).get("reason")
        finish = "length" if reason == "max_output_tokens" else (
            "stop" if data.get("status") == "completed" else data.get("status")
        )
        return ModelResponse(
            text="\n".join(text_parts).strip(),
            model=data.get("model") or cfg.model_id or cfg.name,
            latency_sec=time.time() - started,
            attempts=attempt,
            finish_reason=finish,
            usage=usage,
            dropped_params=dropped,
            reasoning_text="\n\n".join(summary_parts).strip() or None,
        )

    def _parse_anthropic(
        self, raw: Any, cfg: ModelConfig, started: float, attempt: int, dropped: List[str]
    ) -> "ModelResponse":
        """Read an Anthropic Messages reply into the chat-shaped record.

        `thinking_tokens` is mapped onto `reasoning_tokens`.
        """
        data = raw if isinstance(raw, dict) else (
            raw.model_dump() if hasattr(raw, "model_dump") else dict(raw))
        blocks = data.get("content") or []
        text = "\n".join(b.get("text", "") for b in blocks
                         if isinstance(b, dict) and b.get("type") == "text").strip()
        thinking = "\n".join(b.get("thinking", "") for b in blocks
                             if isinstance(b, dict) and b.get("type") == "thinking").strip()

        raw_usage = data.get("usage") or {}
        usage: Dict[str, Any] = {}
        if raw_usage:
            out = raw_usage.get("output_tokens")
            prompt = raw_usage.get("input_tokens")
            usage = {
                "prompt_tokens": prompt,
                "completion_tokens": out,
                "total_tokens": raw_usage.get("total_tokens")
                or ((out or 0) + (prompt or 0) or None),
                "cache_creation_input_tokens": raw_usage.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": raw_usage.get("cache_read_input_tokens"),
            }
            thinking_tokens = (raw_usage.get("output_tokens_details") or {}).get("thinking_tokens")
            if thinking_tokens is not None:
                usage["completion_tokens_details"] = {"reasoning_tokens": thinking_tokens}
            usage = {k: v for k, v in usage.items() if v is not None}

        # Anthropic's spelling of finish_reason="length".
        stop = data.get("stop_reason")
        finish = "length" if stop == "max_tokens" else ("stop" if stop == "end_turn" else stop)
        return ModelResponse(
            text=text,
            model=data.get("model") or cfg.model_id or cfg.name,
            latency_sec=time.time() - started,
            attempts=attempt,
            finish_reason=finish,
            usage=usage,
            dropped_params=dropped,
            reasoning_text=thinking or None,
        )

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _drop_unsupported(kwargs: Dict[str, Any], message: str) -> List[str]:
        match = _UNSUPPORTED_PARAM.search(message)
        if not match:
            return []
        param = match.group(2).lower()
        if param in kwargs and param in DROPPABLE:
            kwargs.pop(param)
            return [param]
        # Some proxies name the offending field only implicitly; drop the usual suspect.
        for candidate in DROPPABLE:
            if candidate in message.lower() and candidate in kwargs:
                kwargs.pop(candidate)
                return [candidate]
        return []

    @staticmethod
    def _raise_budget_for_length(cfg: ModelConfig, kwargs: Dict[str, Any], response: "ModelResponse") -> bool:
        """Double the output budget when reasoning consumed all of it.

        A hard sample can spend the whole allowance on thinking and return no
        text, which would otherwise be scored as a wrong answer. Returns True if
        the budget was raised and the call should be retried.
        """
        ceiling = cfg.max_output_tokens_ceiling
        if not ceiling or response.text or response.finish_reason != "length":
            return False
        field = next((f for f in ("max_completion_tokens", "max_output_tokens", "max_tokens")
                      if f in kwargs), "max_tokens")
        current = kwargs.get(field)
        if not current or current >= ceiling:
            return False
        kwargs[field] = min(int(current) * 2, ceiling)
        return True

    @staticmethod
    def _sleep(attempt: int, cfg: ModelConfig) -> None:
        delay = min(cfg.request.initial_backoff * (2 ** (attempt - 1)), cfg.request.max_backoff)
        time.sleep(delay * (0.7 + 0.6 * random.random()))

    @staticmethod
    def _parse(raw: Any, cfg: ModelConfig, started: float, attempt: int, dropped: List[str]) -> ModelResponse:
        choice = raw.choices[0] if getattr(raw, "choices", None) else None
        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or "").strip()
        if not text and message is not None:
            # Some providers return a list of content parts.
            parts = getattr(message, "content", None)
            if isinstance(parts, list):
                text = "\n".join(
                    p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
        reasoning = _reasoning_text(message)

        usage: Dict[str, Any] = {}
        if getattr(raw, "usage", None) is not None:
            usage_obj = raw.usage
            dump = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else dict(usage_obj)
            usage = {k: v for k, v in dump.items() if v is not None}
            _drop_bogus_reasoning_tokens(cfg, usage)

        return ModelResponse(
            text=text,
            model=getattr(raw, "model", cfg.model_id or cfg.name),
            latency_sec=time.time() - started,
            attempts=attempt,
            finish_reason=getattr(choice, "finish_reason", None) if choice else None,
            usage=usage,
            dropped_params=dropped,
            reasoning_text=reasoning,
        )

    def list_remote_models(self) -> List[str]:
        return sorted(m.id for m in self._client.models.list().data)


def _to_anthropic_content(content: Any) -> Any:
    """Translate chat content blocks into Anthropic's shape.

    Images are the only real difference: OpenAI takes a (possibly `data:`) URL
    in `image_url`, Anthropic wants the base64 and its media type split apart
    in a `source` object. Media is still built once, in `hss_eval.media`, in
    the chat shape -- this only re-packages it.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks: List[Dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif kind == "image_url":
            url = (part.get("image_url") or {}).get("url") or ""
            match = _DATA_URL.match(url)
            if match:
                blocks.append({"type": "image", "source": {
                    "type": "base64",
                    "media_type": match.group(1),
                    "data": match.group(2),
                }})
            else:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            # Claude entries are `native_video: false`, so nothing should reach here.
            raise ValueError(
                f"cannot send a {kind!r} block to the Anthropic Messages API"
            )
    return blocks


def _to_responses_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one chat message into the Responses API's content vocabulary.

    Same bytes, different type names: `text` -> `input_text`, `image_url` ->
    `input_image`, `file` -> `input_file`. Media is built once, in
    `hss_eval.media`, in the chat shape; converting here keeps a second
    media pipeline from existing.
    """
    content = message.get("content")
    if isinstance(content, str):
        return {"role": message.get("role", "user"),
                "content": [{"type": "input_text", "text": content}]}
    parts: List[Dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            parts.append({"type": "input_text", "text": part.get("text", "")})
        elif kind == "image_url":
            image = part.get("image_url") or {}
            block = {"type": "input_image", "image_url": image.get("url")}
            if image.get("detail"):
                block["detail"] = image["detail"]
            parts.append(block)
        elif kind == "file":
            file_block = part.get("file") or {}
            parts.append({"type": "input_file",
                          "filename": file_block.get("filename"),
                          "file_data": file_block.get("file_data")})
        else:
            parts.append(part)
    return {"role": message.get("role", "user"), "content": parts}


def _drop_bogus_reasoning_tokens(cfg: ModelConfig, usage: Dict[str, Any]) -> None:
    """Remove `reasoning_tokens` for Anthropic models on the chat route.

    LiteLLM's OpenAI-compatible shape does not carry Anthropic's thinking-token
    count (BerriAI/litellm#31759); the number it reports there is not the billed
    thinking, so it is dropped and reported as unavailable. Use `api: messages`
    to get the real figure.
    """
    if cfg.vendor != "anthropic":
        return
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        details.pop("reasoning_tokens", None)
        details.pop("text_tokens", None)


def _reasoning_text(message: Any) -> Optional[str]:
    """The model's reasoning trace, whichever shape the provider used:
    `reasoning_content` / `reasoning` (e.g. Gemini, Kimi) or `thinking_blocks`
    (Anthropic on the chat route). Empty strings are normalised to None so "no trace" and "an empty trace"
    are not two different things downstream.
    """
    if message is None:
        return None
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    blocks = getattr(message, "thinking_blocks", None)
    if isinstance(blocks, list):
        joined = "\n".join(
            (b.get("thinking") or b.get("text") or "")
            for b in blocks
            if isinstance(b, dict)
        ).strip()
        if joined:
            return joined
    return None


def _error_message(exc: APIStatusError) -> str:
    try:
        body = exc.body
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or body)
            return str(body)
    except Exception:  # noqa: BLE001
        pass
    return str(exc)
