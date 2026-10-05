"""Stage 1: send each (media, prompt) to a model and record its answer."""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import LiteLLMClient
from .config import ModelConfig
from .dataset import Sample
from .media import MediaCache
from .messages import build_request, effective_system_prompt
from .prompts import system_prompt_fingerprint
from .utils import JsonlWriter, load_resumable_ids, threaded_map

log = logging.getLogger(__name__)


@dataclass
class GenerationOptions:
    workers: int = 4
    resume: bool = True
    system_prompt: Optional[str] = None
    dry_run: bool = False
    # Times to answer each sample (k). Each attempt is a separate trial.
    attempts: int = 1


def responses_path(run_dir: Path, model_name: str) -> Path:
    return Path(run_dir) / "responses" / f"{model_name}.jsonl"


def generate_responses(
    samples: List[Sample],
    cfg: ModelConfig,
    client: Optional[LiteLLMClient],
    cache: MediaCache,
    run_dir: Path,
    options: Optional[GenerationOptions] = None,
) -> Path:
    """Write one JSONL record per sample to `<run_dir>/responses/<model>.jsonl`.

    `client` may be None for a dry run, which prepares media and messages only.
    """
    options = options or GenerationOptions()
    if client is None and not options.dry_run:
        raise ValueError("a client is required unless options.dry_run is set")
    cfg.require_ready()
    # One file per condition: the label carries the reasoning effort.
    out_path = responses_path(run_dir, cfg.result_label)

    # Recorded on every row and part of the resume check, so answers produced
    # under a different system prompt are never reused.
    system_hash = system_prompt_fingerprint(
        effective_system_prompt(options.system_prompt, cfg.send_media)
    )

    attempts = max(1, int(options.attempts))
    # One unit of work per (sample, attempt). k attempts of a sample are separate
    # measurements, so they are scheduled and resumed independently.
    todo = [(sample, n) for sample in samples for n in range(1, attempts + 1)]
    total = len(todo)
    if options.resume:
        # An answer is reusable only if the question, the media and the system
        # prompt are unchanged.
        done, changed = load_resumable_ids(
            out_path,
            fingerprints={
                s.sample_id: {**s.fingerprints(), "system_prompt_hash": system_hash}
                for s in samples
            },
            keys=("input_hash", "system_prompt_hash"),
        )
        todo = [(s, n) for s, n in todo if f"{s.sample_id}#{n}" not in done]
        if done:
            log.info("%s: resuming, %d/%d trial(s) already done", cfg.name, len(done), total)
        if changed:
            log.info(
                "%s: %d trial(s) whose sample was edited since it was answered; "
                "re-generating (%s)",
                cfg.name, len(changed), ", ".join(sorted(changed)[:3]),
            )
    if not todo:
        log.info("%s: nothing to do", cfg.name)
        return out_path
    if attempts > 1:
        log.info(
            "%s: %d sample(s) x %d attempt(s) -> %d trial(s), %d outstanding",
            cfg.name, len(samples), attempts, total, len(todo),
        )

    writer = JsonlWriter(out_path)

    def run_one(item: tuple[Sample, int]) -> Dict[str, Any]:
        sample, attempt = item
        started = time.time()
        record: Dict[str, Any] = {
            "sample_id": sample.sample_id,
            "attempt": attempt,
            "attempts_requested": attempts,
            "row_index": sample.row_index,
            "model": cfg.name,
            "model_id": cfg.model_id,
            "reasoning_effort": cfg.reasoning_effort if cfg.supports_reasoning_effort else None,
            "media_kind": sample.media_kind,
            "domain": sample.domain,
            "subdomain": sample.subdomain,
            "prompt": sample.prompt,
            "media_path": sample.media_path,
            "media_sent": cfg.send_media,
            # What this answer was produced from; a later run compares these
            # against the dataset to tell an unchanged sample from an edited one.
            "input_hash": sample.input_hash,
            "rubric_hash": sample.rubric_hash,
            "system_prompt_hash": system_hash,
        }
        try:
            prepared = build_request(
                sample, cfg, cache, options.system_prompt, include_media=cfg.send_media
            )
            record["media"] = prepared.media_meta
            record["num_media_blocks"] = prepared.num_media_blocks
            if options.dry_run:
                record.update(status="dry_run", answer="")
                return record
            response = client.complete(cfg, prepared.messages)
            record.update(
                status="ok" if response.text else "empty",
                answer=response.text,
                response=response.to_dict(),
            )
            if not response.text:
                record["error"] = f"empty completion (finish_reason={response.finish_reason})"
        except Exception as exc:  # noqa: BLE001 - recorded, run continues
            record.update(
                status="error",
                answer="",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=6),
            )
            label = f"{sample.sample_id}#{attempt}" if attempts > 1 else sample.sample_id
            log.error("%s / %s failed: %s: %s", cfg.name, label, type(exc).__name__, exc)
        record["wall_sec"] = round(time.time() - started, 3)
        return record

    threaded_map(
        run_one,
        todo,
        workers=options.workers,
        desc=f"generate[{cfg.name}]",
        on_result=writer.write,
    )
    writer.close()
    return out_path
