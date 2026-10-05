"""Stage 2: grade model answers against the per-sample rubric criteria.

Each criterion is graded independently as met / not met. Two headline metrics
come out of that:

* `rubric_score`   -- weighted fraction of criteria met (partial credit)
* `strict_pass`    -- 1.0 only if every criterion is met
"""

from __future__ import annotations

import hashlib
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import prompts
from .client import LiteLLMClient
from .config import ModelConfig
from .dataset import Sample
from .media import MediaCache
from .messages import prepare_media
from .utils import (
    JsonlWriter,
    attempt_of,
    coerce_bool,
    extract_json_object,
    load_resumable_ids,
    threaded_map,
    trial_key,
)

log = logging.getLogger(__name__)


@dataclass
class JudgeOptions:
    workers: int = 4
    resume: bool = True
    include_media: bool = False
    max_parse_retries: int = 2


def judgments_path(run_dir: Path, model_name: str, judge_name: str) -> Path:
    return Path(run_dir) / "judgments" / f"{model_name}__by__{judge_name}.jsonl"


def answer_fingerprint(answer: str) -> str:
    """Identifies the answer a judgment was made about.

    A judgment of a failed response is a valid record ("no answer produced").
    Keying on the answer makes a retried sample's new answer get judged afresh.
    """
    return hashlib.sha1((answer or "").strip().encode("utf-8")).hexdigest()[:12]


def judge_responses(
    samples: List[Sample],
    responses: List[Dict[str, Any]],
    judge_cfg: ModelConfig,
    client: LiteLLMClient,
    run_dir: Path,
    model_name: str,
    cache: Optional[MediaCache] = None,
    options: Optional[JudgeOptions] = None,
) -> Path:
    options = options or JudgeOptions()
    judge_cfg.require_ready()
    label = judge_cfg.name
    out_path = judgments_path(run_dir, model_name, label)

    by_id = {s.sample_id: s for s in samples}
    pending = [r for r in responses if r.get("sample_id") in by_id]
    if options.resume:
        # A judgment survives only while the answer and what it was graded
        # against are both unchanged. A fixed rubric or golden response re-judges
        # the existing answer; it does not cost another generation. A *new*
        # answer (a retried failure) invalidates the judgment outright.
        # Under pass@k each attempt has its own answer, so the answer hash is
        # keyed per trial rather than per sample.
        expected = {}
        for record in pending:
            sample = by_id[record["sample_id"]]
            expected[trial_key(record)] = {
                **sample.fingerprints(),
                "answer_hash": answer_fingerprint(record.get("answer") or ""),
            }
        done, changed = load_resumable_ids(
            out_path,
            fingerprints=expected,
            keys=("input_hash", "rubric_hash", "answer_hash"),
        )
        pending = [r for r in pending if trial_key(r) not in done]
        if done:
            log.info("%s: resuming judge, %d already graded", model_name, len(done))
        if changed:
            log.info(
                "%s: %d sample(s) edited since they were graded; re-judging (%s)",
                model_name, len(changed), ", ".join(sorted(changed)[:3]),
            )
    if not pending:
        log.info("%s: nothing to judge", model_name)
        return out_path

    writer = JsonlWriter(out_path)

    def run_one(response_record: Dict[str, Any]) -> Dict[str, Any]:
        sample = by_id[response_record["sample_id"]]
        started = time.time()
        record: Dict[str, Any] = {
            "sample_id": sample.sample_id,
            "attempt": attempt_of(response_record),
            "row_index": sample.row_index,
            "model": model_name,
            "judge": label,
            "judge_model_id": judge_cfg.model_id,
            "media_kind": sample.media_kind,
            "domain": sample.domain,
            "subdomain": sample.subdomain,
            "num_criteria": len(sample.criteria),
            # What was graded, and what it was graded against. `report` drops a
            # judgment whose rubric has since been rewritten.
            "input_hash": sample.input_hash,
            "rubric_hash": sample.rubric_hash,
            "answer_hash": answer_fingerprint(response_record.get("answer") or ""),
        }
        answer = (response_record.get("answer") or "").strip()

        # An upstream failure is a zero, not a grading error.
        if response_record.get("status") not in {"ok", "empty"} or not answer:
            record.update(
                status="ok",
                graded=False,
                reason=response_record.get("error") or "no answer produced",
                criteria=[
                    {"id": c.id, "title": c.title, "met": False, "justification": "no answer produced"}
                    for c in sample.criteria
                ],
                num_met=0,
                rubric_score=0.0,
                strict_pass=False,
            )
            record["wall_sec"] = round(time.time() - started, 3)
            return record

        if not sample.criteria:
            record.update(
                status="skipped",
                graded=False,
                reason="sample has no rubric criteria",
                criteria=[],
                num_met=0,
                rubric_score=None,
                strict_pass=None,
            )
            record["wall_sec"] = round(time.time() - started, 3)
            return record

        try:
            messages = _build_judge_messages(sample, answer, judge_cfg, options, cache)
            parsed, raw_text, attempts = _grade_with_retries(client, judge_cfg, messages, sample, options)
            graded = _score(sample, parsed)
            record.update(status="ok", graded=True, judge_raw=raw_text, judge_attempts=attempts, **graded)
            if parsed is None:
                record["reason"] = "judge output was not parseable JSON"
        except Exception as exc:  # noqa: BLE001
            record.update(
                status="error",
                graded=False,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=6),
                criteria=[],
                num_met=0,
                rubric_score=None,
                strict_pass=None,
            )
            log.error("judge %s / %s failed: %s", judge_cfg.name, sample.sample_id, exc)
        record["wall_sec"] = round(time.time() - started, 3)
        return record

    threaded_map(
        run_one,
        pending,
        workers=options.workers,
        desc=f"judge[{model_name}]",
        on_result=writer.write,
    )
    writer.close()
    return out_path


# --------------------------------------------------------------------------
def _build_judge_messages(
    sample: Sample,
    answer: str,
    judge_cfg: ModelConfig,
    options: JudgeOptions,
    cache: Optional[MediaCache],
) -> List[Dict[str, Any]]:
    system_prompt = prompts.JUDGE_SYSTEM_PROMPT
    user_text = prompts.render_judge_user_prompt(sample, answer)
    if not options.include_media:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
    if cache is None:
        raise ValueError("include_media=True requires a MediaCache")
    payload = prepare_media(sample, judge_cfg, cache)
    content = [
        {"type": "text", "text": f"Media under evaluation. {payload.preamble}"},
        *payload.blocks,
        {"type": "text", "text": user_text},
    ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _grade_with_retries(
    client: LiteLLMClient,
    judge_cfg: ModelConfig,
    messages: List[Dict[str, Any]],
    sample: Sample,
    options: JudgeOptions,
):
    attempts = 0
    raw_text = ""
    for attempt in range(1, options.max_parse_retries + 2):
        attempts = attempt
        response = client.complete(judge_cfg, messages)
        raw_text = response.text
        parsed = extract_json_object(raw_text)
        if parsed is not None and isinstance(parsed.get("criteria"), list):
            return parsed, raw_text, attempts
        if attempt <= options.max_parse_retries:
            log.warning(
                "judge %s returned unparseable output for %s (attempt %d); retrying",
                judge_cfg.name, sample.sample_id, attempt,
            )
            messages = messages + [
                {"role": "assistant", "content": raw_text or ""},
                {
                    "role": "user",
                    "content": (
                        "That was not valid JSON. Reply with ONLY the JSON object, no fences and no "
                        f'prose, with one entry per criterion id in {prompts.criteria_ids_json(sample.criteria)}.'
                    ),
                },
            ]
    return None, raw_text, attempts


def _score(sample: Sample, parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Align the judge's verdicts with the rubric and compute scores."""
    verdicts: Dict[str, Dict[str, Any]] = {}
    ordered: List[Dict[str, Any]] = []
    if parsed:
        raw_items = parsed.get("criteria") or []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            ordered.append(item)
            cid = str(item.get("id") or "").strip()
            if cid:
                verdicts[cid] = item

    graded: List[Dict[str, Any]] = []
    total_weight = 0.0
    met_weight = 0.0
    for position, crit in enumerate(sample.criteria):
        item = verdicts.get(crit.id)
        if item is None and position < len(ordered):
            item = ordered[position]  # judge dropped/renamed ids: fall back to order
        met = coerce_bool((item or {}).get("met"))
        justification = str((item or {}).get("justification") or "").strip()
        if met is None:
            met = False
            justification = justification or "judge did not return a verdict for this criterion"
        total_weight += crit.weight
        if met:
            met_weight += crit.weight
        graded.append(
            {"id": crit.id, "title": crit.title, "weight": crit.weight, "met": met, "justification": justification}
        )

    num_met = sum(1 for c in graded if c["met"])
    return {
        "criteria": graded,
        "num_met": num_met,
        "rubric_score": (met_weight / total_weight) if total_weight else None,
        "strict_pass": bool(graded) and num_met == len(graded),
        "judge_comment": str((parsed or {}).get("overall_comment") or "").strip(),
    }
