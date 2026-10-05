"""Stage 3: aggregate judgments into per-model scores and write summary.json."""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .utils import dedupe_records, read_jsonl, write_json

log = logging.getLogger(__name__)


@dataclass
class ModelScore:
    model: str
    display_name: str
    judge: str
    num_samples: int      # distinct samples evaluated (not trials)
    num_graded: int
    num_errors: int
    accuracy: Optional[float]
    rubric_score: Optional[float]
    criteria_met: int
    criteria_total: int
    breakdowns: Dict[str, Dict[str, Dict[str, Any]]]
    # Graded (sample, attempt) pairs. Equals num_samples when k=1.
    num_trials: int = 0
    # Bookkeeping for a dataset that grows over time.
    scope_size: int = 0                                   # samples in the current dataset
    missing_ids: List[str] = field(default_factory=list)  # in dataset, not yet evaluated
    orphan_ids: List[str] = field(default_factory=list)   # evaluated, no longer in dataset
    stale_ids: List[str] = field(default_factory=list)    # evaluated, but the sample has since been edited
    # pass@k. Empty/None when the run used a single attempt per sample.
    per_sample: List["SampleScore"] = field(default_factory=list)
    difficulty: Dict[str, Any] = field(default_factory=dict)
    # Scored 0 because the call failed or returned nothing, not because the
    # answer was wrong. Filled in by build_report.
    no_answer_ids: List[str] = field(default_factory=list)
    # Token usage, from the response records. Filled in by build_report.
    tokens: Dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> Optional[float]:
        return (self.num_samples / self.scope_size) if self.scope_size else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "display_name": self.display_name,
            "judge": self.judge,
            "accuracy": _round(self.accuracy),
            "rubric_score": _round(self.rubric_score),
            "num_samples": self.num_samples,
            "num_trials": self.num_trials,
            "num_graded": self.num_graded,
            "num_errors": self.num_errors,
            "criteria_met": self.criteria_met,
            "criteria_total": self.criteria_total,
            "scope_size": self.scope_size,
            "coverage": _round(self.coverage),
            "num_missing": len(self.missing_ids),
            "missing_sample_ids": self.missing_ids,
            "num_orphans": len(self.orphan_ids),
            "orphan_sample_ids": self.orphan_ids,
            "num_stale": len(self.stale_ids),
            "stale_sample_ids": self.stale_ids,
            "num_no_answer": len(self.no_answer_ids),
            "no_answer_sample_ids": self.no_answer_ids,
            "k": self.k,
            "pass_at_k": _round(self.pass_at_k),
            "difficulty": self.difficulty,
            "tokens": self.tokens,
            "breakdowns": self.breakdowns,
            "per_sample": [s.to_dict() for s in self.per_sample],
        }

    @property
    def k(self) -> int:
        """Attempts per sample actually present in the results."""
        return max((s.attempts for s in self.per_sample), default=1)

    @property
    def pass_at_k(self) -> Optional[float]:
        """Fraction of samples solved by at least one attempt.

        Equals `accuracy` when k=1, and rises above it when a model is
        inconsistent rather than incapable.
        """
        if not self.per_sample:
            return None
        return _mean([1.0 if s.solved else 0.0 for s in self.per_sample])


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def score_model(
    judgment_records: List[Dict[str, Any]],
    model: str,
    display_name: str = "",
    judge: str = "",
    breakdown_keys: Iterable[str] = ("media_kind", "domain", "subdomain"),
    scope_ids: Optional[Set[str]] = None,
    fingerprints: Optional[Dict[str, Dict[str, str]]] = None,
    too_easy_at: float = 0.6,
    sample_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> ModelScore:
    """Score one model.

    `scope_ids` is the set of sample ids in the *current* dataset. Records are
    deduplicated by sample id (append-only JSONL can hold a retry of the same
    sample) and restricted to that scope, so rows from samples that were removed
    cannot skew the metrics.

    `fingerprints` maps sample id -> the content hashes the current dataset
    expects. Sample ids are stable across edits, so this is what catches a
    judgment made against a rubric or a prompt that has since been rewritten:
    such a judgment is dropped and the sample counts as pending, rather than
    contributing a score nobody could reproduce.
    """
    records = dedupe_records(judgment_records)
    orphan_ids: List[str] = []
    if scope_ids is not None:
        in_scope, orphans = [], []
        for record in records:
            (in_scope if record.get("sample_id") in scope_ids else orphans).append(record)
        # One sample can contribute several records (k attempts), so report
        # distinct samples: a count of records would read as a sample count.
        records, orphan_ids = in_scope, sorted({str(r.get("sample_id")) for r in orphans})
        if orphan_ids:
            log.warning(
                "%s: ignoring %d evaluated sample(s) no longer in the dataset (e.g. %s)",
                model, len(orphan_ids), ", ".join(orphan_ids[:3]),
            )

    stale_ids: List[str] = []
    if fingerprints:
        fresh, stale = [], []
        for record in records:
            (stale if _is_stale(record, fingerprints) else fresh).append(record)
        records, stale_ids = fresh, sorted({str(r.get("sample_id")) for r in stale})
        if stale_ids:
            log.warning(
                "%s: ignoring %d edited sample(s) whose judgments predate the edit "
                "(e.g. %s) -- re-run to refresh them",
                model, len(stale_ids), ", ".join(stale_ids[:3]),
            )

    evaluated = {str(r.get("sample_id")) for r in records}
    missing_ids = sorted(scope_ids - evaluated) if scope_ids is not None else []

    usable = [r for r in records if r.get("rubric_score") is not None]
    errors = [r for r in records if r.get("status") == "error"]
    scores = [float(r["rubric_score"]) for r in usable]
    strict = [1.0 if r.get("strict_pass") else 0.0 for r in usable]
    per_sample = score_samples(usable, sample_meta)

    breakdowns: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key in breakdown_keys:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in usable:
            # Labels come from the current dataset, not the stored record.
            groups[str(_label(record, key, sample_meta) or "unknown")].append(record)
        breakdowns[key] = {
            name: {
                "n": len(rows),
                "accuracy": _round(_mean([1.0 if r.get("strict_pass") else 0.0 for r in rows])),
                "rubric_score": _round(_mean([float(r["rubric_score"]) for r in rows])),
            }
            for name, rows in sorted(groups.items())
        }

    return ModelScore(
        model=model,
        display_name=display_name or model,
        judge=judge or (records[0].get("judge", "") if records else ""),
        # Coverage is about samples, not trials: with k=3 a fully covered run of
        # 4 samples has 12 records and must still read 4/4, not 12/4.
        num_samples=len({str(r.get("sample_id")) for r in records}),
        num_trials=len(records),
        num_graded=len(usable),
        num_errors=len(errors),
        accuracy=_mean(strict),
        rubric_score=_mean(scores),
        criteria_met=sum(int(r.get("num_met") or 0) for r in usable),
        criteria_total=sum(len(r.get("criteria") or []) for r in usable),
        breakdowns=breakdowns,
        scope_size=len(scope_ids) if scope_ids is not None else len(records),
        missing_ids=missing_ids,
        orphan_ids=orphan_ids,
        stale_ids=stale_ids,
        per_sample=per_sample,
        difficulty=difficulty_summary(per_sample, too_easy_at=too_easy_at),
    )


def token_usage(response_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Token cost of producing the answers, from the stored `response.usage`.

    `output_tokens` is the provider's `completion_tokens`, which **includes**
    reasoning/thinking tokens -- at effort=high most of the output can be
    thinking the answer never shows, so a model that looks terse can still be
    the expensive one. `reasoning_tokens` is broken out so that share is
    visible rather than inferred.

    Medians are reported alongside means because the distribution is heavily
    right-skewed: one hard sample can spend the whole budget on thinking.
    """
    out, reasoning, prompt, total = [], [], [], []
    reported_any = False
    for record in response_records:
        usage = (record.get("response") or {}).get("usage") or {}
        if not usage:
            continue
        completion = usage.get("completion_tokens")
        if completion is None:
            continue
        out.append(int(completion))
        prompt.append(int(usage.get("prompt_tokens") or 0))
        total.append(int(usage.get("total_tokens") or 0)
                     or int(completion) + int(usage.get("prompt_tokens") or 0))
        details = usage.get("completion_tokens_details") or {}
        rt = details.get("reasoning_tokens")
        # A provider that omits the field entirely is "not reported"; one that
        # sends a real number (even 0) has told us something.
        if rt is not None:
            reported_any = True
        reasoning.append(int(rt or 0))
    if not out:
        return {}
    # An all-zero reasoning column means "not reported", not "did not think":
    # some routes omit the split even though completion_tokens includes it.
    reported = reported_any and any(reasoning)
    return {
        "n_with_usage": len(out),
        "reasoning_tokens_reported": reported,
        "output_tokens_mean": round(statistics.fmean(out), 1),
        "output_tokens_median": int(statistics.median(out)),
        "output_tokens_max": max(out),
        "output_tokens_total": sum(out),
        "reasoning_tokens_mean": round(statistics.fmean(reasoning), 1) if reported else None,
        "reasoning_tokens_total": sum(reasoning) if reported else None,
        # What fraction of the output was thinking rather than answer.
        "reasoning_share": (_round(sum(reasoning) / sum(out))
                            if reported and sum(out) else None),
        "prompt_tokens_mean": round(statistics.fmean(prompt), 1),
        "prompt_tokens_total": sum(prompt),
        "total_tokens_mean": round(statistics.fmean(total), 1),
        "total_tokens_median": int(statistics.median(total)),
        "total_tokens_total": sum(total),
    }


def _label(
    record: Dict[str, Any], key: str, sample_meta: Optional[Dict[str, Dict[str, str]]]
) -> Any:
    """Breakdown label for a record, preferring the current dataset's value."""
    if sample_meta:
        meta = sample_meta.get(str(record.get("sample_id")))
        if meta and meta.get(key):
            return meta[key]
    return record.get(key)


def _is_stale(record: Dict[str, Any], fingerprints: Dict[str, Dict[str, str]]) -> bool:
    """True if the sample has been edited since this record was written."""
    expected = fingerprints.get(str(record.get("sample_id"))) or {}
    return any(
        expected.get(key) and record.get(key) and record[key] != expected[key]
        for key in ("input_hash", "rubric_hash")
    )


@dataclass
class SampleScore:
    """How one sample fared across its k attempts."""

    sample_id: str
    attempts: int
    num_passed: int
    rubric_score: Optional[float]
    media_kind: str = ""
    domain: str = ""

    @property
    def pass_rate(self) -> float:
        """Fraction of attempts that met every criterion."""
        return self.num_passed / self.attempts if self.attempts else 0.0

    @property
    def solved(self) -> bool:
        """pass@k: did *any* attempt solve it."""
        return self.num_passed > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "attempts": self.attempts,
            "num_passed": self.num_passed,
            "pass_rate": _round(self.pass_rate),
            "solved": self.solved,
            "rubric_score": _round(self.rubric_score),
            "media_kind": self.media_kind,
            "domain": self.domain,
        }


def score_samples(
    judgments: List[Dict[str, Any]],
    sample_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[SampleScore]:
    """Collapse per-attempt judgments into one row per sample.

    Labels come from the current dataset when `sample_meta` is given, for the
    same reason the breakdowns do: a relabelled sample should not need
    re-judging to be reported correctly.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in judgments:
        groups[str(record.get("sample_id"))].append(record)
    out: List[SampleScore] = []
    for sample_id, rows in sorted(groups.items()):
        first = rows[0]
        out.append(
            SampleScore(
                sample_id=sample_id,
                attempts=len(rows),
                num_passed=sum(1 for r in rows if r.get("strict_pass")),
                rubric_score=_mean([float(r["rubric_score"]) for r in rows]),
                media_kind=str(_label(first, "media_kind", sample_meta) or ""),
                domain=str(_label(first, "domain", sample_meta) or ""),
            )
        )
    return out


def difficulty_summary(
    per_sample: List[SampleScore], too_easy_at: float = 0.6
) -> Dict[str, Any]:
    """Item difficulty across k attempts -- the benchmark-quality view.

    `pass@k` is the standard "at least one of k attempts succeeded". It rises
    with k by construction, so it is reported next to `pass@1` (the mean
    per-attempt rate, which does not) rather than instead of it.

    A sample nearly always solved carries little signal about model ability;
    one never solved is either genuinely hard or broken. Both lists are emitted
    so they can be inspected rather than guessed at.
    """
    if not per_sample:
        return {}
    k = max(s.attempts for s in per_sample)
    too_easy = [s for s in per_sample if s.pass_rate >= too_easy_at]
    never = [s for s in per_sample if s.num_passed == 0]
    always = [s for s in per_sample if s.attempts > 0 and s.num_passed == s.attempts]
    flaky = [s for s in per_sample if 0 < s.num_passed < s.attempts]
    return {
        "k": k,
        "num_samples": len(per_sample),
        "pass_at_k": _round(_mean([1.0 if s.solved else 0.0 for s in per_sample])),
        "pass_at_1": _round(_mean([s.pass_rate for s in per_sample])),
        "too_easy_at": too_easy_at,
        "num_too_easy": len(too_easy),
        "too_easy_sample_ids": [s.sample_id for s in too_easy],
        "num_always_solved": len(always),
        "always_solved_sample_ids": [s.sample_id for s in always],
        "num_never_solved": len(never),
        "never_solved_sample_ids": [s.sample_id for s in never],
        # Solved sometimes but not always: the items that actually discriminate,
        # and the reason a single-attempt run can mis-rank two close models.
        "num_flaky": len(flaky),
        "pass_rate_histogram": _pass_rate_histogram(per_sample),
    }


def _pass_rate_histogram(per_sample: List[SampleScore]) -> Dict[str, int]:
    hist: Dict[str, int] = defaultdict(int)
    for s in per_sample:
        hist[f"{s.num_passed}/{s.attempts}"] += 1
    return dict(sorted(hist.items()))


def rank(score: ModelScore):
    """Leaderboard order: accuracy first, rubric score as tie-break."""
    return (score.accuracy is None, -(score.accuracy or 0), -(score.rubric_score or 0))


def build_report(
    run_dir: Path | str,
    judgment_files: Dict[str, Path],
    response_files: Optional[Dict[str, Path]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    display_names: Optional[Dict[str, str]] = None,
    scope_ids: Optional[Set[str]] = None,
    fingerprints: Optional[Dict[str, Dict[str, str]]] = None,
    too_easy_at: float = 0.6,
    sample_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Score every model in a run and write `reports/summary.json`.

    `scope_ids` are the sample ids of the current dataset and `fingerprints`
    their content hashes. Together they make the numbers incremental-safe:
    retried duplicates collapse, samples added to the dataset since the run are
    reported as pending, results for removed samples are dropped, and results for
    samples that were *edited* are dropped rather than silently skewing scores.
    """
    run_dir = Path(run_dir)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    display_names = display_names or {}

    judgments_by_model = {model: read_jsonl(path) for model, path in judgment_files.items()}
    scores = [
        score_model(
            records,
            model=model,
            display_name=display_names.get(model, model),
            scope_ids=scope_ids,
            fingerprints=fingerprints,
            too_easy_at=too_easy_at,
            sample_meta=sample_meta,
        )
        for model, records in judgments_by_model.items()
        if records
    ]

    for score in scores:
        records = dedupe_records(read_jsonl((response_files or {}).get(score.model, Path("/nonexistent"))))
        if scope_ids is not None:
            records = [r for r in records if r.get("sample_id") in scope_ids]
        # Responses for an edited sample describe the pre-edit question, so they
        # are excluded here for the same reason their judgments are: otherwise a
        # sample can be dropped from scoring yet still counted as a no-answer,
        # and its tokens billed to the current dataset.
        if fingerprints:
            records = [r for r in records if not _is_stale(r, fingerprints)]
        score.no_answer_ids = sorted(
            {str(r.get("sample_id")) for r in records if r.get("status") not in {"ok", "dry_run"}}
        )
        score.tokens = token_usage(records)

    _warn_about_coverage(scores)

    summary = {
        "run": run_dir.name,
        "dataset": dataset_info or {},
        "models": [s.to_dict() for s in sorted(scores, key=rank)],
    }
    path = reports_dir / "summary.json"
    write_json(path, summary)
    summary["paths"] = {"summary_json": str(path)}
    return summary


def _warn_about_coverage(scores: List[ModelScore]) -> None:
    """Log anything that makes the scores less comparable than they look."""
    if len({s.num_samples for s in scores}) > 1:
        log.warning(
            "models were scored on different numbers of samples %s -- fill the gaps before comparing",
            {s.model: s.num_samples for s in scores},
        )
    for score in scores:
        if score.missing_ids:
            log.warning(
                "%s: %d of %d dataset samples not evaluated yet (e.g. %s)",
                score.model, len(score.missing_ids), score.scope_size, ", ".join(score.missing_ids[:3]),
            )
        if score.no_answer_ids:
            log.warning(
                "%s: %d sample(s) scored 0 because the call failed or returned nothing (%s) -- re-run to retry",
                score.model, len(score.no_answer_ids), ", ".join(score.no_answer_ids[:3]),
            )


def format_leaderboard(summary: Dict[str, Any]) -> str:
    """Compact stdout view of the same numbers that go into summary.json."""
    dataset = summary.get("dataset") or {}
    models = summary.get("models", [])
    k = max((entry.get("k") or 1 for entry in models), default=1)

    header = f"{'model':26}{'accuracy':>10}{'rubric':>9}"
    if k > 1:
        # pass@1 is the per-attempt rate already in `accuracy`; pass@k is the
        # headline that rises with k, so both are shown to keep them honest.
        header += f"{f'pass@{k}':>9}"
    header += f"{'out-tok':>13}{'think-tok':>13}{'total-tok':>14}{'coverage':>11}{'no-answer':>11}"

    lines = [
        f"run: {summary.get('run')}  |  dataset: {dataset.get('num_samples', '?')} samples, "
        f"{dataset.get('num_criteria', '?')} criteria"
        + (f"  |  k={k} attempts/sample" if k > 1 else ""),
        "",
        header,
        "-" * len(header),
    ]
    for entry in models:
        accuracy = "-" if entry["accuracy"] is None else f"{100 * entry['accuracy']:.1f}%"
        rubric = "-" if entry["rubric_score"] is None else f"{100 * entry['rubric_score']:.1f}%"
        coverage = f"{entry['num_samples']}/{entry['scope_size']}"
        row = f"{entry['display_name'][:25]:26}{accuracy:>10}{rubric:>9}"
        if k > 1:
            at_k = entry.get("pass_at_k")
            row += f"{('-' if at_k is None else f'{100 * at_k:.1f}%'):>9}"
        # Per-sample means, not totals: a total scales with how many samples a
        # model happened to cover, so a partially-evaluated model looks cheap
        # next to a complete one. Totals remain in summary.json for cost
        # attribution, where they are the right number.
        tok = entry.get('tokens') or {}
        out_t = tok.get('output_tokens_mean')
        think_t = tok.get('reasoning_tokens_mean')
        total_t = tok.get('total_tokens_mean')
        row += f"{('-' if out_t is None else f'{out_t:,.0f}'):>13}"
        # NA = the endpoint did not report the split, not that no thinking happened.
        row += f"{('NA' if think_t is None else f'{think_t:,.0f}'):>13}"
        row += f"{('-' if total_t is None else f'{total_t:,.0f}'):>14}"
        row += f"{coverage:>11}{entry['num_no_answer']:>11}"
        lines.append(row)

    if k > 1:
        lines.append("")
        lines.append("accuracy = pass@1 (mean over attempts); pass@k = solved by at least one attempt")
    lines.append("")
    lines.append("token columns are per-sample means; totals are in summary.json")
    partial = [
        (e["display_name"], (e.get("tokens") or {}).get("n_with_usage"), e["num_samples"])
        for e in models
        if (e.get("tokens") or {}).get("n_with_usage") not in (None, e["num_samples"])
    ]
    if partial:
        lines.append("")
        lines.append("token totals cover only the trials that recorded usage: "
                     + ", ".join(f"{n} {a}/{b}" for n, a, b in partial))
    return "\n".join(lines)


def _best_per_alias(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One entry per registry alias: the condition it scored highest at."""
    best: Dict[str, Dict[str, Any]] = {}
    for entry in models:
        alias = str(entry.get("model", "")).split("@", 1)[0]
        current = best.get(alias)
        if current is None or (entry.get("accuracy") or 0.0) > (current.get("accuracy") or 0.0):
            best[alias] = entry
    return list(best.values())


def format_difficulty(summary: Dict[str, Any], too_easy_at: float = 0.6) -> str:
    """Item difficulty pooled across models -- the benchmark-quality view.

    A sample every model solves on nearly every attempt is not measuring
    anything, and one no model ever solves is either genuinely hard or broken.
    Pooling across models is what makes the judgement about the *item* rather
    than about one model's luck.
    """
    models = summary.get("models", [])
    if not models:
        return ""

    # One vote per *model*, not per result file. An alias evaluated at two
    # reasoning efforts is two leaderboard rows but one model here, and a
    # deliberately weakened configuration voting alongside the real one would
    # push items across the too-easy line for a reason about the ablation
    # rather than about the item. Each alias is represented by its best
    # condition -- how a model is normally reported.
    models = _best_per_alias(models)
    pooled: Dict[str, Dict[str, int]] = defaultdict(lambda: {"passed": 0, "attempts": 0})
    for entry in models:
        for row in entry.get("per_sample", []):
            cell = pooled[row["sample_id"]]
            cell["passed"] += int(row.get("num_passed") or 0)
            cell["attempts"] += int(row.get("attempts") or 0)
    if not pooled:
        return ""

    rates = {
        sid: (c["passed"] / c["attempts"]) if c["attempts"] else 0.0
        for sid, c in pooled.items()
    }
    too_easy = sorted(sid for sid, r in rates.items() if r >= too_easy_at)
    never = sorted(sid for sid, r in rates.items() if r == 0.0)
    k = max((entry.get("k") or 1 for entry in models), default=1)
    num_models = len(models)

    lines = [
        "",
        f"item difficulty  (pooled over {num_models} model(s) x k={k}, {len(rates)} samples)",
        "-" * 67,
        f"  too easy   (pass rate >= {100 * too_easy_at:.0f}%): {len(too_easy):4d}  "
        f"({100 * len(too_easy) / len(rates):.0f}%)",
        f"  never solved (pass rate == 0%)   : {len(never):4d}  "
        f"({100 * len(never) / len(rates):.0f}%)",
        f"  discriminating (in between)      : "
        f"{len(rates) - len(too_easy) - len(never):4d}",
    ]
    if too_easy:
        lines.append(f"  too-easy ids: {', '.join(too_easy[:8])}"
                     + (f" ... (+{len(too_easy) - 8} more)" if len(too_easy) > 8 else ""))
    if never:
        lines.append(f"  never-solved ids: {', '.join(never[:8])}"
                     + (f" ... (+{len(never) - 8} more)" if len(never) > 8 else ""))
    return "\n".join(lines)
