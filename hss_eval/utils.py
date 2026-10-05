"""Small shared helpers: JSONL IO, logging setup, threaded map with progress."""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")
R = TypeVar("R")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------
class JsonlWriter:
    """Append-only JSONL writer, safe to use from multiple threads."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logging.getLogger(__name__).warning("%s:%d is not valid JSON; skipping", path, lineno)
    return out


def attempt_of(record: Dict[str, Any]) -> int:
    """Which repeat of a sample this record is (1-based)."""
    try:
        return max(1, int(record.get("attempt") or 1))
    except (TypeError, ValueError):
        return 1


def trial_key(record: Dict[str, Any], key: str = "sample_id") -> Optional[str]:
    """Identity of one *trial*: a sample plus which repeat of it this is.

    pass@k runs a sample k times, so the sample id alone no longer identifies a
    row -- deduplicating on it would collapse k attempts into one and silently
    turn a pass@k run back into a single-attempt one.
    """
    ident = record.get(key)
    if ident is None:
        return None
    return f"{ident}#{attempt_of(record)}"


def load_resumable_ids(
    path: Path | str,
    fingerprints: Optional[Dict[str, Dict[str, str]]] = None,
    keys: Sequence[str] = (),
    require_ok: bool = True,
) -> tuple[set[str], set[str]]:
    """Split the trials in a result file into (reusable, changed).

    Both sets contain *trial keys* (`"<sample_id>#<attempt>"`, see `trial_key`),
    not bare sample ids, so a partially completed pass@k run resumes at the right
    attempt instead of re-running all k or none.

    `fingerprints` maps sample id -> the hashes the *current* dataset expects
    (see `dataset.Sample.fingerprints`). A record whose stored hash differs from
    the expected one describes a sample that has since been edited, so its result
    can no longer be reused.

    The file is append-only, so the last `ok` record for a trial wins: a sample
    re-run after an edit ends up reusable again.
    """
    verdict: Dict[str, bool] = {}
    for record in read_jsonl(path):
        if require_ok and record.get("status") != "ok":
            continue
        key = trial_key(record)
        if key is None:
            continue
        # Expectations may be keyed per trial (the judge, whose answer hash
        # differs between attempts) or per sample (generation, where every
        # attempt shares the same prompt and media).
        lookup = fingerprints or {}
        expected = lookup.get(key) or lookup.get(str(record.get("sample_id"))) or {}
        verdict[key] = all(
            not expected.get(field) or not record.get(field) or record[field] == expected[field]
            for field in keys
        )
    reusable = {k for k, ok in verdict.items() if ok}
    return reusable, set(verdict) - reusable


def dedupe_records(
    records: List[Dict[str, Any]], key: str = "sample_id"
) -> List[Dict[str, Any]]:
    """One record per *trial*, preferring a successful row, then the latest.

    JSONL files are append-only, so a trial that errored on one pass and
    succeeded on a later one has two rows. Counting both would double-count it
    and inflate the error tally. Under pass@k the unit is (sample, attempt), so
    distinct attempts of one sample are kept -- they are the measurement.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for record in records:
        ident = trial_key(record, key)
        if ident is None:
            continue
        previous = best.get(ident)
        if previous is None or _record_rank(record) >= _record_rank(previous):
            best[ident] = record
    return list(best.values())


def _record_rank(record: Dict[str, Any]) -> int:
    """Higher wins. A graded/ok row beats an errored or empty one."""
    status = record.get("status")
    if status == "ok":
        return 2
    if status in {"skipped", "dry_run"}:
        return 1
    return 0


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------
def threaded_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    workers: int = 4,
    desc: str = "",
    on_result: Optional[Callable[[R], None]] = None,
) -> List[R]:
    """Run `fn` over `items` with a thread pool, streaming results to `on_result`.

    Exceptions inside `fn` are the callee's responsibility; anything that escapes
    is logged and skipped so one bad sample cannot kill a run.
    """
    if not items:
        return []
    results: List[R] = []
    workers = max(1, min(workers, len(items)))
    log = logging.getLogger(__name__)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        with tqdm(total=len(futures), desc=desc, unit="sample", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.error("task failed unexpectedly: %s: %s", type(exc).__name__, exc)
                else:
                    results.append(result)
                    if on_result is not None:
                        on_result(result)
                bar.update(1)
    return results


# --------------------------------------------------------------------------
# JSON extraction (judge output)
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a JSON object out of a model response."""
    if not text:
        return None
    candidates: List[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    for match in _FENCE.finditer(text):
        candidates.append(match.group(1).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "y", "met", "pass", "passed", "1"}:
            return True
        if low in {"false", "no", "n", "unmet", "not met", "fail", "failed", "0"}:
            return False
    return None
