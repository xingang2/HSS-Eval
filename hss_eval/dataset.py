"""Dataset loading: the HSS release -> `Sample` objects with parsed rubrics.

The benchmark is distributed on the Hugging Face Hub as `data/test.jsonl` plus a
`media/` folder of images and videos. `--dataset` accepts any of:

* a Hub dataset id, e.g. `ScaleAI/HSS` (downloaded once into the HF cache);
* a local copy of that repo (a directory containing `data/test.jsonl`);
* a JSONL file in the same format, whose `media_path` values are relative to
  the repo root (the parent of the file's `data/` directory).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

DEFAULT_DATASET = "ScaleAI/HSS"
DATA_FILE = Path("data") / "test.jsonl"

# The HSS taxonomy: four domains, each with its own subdomains. Every subdomain
# belongs to exactly one domain.
HSS_SUBDOMAINS: Dict[str, List[str]] = {
    "Temporal & Causal Dynamics": [
        "Retrodiction",
        "Mechanistic Causality",
        "Extrapolation",
    ],
    "Physical & Spatial Logic": [
        "Hidden & Invisible",
        "Affordance & Feasibility",
        'Spatial "Alien Viewpoint"',
        "Spatial Reachability",
    ],
    "Social Understanding": [
        "Theory of Mind",
        "Social Role, Norm & Power Dynamics",
    ],
    "Abstract & Contextual Inference": [
        "Change & Consequence",
        "Patterns & Pareidolia",
    ],
}
HSS_DOMAINS = list(HSS_SUBDOMAINS)


@dataclass
class Criterion:
    id: str
    title: str
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "weight": self.weight}


@dataclass
class Sample:
    sample_id: str
    row_index: int
    prompt: str
    media_path: str          # as written in the dataset, relative to its root
    media_file: Path         # resolved absolute path on disk
    media_kind: str          # "image" | "video"
    domain: str
    subdomain: str
    golden_response: str
    criteria: List[Criterion] = field(default_factory=list)
    # Content fingerprints: what the model is shown, and what the judge grades
    # against. They let a resumed run tell an unchanged sample from an edited one.
    input_hash: str = ""
    rubric_hash: str = ""

    @property
    def is_video(self) -> bool:
        return self.media_kind == "video"

    def fingerprints(self) -> Dict[str, str]:
        return {"input_hash": self.input_hash, "rubric_hash": self.rubric_hash}


def _digest(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()[:12]


def input_fingerprint(prompt: str, media_path: str) -> str:
    """What the model under evaluation is shown. A change invalidates the answer."""
    return _digest(prompt.strip(), media_path.strip())


def rubric_fingerprint(golden_response: str, criteria: List[Criterion]) -> str:
    """What the judge grades against. A change invalidates the judgment only."""
    payload = [golden_response.strip()]
    payload += [f"{c.id}\x01{c.title}\x01{c.weight}" for c in criteria]
    return _digest(*payload)


def parse_criteria(raw: Any) -> List[Criterion]:
    """Parse `rubric_criteria`: a list of `{"id", "title"}` objects (or its JSON)."""
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else []
    out: List[Criterion] = []
    for i, item in enumerate(raw or [], start=1):
        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            cid = str(item.get("id") or f"c{i}")
            weight = float(item.get("weight", 1.0) or 1.0)
        else:
            title, cid, weight = str(item).strip(), f"c{i}", 1.0
        if title:
            out.append(Criterion(id=cid, title=title, weight=weight))
    return out


def resolve_dataset(source: str | Path) -> Path:
    """Path to `data/test.jsonl` for a Hub id, a local repo copy, or a JSONL file."""
    path = Path(source).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / DATA_FILE
        if not candidate.is_file():
            raise FileNotFoundError(f"{path} has no {DATA_FILE}")
        return candidate
    source = str(source)
    if source.count("/") != 1:
        raise FileNotFoundError(f"No such dataset file or directory: {source}")
    from huggingface_hub import snapshot_download

    log.info("fetching %s from the Hugging Face Hub (cached after the first run)", source)
    root = Path(snapshot_download(source, repo_type="dataset"))
    return root / DATA_FILE


def load_dataset(
    source: str | Path = DEFAULT_DATASET,
    media_kinds: Optional[Iterable[str]] = None,
    domains: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    sample_ids: Optional[Iterable[str]] = None,
) -> List[Sample]:
    """Read the benchmark and apply optional filters."""
    data_file = resolve_dataset(source)
    root = data_file.parent.parent
    samples: List[Sample] = []
    seen: set[str] = set()
    with open(data_file, encoding="utf-8") as fh:
        for row_index, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["task_id"])
            if sample_id in seen:
                raise ValueError(f"{data_file}: duplicate task_id {sample_id}")
            seen.add(sample_id)
            criteria = parse_criteria(row.get("rubric_criteria"))
            media_path = str(row["media_path"])
            golden = str(row.get("golden_response") or "")
            prompt = str(row["prompt"])
            samples.append(
                Sample(
                    sample_id=sample_id,
                    row_index=row_index,
                    prompt=prompt,
                    media_path=media_path,
                    media_file=root / media_path,
                    media_kind=str(row["media_type"]).strip().lower(),
                    domain=str(row.get("domain") or ""),
                    subdomain=str(row.get("subdomain") or ""),
                    golden_response=golden,
                    criteria=criteria,
                    input_hash=input_fingerprint(prompt, media_path),
                    rubric_hash=rubric_fingerprint(golden, criteria),
                )
            )

    if media_kinds:
        wanted = {k.strip().lower() for k in media_kinds}
        samples = [s for s in samples if s.media_kind in wanted]
    if domains:
        wanted_domains = {d.strip().lower() for d in domains if d.strip()}
        samples = [s for s in samples if s.domain.lower() in wanted_domains]
    if sample_ids:
        wanted_ids = {s.strip() for s in sample_ids if s.strip()}
        samples = [s for s in samples if s.sample_id in wanted_ids]
    if limit is not None:
        samples = samples[:limit]
    return samples


def dataset_summary(samples: List[Sample]) -> Dict[str, Any]:
    return {
        "num_samples": len(samples),
        "num_criteria": sum(len(s.criteria) for s in samples),
        "by_media_kind": dict(Counter(s.media_kind for s in samples)),
        "by_domain": dict(Counter(s.domain for s in samples)),
        "by_subdomain": dict(Counter(f"{s.domain} / {s.subdomain}" for s in samples)),
        "missing_media": [s.sample_id for s in samples if not s.media_file.is_file()],
    }
