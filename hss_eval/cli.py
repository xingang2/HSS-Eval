"""Command line interface.

    python -m hss_eval inspect-data
    python -m hss_eval list-models
    python -m hss_eval run --models gpt-6-astra --run my-run
    python -m hss_eval generate --models claude-opus-5 --limit 5
    python -m hss_eval judge --models claude-opus-5
    python -m hss_eval report --run my-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__
from .client import LiteLLMClient
from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODELS_YAML,
    DEFAULT_RESULTS_DIR,
    NO_MEDIA_SUFFIX,
    ModelConfig,
    Registry,
    load_credentials,
    load_registry,
)
from .dataset import DEFAULT_DATASET, HSS_DOMAINS, HSS_SUBDOMAINS, dataset_summary, load_dataset
from .generate import GenerationOptions, generate_responses, responses_path
from .judge import JudgeOptions, judge_responses, judgments_path
from .media import MediaCache
from .report import build_report, format_difficulty, format_leaderboard
from .utils import (
    JsonlWriter,
    dedupe_records,
    load_resumable_ids,
    read_jsonl,
    setup_logging,
    trial_key,
    write_json,
)

log = logging.getLogger("hss")


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help="Hugging Face dataset id, local copy of it, or a JSONL file (default: %(default)s)",
    )
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_YAML, help="model registry YAML")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="where extracted video frames are cached")
    parser.add_argument("--env-file", type=Path, default=None, help="defaults to ./.env")
    parser.add_argument("--run", dest="run_name", default=None, help="run name (default: timestamp)")
    parser.add_argument("-v", "--verbose", action="store_true")


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", default="all", help="comma-separated aliases, or 'all'")
    parser.add_argument("--media-kinds", default=None, help="filter: image,video")
    parser.add_argument("--domains", default=None, help="filter: comma-separated domain names")
    parser.add_argument("--sample-ids", default=None, help="filter: comma-separated task ids")
    parser.add_argument("--limit", type=int, default=None, help="keep only the first N samples")
    parser.add_argument(
        "-k", "--attempts", type=int, default=1,
        help="answer each sample K times (default 1). Cost scales linearly with K",
    )
    parser.add_argument(
        "--too-easy-at", type=float, default=0.6,
        help="pass rate at or above which a sample is reported as too easy (default 0.6)",
    )
    parser.add_argument("--workers", type=int, default=4, help="concurrent requests per model")
    parser.add_argument(
        "--model-workers", type=int, default=1,
        help="models evaluated concurrently (default 1). In-flight requests = "
             "--model-workers x --workers",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="re-run completed samples")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="override video.max_frames for every selected model")
    parser.add_argument("--fps", type=float, default=None,
                        help="override video.fps for every selected model")
    parser.add_argument(
        "--effort", default=None,
        help="override reasoning effort for every selected model; models that do "
             "not support the level are skipped",
    )
    parser.add_argument(
        "--system-prompt", default=None,
        help="system message for the answering model: a literal string or @path to a "
             "file. Default: none (and a short neutral prompt with --no-media)",
    )
    parser.add_argument(
        "--no-media", dest="send_media", action="store_false",
        help="blind control: send the question without its image/video. Results go "
             "to a separate '<label>+nomedia' file",
    )


def _add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-model", default=None, help="judge alias (default: judge.default in YAML)")
    parser.add_argument("--judge-workers", type=int, default=None, help="defaults to --workers")
    parser.add_argument(
        "--judge-include-media", action="store_true", default=None,
        help="also show the media to the judge (default from YAML: off)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hss", description="Humanity's Sixth Sense evaluation harness")
    parser.add_argument("--version", action="version", version=f"hss_eval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-models", help="show registry entries (and optionally endpoint models)")
    _add_common(p_list)
    p_list.add_argument("--remote", action="store_true", help="also list the endpoint's models")
    p_list.add_argument("--grep", default=None, help="filter remote model ids by substring")

    p_data = sub.add_parser("inspect-data", help="dataset statistics")
    _add_common(p_data)
    p_data.add_argument("--show", type=int, default=0, help="print the first N samples")

    p_gen = sub.add_parser("generate", help="stage 1: collect model answers")
    _add_common(p_gen)
    _add_selection(p_gen)
    p_gen.add_argument("--dry-run", action="store_true", help="prepare media/messages but do not call the API")

    p_judge = sub.add_parser("judge", help="stage 2: grade answers against rubrics")
    _add_common(p_judge)
    _add_selection(p_judge)
    _add_judge_args(p_judge)

    p_status = sub.add_parser("status", help="what is done / pending for a run")
    _add_common(p_status)
    _add_selection(p_status)
    _add_judge_args(p_status)

    p_report = sub.add_parser("report", help="stage 3: aggregate scores")
    _add_common(p_report)
    _add_selection(p_report)
    _add_judge_args(p_report)

    p_run = sub.add_parser("run", help="generate + judge + report")
    _add_common(p_run)
    _add_selection(p_run)
    _add_judge_args(p_run)

    p_frames = sub.add_parser("sample-frames", help="debug: run video sampling only, no API calls")
    _add_common(p_frames)
    _add_selection(p_frames)

    return parser


# --------------------------------------------------------------------------
# shared setup
# --------------------------------------------------------------------------
class Context:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.registry: Registry = load_registry(args.models_config)
        self.cache = MediaCache(args.cache_dir)
        self._client: Optional[LiteLLMClient] = None
        self.samples = load_dataset(
            args.dataset,
            media_kinds=_split(getattr(args, "media_kinds", None)),
            domains=_split(getattr(args, "domains", None)),
            limit=getattr(args, "limit", None),
            sample_ids=_split(getattr(args, "sample_ids", None)),
        )
        self.dataset_info = dataset_summary(self.samples)

    def fingerprints(self) -> Dict[str, Dict[str, str]]:
        """Sample id -> content hashes, so results for edited samples are caught."""
        return {s.sample_id: s.fingerprints() for s in self.samples}

    def sample_meta(self) -> Dict[str, Dict[str, str]]:
        """Sample id -> the labels the report groups by."""
        return {
            s.sample_id: {"media_kind": s.media_kind, "domain": s.domain, "subdomain": s.subdomain}
            for s in self.samples
        }

    @property
    def client(self) -> LiteLLMClient:
        if self._client is None:
            creds = load_credentials(self.args.env_file)
            self._client = LiteLLMClient(creds)
            log.info("endpoint: %s", creds.openai_base_url)
        return self._client

    def models(self) -> List[ModelConfig]:
        chosen = self.registry.resolve_selection(getattr(self.args, "models", "all"))
        chosen = [self._apply_video_overrides(cfg) for cfg in chosen]
        chosen = self._apply_effort_override(chosen)
        if not getattr(self.args, "send_media", True):
            log.info("blind control: media will NOT be sent (%d model(s))", len(chosen))
            chosen = [replace(cfg, send_media=False) for cfg in chosen]
        return chosen

    def _apply_effort_override(self, models: List[ModelConfig]) -> List[ModelConfig]:
        """Apply --effort, skipping (and logging) models that do not accept it."""
        effort = getattr(self.args, "effort", None)
        if not effort:
            return models
        kept: List[ModelConfig] = []
        for cfg in models:
            if cfg.supported_efforts and effort not in cfg.supported_efforts:
                log.warning(
                    "%s: skipping -- effort %r not supported (accepts %s)",
                    cfg.name, effort, ", ".join(cfg.supported_efforts),
                )
                continue
            kept.append(replace(cfg, reasoning_effort=effort))
        if not kept:
            raise RuntimeError(f"No selected model supports reasoning effort {effort!r}.")
        log.info("reasoning effort override -> %s (%d model(s))", effort, len(kept))
        return kept

    def _apply_video_overrides(self, cfg: ModelConfig) -> ModelConfig:
        """Apply --max-frames / --fps on top of the registry."""
        max_frames = getattr(self.args, "max_frames", None)
        fps = getattr(self.args, "fps", None)
        if max_frames is None and fps is None:
            return cfg
        video = replace(
            cfg.video,
            max_frames=max_frames if max_frames is not None else cfg.video.max_frames,
            fps=fps if fps is not None else cfg.video.fps,
        )
        log.info("%s: video override -> fps=%g max_frames=%d", cfg.name, video.fps, video.max_frames)
        return replace(cfg, video=video)

    def judge_cfg(self) -> ModelConfig:
        name = getattr(self.args, "judge_model", None) or self.registry.judge_settings.default
        cfg = self.registry.get(name)
        cfg.require_ready()
        return cfg

    def judge_name(self) -> str:
        return getattr(self.args, "judge_model", None) or self.registry.judge_settings.default

    def include_media(self) -> bool:
        override = getattr(self.args, "judge_include_media", None)
        return self.registry.judge_settings.include_media if override is None else bool(override)

    def run_dir(self, create: bool = True) -> Path:
        name = self.args.run_name or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
        path = Path(self.args.results_dir) / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def latest_run_dir(self) -> Path:
        if self.args.run_name:
            path = Path(self.args.results_dir) / self.args.run_name
            if not path.exists():
                raise FileNotFoundError(f"No such run: {path}")
            return path
        candidates = sorted(
            (p for p in Path(self.args.results_dir).glob("*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No runs found under {self.args.results_dir}")
        return candidates[-1]


def _split(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolved_system_prompt(args) -> Optional[str]:
    """--system-prompt as the generator wants it: None = the arm's default."""
    value = getattr(args, "system_prompt", None)
    if value is None:
        return None
    if value.startswith("@"):
        return Path(value[1:]).expanduser().read_text()
    return value


def _snapshot(ctx: Context, run_dir: Path, models: List[ModelConfig], judge: Optional[ModelConfig]) -> None:
    """Record this invocation: run_config.json once, run_history.jsonl every time."""
    entry = {
        "hss_eval_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(ctx.args.dataset),
        "dataset_info": {k: v for k, v in ctx.dataset_info.items() if k != "missing_media"},
        "models_config": str(ctx.args.models_config),
        "models": [
            {
                "name": m.name,
                "model_id": m.model_id,
                "reasoning_effort": m.reasoning_effort if m.supports_reasoning_effort else None,
                "native_video": m.native_video,
                "send_media": m.send_media,
                "video": vars(m.video),
                "image": vars(m.image),
                "max_output_tokens": m.max_output_tokens,
            }
            for m in models
        ],
        "judge": None
        if judge is None
        else {
            "name": judge.name,
            "model_id": judge.model_id,
            "reasoning_effort": judge.reasoning_effort,
            "include_media": ctx.include_media(),
        },
        "system_prompt": _resolved_system_prompt(ctx.args),
        "argv": sys.argv[1:],
    }
    if not (run_dir / "run_config.json").exists():
        write_json(run_dir / "run_config.json", entry)
    JsonlWriter(run_dir / "run_history.jsonl").write(entry)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_list_models(ctx: Context) -> int:
    print(f"Registry: {ctx.registry.source}\n")
    print(f"{'alias':22} {'enabled':8} {'native_video':13} {'effort':7} model_id")
    print("-" * 100)
    for cfg in ctx.registry.models.values():
        print(
            f"{cfg.name:22} {str(cfg.enabled):8} {str(cfg.native_video):13} "
            f"{str(cfg.reasoning_effort or '-'):7} {cfg.model_id or '(unset)'}"
        )
    print("\nJudges:")
    for cfg in ctx.registry.judges.values():
        marker = " (default)" if cfg.name == ctx.registry.judge_settings.default else ""
        print(f"  {cfg.name:22} {cfg.model_id}{marker}")

    if ctx.args.remote:
        print("\nEndpoint models:")
        ids = ctx.client.list_remote_models()
        needle = (ctx.args.grep or "").lower()
        shown = [i for i in ids if needle in i.lower()] if needle else ids
        for model_id in shown:
            print(f"  {model_id}")
        print(f"  ({len(shown)} of {len(ids)} shown)")
    return 0


def cmd_inspect_data(ctx: Context) -> int:
    info = ctx.dataset_info
    print(f"Dataset: {ctx.args.dataset}")
    print(f"  samples        : {info['num_samples']}")
    print(f"  rubric criteria: {info['num_criteria']}")
    print(f"  by media kind  : {info['by_media_kind']}")
    print("\n  by domain / subdomain:")
    for domain in HSS_DOMAINS:
        print(f"      {info['by_domain'].get(domain, 0):4d}  {domain}")
        for subdomain in HSS_SUBDOMAINS[domain]:
            count = info["by_subdomain"].get(f"{domain} / {subdomain}", 0)
            print(f"            {count:4d}  {subdomain}")
    if info["missing_media"]:
        print(f"\n  WARNING {len(info['missing_media'])} sample(s) have no media file on disk: "
              f"{info['missing_media'][:10]}")

    for sample in ctx.samples[: ctx.args.show]:
        print("\n" + "=" * 90)
        print(f"{sample.sample_id}  [{sample.media_kind}] {sample.domain} / {sample.subdomain}")
        print(f"  prompt: {sample.prompt}")
        print(f"  media : {sample.media_path}")
        for crit in sample.criteria:
            print(f"    - ({crit.id}) {crit.title}")
    return 0


def cmd_generate(ctx: Context) -> int:
    models = ctx.models()
    run_dir = ctx.run_dir()
    _snapshot(ctx, run_dir, models, None)
    log.info("run dir: %s", run_dir)
    log.info("%d samples x %d model(s)", len(ctx.samples), len(models))

    options = GenerationOptions(
        workers=ctx.args.workers,
        resume=ctx.args.resume,
        dry_run=getattr(ctx.args, "dry_run", False),
        attempts=getattr(ctx.args, "attempts", 1),
        system_prompt=_resolved_system_prompt(ctx.args),
    )
    client = None if options.dry_run else ctx.client  # a dry run needs no credentials
    for cfg in models:
        path = generate_responses(ctx.samples, cfg, client, ctx.cache, run_dir, options)
        log.info("%s -> %s", cfg.name, path)
    print(run_dir)
    return 0


def cmd_judge(ctx: Context) -> int:
    models = ctx.models()
    run_dir = ctx.latest_run_dir()
    judge_cfg = ctx.judge_cfg()
    log.info("judging run %s with %s", run_dir.name, judge_cfg.name)

    options = JudgeOptions(
        workers=ctx.args.judge_workers or ctx.args.workers,
        resume=ctx.args.resume,
        include_media=ctx.include_media(),
    )
    for cfg in models:
        resp_path = responses_path(run_dir, cfg.result_label)
        records = read_jsonl(resp_path)
        if not records:
            log.warning("%s: no responses at %s -- run `generate` first", cfg.name, resp_path)
            continue
        path = judge_responses(
            ctx.samples, records, judge_cfg, ctx.client, run_dir, cfg.result_label,
            ctx.cache, options
        )
        log.info("%s -> %s", cfg.name, path)
    print(run_dir)
    return 0


def cmd_report(ctx: Context) -> int:
    run_dir = ctx.latest_run_dir()
    judge_name = ctx.judge_name()

    judgment_files: Dict[str, Path] = {}
    response_files: Dict[str, Path] = {}
    display_names: Dict[str, str] = {}
    for cfg in _report_models(ctx, run_dir):
        label = cfg.result_label
        jpath = judgments_path(run_dir, label, judge_name)
        if not jpath.exists():
            continue
        judgment_files[label] = jpath
        response_files[label] = responses_path(run_dir, label)
        display_names[label] = cfg.label()
    _disambiguate(display_names)

    if not judgment_files:
        log.error("no judgments by %s found under %s/judgments", judge_name, run_dir)
        return 1

    summary = build_report(
        run_dir, judgment_files, response_files, ctx.dataset_info, display_names,
        scope_ids={s.sample_id for s in ctx.samples}, fingerprints=ctx.fingerprints(),
        too_easy_at=getattr(ctx.args, "too_easy_at", 0.6),
        sample_meta=ctx.sample_meta(),
    )
    print(format_leaderboard(summary))
    print(format_difficulty(summary, getattr(ctx.args, "too_easy_at", 0.6)))
    log.info("wrote %s", summary["paths"]["summary_json"])
    return 0


def _disambiguate(display_names: Dict[str, str]) -> None:
    """Spell out the effort only when one alias appears at two efforts."""
    counts = Counter(display_names.values())
    for label, name in display_names.items():
        if counts[name] > 1:
            _, _, effort = label.partition("@")
            display_names[label] = f"{name} ({effort or 'default'})"


def _report_models(ctx: Context, run_dir: Path) -> List[ModelConfig]:
    """Models to include in a report: the selection, or whatever the run holds."""
    selection = getattr(ctx.args, "models", "all")
    if selection and selection.strip().lower() not in {"all", "*"}:
        return ctx.registry.resolve_selection(selection)
    found: List[ModelConfig] = []
    for path in sorted((run_dir / "responses").glob("*.jsonl")):
        stem = path.stem
        blind = stem.endswith(NO_MEDIA_SUFFIX)
        alias = ModelConfig.alias_of(stem)
        effort = stem.removesuffix(NO_MEDIA_SUFFIX).partition("@")[2] or None
        try:
            # The filename records the condition it was produced under, so it
            # wins over the registry's current effort setting.
            found.append(replace(ctx.registry.get(alias), reasoning_effort=effort,
                                 send_media=not blind))
        except KeyError:
            # Not a registry alias: report it under its own label.
            found.append(ModelConfig(name=stem, model_id=stem, display_name=stem,
                                     reasoning_effort=None))
    return found or ctx.registry.enabled_models()


def cmd_status(ctx: Context) -> int:
    """Per model: answered / judged / pending / changed / errored / orphan trials."""
    run_dir = ctx.latest_run_dir()
    fingerprints = ctx.fingerprints()
    judge_name = ctx.judge_name()
    attempts = max(1, int(getattr(ctx.args, "attempts", 1)))
    scope = {f"{s.sample_id}#{n}" for s in ctx.samples for n in range(1, attempts + 1)}
    sample_ids = {s.sample_id for s in ctx.samples}

    print(f"run     : {run_dir}")
    print(f"dataset : {ctx.args.dataset} ({len(sample_ids)} samples in scope"
          + (f" x {attempts} attempts = {len(scope)} trials" if attempts > 1 else "") + ")")
    print(f"judge   : {judge_name}\n")
    print(
        f"{'model':28} {'answered':>9} {'judged':>7} {'pending':>8} {'changed':>8} "
        f"{'errored':>8} {'orphan':>7}"
    )
    print("-" * 80)

    models = _report_models(ctx, run_dir)
    if not models:
        log.warning("no per-model result files under %s", run_dir)
        return 1

    total_pending = total_changed = 0
    for cfg in models:
        resp_path = responses_path(run_dir, cfg.result_label)
        judge_path = judgments_path(run_dir, cfg.result_label, judge_name)
        responses = dedupe_records(read_jsonl(resp_path))

        answered, regen = load_resumable_ids(resp_path, fingerprints, keys=("input_hash",))
        judged, rejudge = load_resumable_ids(
            judge_path, fingerprints, keys=("input_hash", "rubric_hash")
        )
        answered &= scope
        judged &= scope
        errored = {
            trial_key(r) for r in responses if r.get("status") not in {"ok", "dry_run"}
        } & scope
        orphan = {r["sample_id"] for r in responses} - sample_ids
        pending = scope - answered - regen
        changed = (regen | rejudge) & scope
        total_pending += len(pending)
        total_changed += len(changed)
        print(
            f"{cfg.result_label:28} {len(answered):>9} {len(judged):>7} {len(pending):>8} "
            f"{len(changed):>8} {len(errored):>8} {len(orphan):>7}"
        )

    print()
    if total_pending or total_changed:
        print(
            f"{total_pending} pending and {total_changed} changed trial(s). Fill them in with:\n"
            f"  python -m hss_eval run --run {run_dir.name} -k {attempts} --models "
            f"{','.join(m.name for m in models)}"
        )
    else:
        print("Everything in scope has been answered and judged for these models.")
    return 0


def cmd_run(ctx: Context) -> int:
    models = ctx.models()
    judge_cfg = ctx.judge_cfg()
    run_dir = ctx.run_dir()
    _snapshot(ctx, run_dir, models, judge_cfg)
    log.info("run dir: %s", run_dir)
    log.info(
        "%d samples (%s) x %d model(s), judge=%s",
        len(ctx.samples), ctx.dataset_info["by_media_kind"], len(models), judge_cfg.name,
    )

    gen_options = GenerationOptions(
        workers=ctx.args.workers,
        resume=ctx.args.resume,
        attempts=getattr(ctx.args, "attempts", 1),
        system_prompt=_resolved_system_prompt(ctx.args),
    )
    judge_options = JudgeOptions(
        workers=ctx.args.judge_workers or ctx.args.workers,
        resume=ctx.args.resume,
        include_media=ctx.include_media(),
    )

    judgment_files: Dict[str, Path] = {}
    response_files: Dict[str, Path] = {}
    display_names: Dict[str, str] = {}

    def evaluate(cfg: ModelConfig):
        """Generate then judge one model. Each writes only its own files."""
        resp_path = generate_responses(ctx.samples, cfg, ctx.client, ctx.cache, run_dir, gen_options)
        records = read_jsonl(resp_path)
        jpath = judge_responses(
            ctx.samples, records, judge_cfg, ctx.client, run_dir, cfg.result_label,
            ctx.cache, judge_options
        )
        return cfg, resp_path, jpath

    model_workers = max(1, min(getattr(ctx.args, "model_workers", 1), len(models)))
    with ThreadPoolExecutor(max_workers=model_workers) as pool:
        for future in as_completed([pool.submit(evaluate, cfg) for cfg in models]):
            try:
                cfg, resp_path, jpath = future.result()
            except Exception as exc:  # noqa: BLE001 - one model must not sink the run
                log.error("model failed: %s: %s", type(exc).__name__, exc)
                continue
            judgment_files[cfg.result_label] = jpath
            response_files[cfg.result_label] = resp_path
            display_names[cfg.result_label] = cfg.label()

    if not judgment_files:
        log.error("every model failed; nothing to report")
        return 1

    summary = build_report(
        run_dir, judgment_files, response_files, ctx.dataset_info, display_names,
        scope_ids={s.sample_id for s in ctx.samples},
        fingerprints=ctx.fingerprints(),
        too_easy_at=getattr(ctx.args, "too_easy_at", 0.6),
        sample_meta=ctx.sample_meta(),
    )
    print()
    print(format_leaderboard(summary))
    print(format_difficulty(summary, getattr(ctx.args, "too_easy_at", 0.6)))
    log.info("artifacts: %s", run_dir)
    return 0


def cmd_sample_frames(ctx: Context) -> int:
    """Exercise the video pipeline without touching the API."""
    from .media.video import frames_tag, sample_frames

    cfg = ctx.models()[0]
    videos = [s for s in ctx.samples if s.is_video]
    log.info("sampling %d video(s) with fps=%s max_frames=%d", len(videos), cfg.video.fps, cfg.video.max_frames)
    total = 0
    for sample in videos:
        media = ctx.cache.fetch(sample.media_file, media_kind="video")
        frames = sample_frames(media.path, ctx.cache.frames_dir(media.path, frames_tag(cfg.video)), cfg.video)
        total += frames.num_frames
        print(
            f"{sample.sample_id}  {frames.info.duration:7.2f}s  "
            f"{frames.num_frames:4d} frames @ {frames.sampled_fps:.4g} fps"
            f"{'  [capped]' if frames.truncated_by_cap else ''}"
        )
    print(f"\ntotal frames: {total} | cache: {ctx.cache.disk_usage_mb():.1f} MiB")
    return 0


COMMANDS = {
    "list-models": cmd_list_models,
    "inspect-data": cmd_inspect_data,
    "generate": cmd_generate,
    "judge": cmd_judge,
    "report": cmd_report,
    "status": cmd_status,
    "run": cmd_run,
    "sample-frames": cmd_sample_frames,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        ctx = Context(args)
        return COMMANDS[args.command](ctx)
    except KeyboardInterrupt:
        log.warning("interrupted; partial results are on disk (re-run to resume)")
        return 130
    except Exception as exc:  # noqa: BLE001
        if args.verbose:
            raise
        log.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
