"""Offline unit tests (no API, no ffmpeg): `python -m pytest tests -q`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hss_eval.config import DEFAULT_MODELS_YAML, ModelConfig, load_registry  # noqa: E402
from hss_eval.dataset import (  # noqa: E402
    HSS_DOMAINS,
    HSS_SUBDOMAINS,
    Criterion,
    Sample,
    load_dataset,
    parse_criteria,
)
from hss_eval.judge import _score, answer_fingerprint  # noqa: E402
from hss_eval.media.video import _uniform_indices, plan_sampling  # noqa: E402
from hss_eval.messages import effective_system_prompt  # noqa: E402
from hss_eval.prompts import BLIND_SYSTEM_PROMPT, system_prompt_fingerprint  # noqa: E402
from hss_eval.utils import (  # noqa: E402
    coerce_bool,
    dedupe_records,
    extract_json_object,
    load_resumable_ids,
)


def _write_jsonl(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# video sampling plan: uniform fps=2, cap 500
# --------------------------------------------------------------------------
def test_short_video_samples_at_two_fps():
    assert plan_sampling(duration=12.04, fps=2.0, max_frames=500) == (2.0, 24, False)


def test_exactly_at_cap_is_not_capped():
    assert plan_sampling(duration=250.0, fps=2.0, max_frames=500) == (2.0, 500, False)


def test_long_video_lowers_fps_instead_of_truncating():
    fps, frames, capped = plan_sampling(duration=600.0, fps=2.0, max_frames=500)
    assert frames == 500 and capped and fps < 2.0
    # Frames must still span the whole clip.
    assert abs(frames / fps - 600.0) < 1e-6


def test_zero_duration_yields_one_frame():
    assert plan_sampling(duration=0.0, fps=2.0, max_frames=500)[1] == 1


def test_uniform_indices_span_endpoints():
    idx = _uniform_indices(302, 50)
    assert len(idx) == 50 and idx[0] == 0 and idx[-1] == 301 and idx == sorted(idx)


def test_uniform_indices_noop_when_under_limit():
    assert _uniform_indices(10, 50) == list(range(10))


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def _row(task_id="t1", prompt="q", media_path="media/images/t1.png", golden="g",
         criteria=None, media_type="image"):
    return {
        "task_id": task_id, "domain": "Social Understanding", "subdomain": "Theory of Mind",
        "media_type": media_type, "media_path": media_path, "prompt": prompt,
        "golden_response": golden,
        "rubric_criteria": criteria if criteria is not None else [{"id": "c1", "title": "states x"}],
        "num_criteria": 1,
    }


def _dataset(root: Path, rows) -> Path:
    _write_jsonl(root / "data" / "test.jsonl", rows)
    return root


def test_parse_criteria_from_list_and_json():
    items = [{"id": "e189", "title": "States the boat moves right to left."},
             {"id": "8c7e", "title": "States the red bicycle is closest."}]
    assert [c.id for c in parse_criteria(items)] == ["e189", "8c7e"]
    assert [c.id for c in parse_criteria(json.dumps(items))] == ["e189", "8c7e"]
    assert parse_criteria([]) == [] and parse_criteria("") == []


def test_dataset_loads_from_a_local_copy(tmp_path):
    root = _dataset(tmp_path, [_row("a"), _row("b", media_type="video",
                                                media_path="media/videos/b.mp4")])
    samples = load_dataset(root)
    assert [s.sample_id for s in samples] == ["a", "b"]
    assert samples[0].media_file == tmp_path / "media/images/t1.png"
    assert samples[1].is_video
    # A JSONL path works as well as the repo directory.
    assert len(load_dataset(root / "data" / "test.jsonl")) == 2


def test_dataset_filters(tmp_path):
    root = _dataset(tmp_path, [_row("a"), _row("b", media_type="video")])
    assert [s.sample_id for s in load_dataset(root, media_kinds=["video"])] == ["b"]
    assert [s.sample_id for s in load_dataset(root, sample_ids=["a"])] == ["a"]
    assert len(load_dataset(root, limit=1)) == 1
    assert len(load_dataset(root, domains=["social understanding"])) == 2


def test_duplicate_task_ids_are_an_error(tmp_path):
    root = _dataset(tmp_path, [_row("a"), _row("a")])
    try:
        load_dataset(root)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_editing_a_prompt_changes_only_the_input_hash(tmp_path):
    before = load_dataset(_dataset(tmp_path / "1", [_row(prompt="what next?")]))[0]
    after = load_dataset(_dataset(tmp_path / "2", [_row(prompt="What next, and why?")]))[0]
    assert before.sample_id == after.sample_id
    assert before.input_hash != after.input_hash
    assert before.rubric_hash == after.rubric_hash


def test_editing_a_rubric_changes_only_the_rubric_hash(tmp_path):
    before = load_dataset(_dataset(tmp_path / "1", [_row()]))[0]
    after = load_dataset(_dataset(tmp_path / "2", [_row(criteria=[
        {"id": "c1", "title": "states x"}, {"id": "c2", "title": "states y"}])]))[0]
    golden = load_dataset(_dataset(tmp_path / "3", [_row(golden="another answer")]))[0]
    assert before.input_hash == after.input_hash == golden.input_hash
    assert before.rubric_hash != after.rubric_hash
    assert before.rubric_hash != golden.rubric_hash


def test_each_subdomain_belongs_to_exactly_one_domain():
    owners = {}
    for domain, subs in HSS_SUBDOMAINS.items():
        for sub in subs:
            owners.setdefault(sub, []).append(domain)
    assert all(len(d) == 1 for d in owners.values())
    assert len(HSS_DOMAINS) == 4 and len(owners) == 11


# --------------------------------------------------------------------------
# judge scoring
# --------------------------------------------------------------------------
def _sample(n: int) -> Sample:
    return Sample(
        sample_id="hss-test", row_index=0, prompt="q", media_path="m.png",
        media_file=Path("m.png"), media_kind="image", domain="d", subdomain="s",
        golden_response="g",
        criteria=[Criterion(id=f"c{i}", title=f"crit {i}") for i in range(1, n + 1)],
    )


def test_score_partial_credit():
    result = _score(_sample(2), {"criteria": [{"id": "c1", "met": True}, {"id": "c2", "met": False}]})
    assert result["rubric_score"] == 0.5 and result["num_met"] == 1
    assert result["strict_pass"] is False


def test_score_strict_pass():
    result = _score(_sample(2), {"criteria": [{"id": "c1", "met": True}, {"id": "c2", "met": "yes"}]})
    assert result["rubric_score"] == 1.0 and result["strict_pass"] is True


def test_score_falls_back_to_order_when_ids_are_wrong():
    parsed = {"criteria": [{"id": "bogus-1", "met": True}, {"id": "bogus-2", "met": True}]}
    assert _score(_sample(2), parsed)["rubric_score"] == 1.0


def test_score_missing_verdict_counts_as_unmet():
    result = _score(_sample(3), {"criteria": [{"id": "c1", "met": True}]})
    assert result["num_met"] == 1 and abs(result["rubric_score"] - 1 / 3) < 1e-9


def test_score_unparseable_judge_output_is_zero():
    result = _score(_sample(2), None)
    assert result["rubric_score"] == 0.0 and result["num_met"] == 0


def test_extract_json_object():
    text = 'Here you go:\n```json\n{"criteria": [], "overall_comment": "ok"}\n```\n'
    assert extract_json_object(text) == {"criteria": [], "overall_comment": "ok"}
    assert extract_json_object('sure: {"a": 1} -- done')["a"] == 1
    assert extract_json_object("no json at all") is None


def test_coerce_bool():
    assert coerce_bool("Yes") is True
    assert coerce_bool("not met") is False
    assert coerce_bool(None) is None


# --------------------------------------------------------------------------
# system prompt
# --------------------------------------------------------------------------
def test_vision_arm_sends_no_system_message_by_default():
    assert effective_system_prompt(None, include_media=True) is None


def test_blind_arm_keeps_its_own_prompt():
    assert effective_system_prompt(None, include_media=False) == BLIND_SYSTEM_PROMPT
    assert effective_system_prompt("", include_media=False) is None


def test_explicit_system_prompt_wins_on_both_arms():
    for include_media in (True, False):
        assert effective_system_prompt("be brief", include_media) == "be brief"


def test_system_prompt_fingerprint():
    assert system_prompt_fingerprint(None) == system_prompt_fingerprint("") == "none"
    assert system_prompt_fingerprint("be brief") == system_prompt_fingerprint("be brief") != "none"


def test_answers_under_a_different_system_prompt_are_not_reusable(tmp_path):
    path = _write_jsonl(tmp_path / "m.jsonl", [
        {"sample_id": "s1", "attempt": 1, "status": "ok", "input_hash": "i",
         "system_prompt_hash": "none"},
    ])
    reusable, changed = load_resumable_ids(
        path, fingerprints={"s1": {"input_hash": "i", "system_prompt_hash": "abc123"}},
        keys=("input_hash", "system_prompt_hash"),
    )
    assert reusable == set() and changed == {"s1#1"}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def test_registry_holds_the_paper_roster():
    registry = load_registry(DEFAULT_MODELS_YAML)
    assert len(registry.models) == 24
    assert registry.judge_settings.default == "judge-claude-opus-5"
    assert registry.get("gemini-3.8-flash").native_video is True
    assert registry.get("gpt-6-astra").native_video is False
    assert registry.get("gpt-6-astra").video.max_frames == 500


def test_configured_effort_is_one_the_model_accepts():
    for cfg in load_registry(DEFAULT_MODELS_YAML).enabled_models():
        assert cfg.supported_efforts, cfg.name
        assert cfg.reasoning_effort in cfg.supported_efforts, cfg.name


def test_result_label_carries_reasoning_effort():
    cfg = ModelConfig(name="m", model_id="x", reasoning_effort="xhigh")
    assert cfg.result_label == "m@xhigh"
    assert ModelConfig.alias_of(cfg.result_label) == "m"
    assert ModelConfig(name="m", model_id="x", reasoning_effort=None,
                       supports_reasoning_effort=False).result_label == "m"
    blind = ModelConfig(name="m", model_id="x", reasoning_effort="max", send_media=False)
    assert blind.result_label == "m@max+nomedia"
    assert ModelConfig.alias_of(blind.result_label) == "m"


def test_selection_all_returns_only_enabled():
    registry = load_registry(DEFAULT_MODELS_YAML)
    names = {m.name for m in registry.resolve_selection("all")}
    assert names == {m.name for m in registry.models.values() if m.enabled and m.model_id}


def test_selecting_an_unknown_model_raises():
    try:
        load_registry(DEFAULT_MODELS_YAML).resolve_selection("no-such-model")
    except KeyError as exc:
        assert "Unknown model alias" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for an unknown model")


def test_claude_asks_for_a_thinking_summary():
    registry = load_registry(DEFAULT_MODELS_YAML)
    for cfg in registry.models.values():
        if cfg.vendor == "anthropic":
            assert cfg.extra_body.get("thinking") == {"type": "adaptive", "display": "summarized"}


def test_base_url_is_used_verbatim():
    from hss_eval.config import Credentials

    host = "https://llm.example.com"
    assert Credentials(f"{host}/", "k").openai_base_url == host
    assert Credentials(f"{host}/v1/", "k").openai_base_url == f"{host}/v1"


# --------------------------------------------------------------------------
# frame byte budget
# --------------------------------------------------------------------------
def _write_frames(tmpdir, count, side=768, quality=95):
    """Synthetic JPEG frames with enough detail to be non-trivially sized."""
    import random

    from PIL import Image

    paths = []
    rng = random.Random(0)
    for i in range(count):
        img = Image.new("RGB", (side, side))
        img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                     for _ in range(side * side)])
        p = Path(tmpdir) / f"frame_{i:05d}.jpg"
        img.save(p, format="JPEG", quality=quality)
        paths.append(p)
    return paths


class _StubFrames:
    def __init__(self, paths):
        self.paths = paths

    @property
    def num_frames(self):
        return len(self.paths)


def test_size_budget_untouched_when_under(tmp_path):
    from hss_eval.messages import _frame_image_config

    cfg = load_registry(DEFAULT_MODELS_YAML).get("claude-opus-5")
    image_cfg, meta = _frame_image_config(_sample(1), cfg, _StubFrames(_write_frames(tmp_path, 3)))
    assert image_cfg is cfg.image and "per_frame_byte_budget" not in meta


def test_frames_fit_the_budget_after_encoding(tmp_path):
    from dataclasses import replace

    from hss_eval.media.image import encode_image_block
    from hss_eval.messages import _frame_image_config

    base = load_registry(DEFAULT_MODELS_YAML).get("claude-opus-5")
    frames = _StubFrames(_write_frames(tmp_path, 5))
    budget = sum(p.stat().st_size for p in frames.paths) // 3
    cfg = replace(base, video=replace(base.video, max_total_image_bytes=budget))
    image_cfg, meta = _frame_image_config(_sample(1), cfg, frames)
    assert meta["per_frame_byte_budget"] == budget // 5 and image_cfg.min_side <= 128
    total = sum(encode_image_block(p, image_cfg)[1]["encoded_bytes"] for p in frames.paths)
    assert total <= budget


def test_min_side_floor_is_respected(tmp_path):
    from hss_eval.config import ImageConfig
    from hss_eval.media.image import image_data_uri

    path = _write_frames(tmp_path, 1, side=768)[0]
    _, meta = image_data_uri(path, ImageConfig(max_side=768, max_bytes=1, min_side=256))
    assert max(meta["encoded_size"]) == 256


def test_frame_extraction_is_serialised_per_directory(tmp_path):
    import concurrent.futures as cf
    import time

    from hss_eval.media import video as video_mod

    out_dir = tmp_path / "frames"
    out_dir.mkdir()
    overlaps, active = [], []

    def fake_extract():
        active.append(1)
        overlaps.append(len(active))
        time.sleep(0.05)
        for i in range(3):
            (out_dir / f"frame_{i:05d}.jpg").write_bytes(b"\xff\xd8\xff")
        active.pop()

    def go(_):
        with video_mod._extraction_lock(out_dir):
            fake_extract()
        return sorted(out_dir.glob(video_mod.FRAME_GLOB))

    with cf.ThreadPoolExecutor(4) as pool:
        results = [f.result() for f in [pool.submit(go, i) for i in range(4)]]
    assert max(overlaps) == 1 and all(len(r) == 3 for r in results)


# --------------------------------------------------------------------------
# resume and scoring
# --------------------------------------------------------------------------
def _judgment(sample_id, score, strict, status="ok", met=1, attempt=1, **extra):
    return {
        "sample_id": sample_id, "attempt": attempt, "status": status, "rubric_score": score,
        "strict_pass": strict, "num_met": met, "criteria": [{"met": strict}],
        "media_kind": "image", "domain": "d", "subdomain": "s", **extra,
    }


def test_resume_reuses_unchanged_and_re_runs_edited_samples(tmp_path):
    path = _write_jsonl(tmp_path / "r.jsonl", [
        {"sample_id": "a", "status": "ok", "input_hash": "i-a"},
        {"sample_id": "b", "status": "ok", "input_hash": "i-old"},
        {"sample_id": "c", "status": "error", "input_hash": "i-c"},
    ])
    fresh, changed = load_resumable_ids(
        path, {"a": {"input_hash": "i-a"}, "b": {"input_hash": "i-b"}, "c": {"input_hash": "i-c"}},
        keys=("input_hash",),
    )
    assert fresh == {"a#1"} and changed == {"b#1"}   # 'c' errored: not done


def test_resume_takes_the_latest_record_for_a_trial(tmp_path):
    path = _write_jsonl(tmp_path / "r.jsonl", [
        {"sample_id": "a", "status": "ok", "input_hash": "i-old"},
        {"sample_id": "a", "status": "ok", "input_hash": "i-new"},
    ])
    fresh, changed = load_resumable_ids(path, {"a": {"input_hash": "i-new"}}, keys=("input_hash",))
    assert fresh == {"a#1"} and changed == set()


def test_dedupe_prefers_successful_retry_and_keeps_attempts():
    rows = [{"sample_id": "a", "status": "error"},
            {"sample_id": "a", "status": "ok", "answer": "good"},
            {"sample_id": "b", "attempt": 1, "status": "ok"},
            {"sample_id": "b", "attempt": 2, "status": "ok"}]
    out = dedupe_records(rows)
    assert len(out) == 3
    assert next(r for r in out if r["sample_id"] == "a")["answer"] == "good"


def test_retried_sample_is_not_double_counted():
    from hss_eval.report import score_model

    records = [_judgment("a", None, None, status="error"), _judgment("a", 1.0, True),
               _judgment("b", 0.0, False, met=0)]
    s = score_model(records, model="m", scope_ids={"a", "b"})
    assert s.num_samples == 2 and s.num_errors == 0 and s.accuracy == 0.5


def test_samples_outside_the_dataset_are_dropped_and_new_ones_pending():
    from hss_eval.report import score_model

    records = [_judgment("a", 1.0, True), _judgment("gone", 1.0, True), _judgment("b", 0.0, False, met=0)]
    s = score_model(records, model="m", scope_ids={"a", "b", "new"})
    assert s.orphan_ids == ["gone"] and s.missing_ids == ["new"]
    assert s.num_samples == 2 and s.accuracy == 0.5


def test_stale_judgments_are_dropped_and_reported_as_pending():
    from hss_eval.report import score_model

    records = [_judgment("a", 1.0, True, input_hash="i-a", rubric_hash="r-a"),
               _judgment("b", 1.0, True, input_hash="i-b", rubric_hash="r-old")]
    fingerprints = {"a": {"input_hash": "i-a", "rubric_hash": "r-a"},
                    "b": {"input_hash": "i-b", "rubric_hash": "r-new"}}
    s = score_model(records, model="m", scope_ids={"a", "b"}, fingerprints=fingerprints)
    assert s.stale_ids == ["b"] and s.missing_ids == ["b"] and s.accuracy == 1.0


def test_edited_sample_is_replaced_not_counted_twice():
    from hss_eval.report import score_model

    records = [_judgment("X", 0.0, False, input_hash="A", rubric_hash="r"),
               _judgment("X", 0.0, False, attempt=2, input_hash="A", rubric_hash="r"),
               _judgment("X", 1.0, True, input_hash="B", rubric_hash="r")]
    s = score_model(records, model="m", scope_ids={"X"},
                    fingerprints={"X": {"input_hash": "B", "rubric_hash": "r"}})
    assert s.k == 1 and s.num_trials == 1 and s.accuracy == 1.0
    assert s.stale_ids == ["X"]


def test_pass_at_k_and_pass_rate():
    from hss_eval.report import score_model

    records = ([_judgment("a", None, n <= 2, attempt=n) for n in range(1, 6)]
               + [_judgment("b", None, False, attempt=n) for n in range(1, 6)]
               + [_judgment("c", None, True, attempt=n) for n in range(1, 6)])
    for r in records:
        r["rubric_score"] = 1.0 if r["strict_pass"] else 0.0
    s = score_model(records, model="m", scope_ids={"a", "b", "c"})
    assert s.k == 5
    assert abs(s.accuracy - 7 / 15) < 1e-9          # pass@1: mean over attempts
    assert abs(s.pass_at_k - 2 / 3) < 1e-9          # solved at least once
    by_id = {row.sample_id: row for row in s.per_sample}
    assert by_id["a"].pass_rate == 0.4 and by_id["b"].num_passed == 0


def test_difficulty_flags_too_easy_and_never_solved():
    from hss_eval.report import difficulty_summary, score_samples

    def att(sid, n, passed):
        return _judgment(sid, 1.0 if passed else 0.0, passed, attempt=n)

    records = ([att("easy", n, True) for n in range(1, 6)]
               + [att("mid", n, n <= 3) for n in range(1, 6)]
               + [att("hard", n, n == 1) for n in range(1, 6)]
               + [att("never", n, False) for n in range(1, 6)])
    d = difficulty_summary(score_samples(records), too_easy_at=0.6)
    assert set(d["too_easy_sample_ids"]) == {"easy", "mid"}
    assert d["never_solved_sample_ids"] == ["never"] and d["num_flaky"] == 2
    assert d["pass_rate_histogram"] == {"0/5": 1, "1/5": 1, "3/5": 1, "5/5": 1}


def test_item_difficulty_gives_one_vote_per_model_not_per_condition():
    from hss_eval.report import format_difficulty

    def entry(label, accuracy, passed):
        return {"model": label, "accuracy": accuracy, "k": 1,
                "per_sample": [{"sample_id": "s1", "num_passed": passed, "attempts": 1}]}

    summary = {"models": [entry("m@xhigh", 1.0, 1), entry("other@high", 1.0, 1), entry("m", 0.0, 0)]}
    out = format_difficulty(summary)
    assert "too easy   (pass rate >= 60%):    1" in out and "pooled over 2 model(s)" in out


def test_one_alias_at_two_efforts_gets_two_distinct_row_labels():
    from hss_eval.cli import _disambiguate

    names = {"m": "M", "m@xhigh": "M", "other@high": "Other"}
    _disambiguate(names)
    assert names == {"m": "M (default)", "m@xhigh": "M (xhigh)", "other@high": "Other"}


# --------------------------------------------------------------------------
# token accounting
# --------------------------------------------------------------------------
def test_token_usage_counts_thinking_in_output():
    from hss_eval.report import token_usage

    t = token_usage([
        {"response": {"usage": {"completion_tokens": 1099, "prompt_tokens": 1268,
                                "completion_tokens_details": {"reasoning_tokens": 1043}}}},
        {"response": {"usage": {"completion_tokens": 501, "prompt_tokens": 400,
                                "completion_tokens_details": {"reasoning_tokens": 457}}}},
    ])
    assert t["output_tokens_total"] == 1600 and t["reasoning_tokens_total"] == 1500
    assert abs(t["reasoning_share"] - 1500 / 1600) < 1e-4


def test_unreported_reasoning_is_not_reported_as_zero():
    from hss_eval.report import token_usage

    t = token_usage([
        {"response": {"usage": {"completion_tokens": 795, "prompt_tokens": 400,
                                "completion_tokens_details": {"reasoning_tokens": 0}}}},
        {"response": {"usage": {"completion_tokens": 500, "prompt_tokens": 100}}},
    ])
    assert t["reasoning_tokens_reported"] is False and t["reasoning_share"] is None
    assert t["output_tokens_total"] == 1295 and t["total_tokens_total"] == 1795


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
class _Msg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_reasoning_text_read_from_either_provider_shape():
    from hss_eval.client import _reasoning_text

    assert _reasoning_text(_Msg(reasoning_content="step one")) == "step one"
    assert _reasoning_text(_Msg(reasoning_content="", thinking_blocks=[
        {"type": "thinking", "thinking": "a"}, {"type": "thinking", "thinking": "b"}])) == "a\nb"
    assert _reasoning_text(_Msg(thinking_blocks=[{"type": "thinking", "thinking": ""}])) is None
    assert _reasoning_text(None) is None


def test_anthropic_chat_route_reasoning_count_is_discarded():
    from hss_eval.client import _drop_bogus_reasoning_tokens

    usage = {"completion_tokens": 635,
             "completion_tokens_details": {"reasoning_tokens": 87, "text_tokens": 548}}
    _drop_bogus_reasoning_tokens(ModelConfig(name="c", model_id="anthropic/c", vendor="anthropic"), usage)
    assert usage["completion_tokens_details"] == {} and usage["completion_tokens"] == 635
    usage = {"completion_tokens_details": {"reasoning_tokens": 519}}
    _drop_bogus_reasoning_tokens(ModelConfig(name="g", model_id="gemini/g", vendor="google"), usage)
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 519


def test_anthropic_request_uses_the_messages_shape():
    from hss_eval.client import LiteLLMClient
    from hss_eval.config import Credentials

    client = LiteLLMClient(Credentials(base_url="https://x", api_key="k"))
    cfg = ModelConfig(name="claude-opus-5", model_id="anthropic/claude-opus-5",
                      vendor="anthropic", api="messages", reasoning_effort="high",
                      max_output_tokens=4096,
                      extra_body={"thinking": {"type": "adaptive", "display": "summarized"}})
    body = client.build_anthropic_kwargs(cfg, [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        ]},
    ])
    assert body["model"] == "claude-opus-5"
    assert body["system"] == "be brief" and body["max_tokens"] == 4096
    assert body["output_config"] == {"effort": "high"}
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["messages"][0]["content"][1] == {
        "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}


def test_anthropic_reply_maps_onto_the_chat_record():
    from hss_eval.client import LiteLLMClient
    from hss_eval.config import Credentials

    client = LiteLLMClient(Credentials(base_url="https://x", api_key="k"))
    parsed = client._parse_anthropic(
        {"content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "42"}],
         "stop_reason": "end_turn",
         "usage": {"input_tokens": 10, "output_tokens": 613,
                   "output_tokens_details": {"thinking_tokens": 377}}},
        ModelConfig(name="c", model_id="anthropic/c"), started=0.0, attempt=1, dropped=[],
    )
    assert (parsed.text, parsed.reasoning_text, parsed.finish_reason) == ("42", "hmm", "stop")
    assert parsed.usage["completion_tokens_details"]["reasoning_tokens"] == 377

    truncated = client._parse_anthropic(
        {"content": [], "stop_reason": "max_tokens", "usage": {"output_tokens": 4096}},
        ModelConfig(name="m", model_id="anthropic/m"), started=0.0, attempt=1, dropped=[])
    assert truncated.finish_reason == "length"
    assert LiteLLMClient._raise_budget_for_length(
        ModelConfig(name="m", model_id="m", max_output_tokens_ceiling=65536),
        {"max_tokens": 4096}, truncated) is True


def test_provider_overload_is_retryable():
    from hss_eval.client import RETRYABLE_STATUS

    assert 529 in RETRYABLE_STATUS
    assert 400 not in RETRYABLE_STATUS and 401 not in RETRYABLE_STATUS


def test_answer_fingerprint_ignores_surrounding_whitespace():
    assert answer_fingerprint(" 42 ") == answer_fingerprint("42") != answer_fingerprint("43")
