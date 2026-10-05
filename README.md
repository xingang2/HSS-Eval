# Humanity's Sixth Sense (HSS) — Evaluation Harness

[Dataset](https://huggingface.co/datasets/ScaleAI/HSS) · [Leaderboard](https://scale.com/leaderboard/hss)

**Humanity's Sixth Sense (HSS)** is a benchmark for *intuitive visual reasoning*:
the implicit temporal, spatial, social and abstract structure that people infer
from an image or a short video at a glance. It has 522 tasks (288 images, 234
videos) across 4 domains and 11 subdomains. Each task pairs one image or video
with a question, a human-written reference answer, and a rubric of
independently checkable criteria.

This repository is the harness used to evaluate models on HSS:

```
image/video + question ──► model under evaluation ──► answer ──► rubric judge ──► scores
```

## Contents

- [Setup](#setup)
- [Quick start](#quick-start)
- [Evaluation protocol](#evaluation-protocol)
- [Outputs](#outputs)
- [Models](#models)
- [Command reference](#command-reference)
- [Repository layout](#repository-layout)
- [Citation](#citation)

## Setup

**Requirements:** Python 3.10+ on Linux or macOS, and `ffmpeg` / `ffprobe` on
`PATH` (used to sample video frames).

```bash
pip install -r requirements.txt
```

**Dataset.** The benchmark is hosted on the Hugging Face Hub as
[`ScaleAI/HSS`](https://huggingface.co/datasets/ScaleAI/HSS) (about 10.5 GB with
media). It is downloaded and cached automatically the first time you run a
command. To use a copy you already have, pass its directory with
`--dataset /path/to/HSS`.

**Model API.** Every model is called through one OpenAI-compatible endpoint. We
recommend a [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy), which
puts all providers behind a single API. Copy `.env.example` to `.env` and fill
it in:

```bash
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=sk-...
```

`OPENAI_BASE_URL` / `OPENAI_API_KEY` are accepted as well. Each model's
`model_id` in [`configs/models.yaml`](configs/models.yaml) must match the name
your endpoint uses for that model.

## Quick start

```bash
# Dataset statistics (downloads the dataset on first use)
python -m hss_eval inspect-data

# The models in the registry, and the ones your endpoint serves
python -m hss_eval list-models --remote

# Prepare requests without calling any API (checks media and video sampling)
python -m hss_eval generate --models gpt-6-astra --limit 5 --dry-run

# Smoke test: 5 tasks, one model, answered, judged and scored
python -m hss_eval run --models gemini-3.8-flash --limit 5 --run smoke

# Full evaluation, as in the paper: every task, 3 attempts each
python -m hss_eval run --models gpt-6-astra -k 3 --run hss --workers 16
```

Runs resume: re-running the same command with the same `--run` only performs
the work that is missing or failed.

## Evaluation protocol

**Input.** Each task is sent as a single user turn holding a one-line media
preamble, the media, and the question verbatim. No system prompt is used.

**Images** are sent as base64 at native resolution. They are downscaled only
when they exceed a payload limit (longest side 4096 px, 4.5 MB).

**Videos** are handled in one of two ways, set per model in the registry:

- *Native video* (`native_video: true`, e.g. Gemini): the whole file is sent as
  one block, falling back to frames above the size limit.
- *Frame sampling* (all other models): frames are sampled uniformly at 2 fps,
  capped at 500 frames. For longer clips the rate is lowered so the frames still
  span the whole video. Frames are scaled to a longest side of 768 px and
  labelled with their timestamps. Endpoints with a lower image limit receive a
  uniformly thinned subset.

**Reasoning effort.** Each model runs at the effort level set for it in the
registry: the one used in the paper, typically the highest its API accepts.

**Judging.** A text-only LLM judge (Claude Opus 5, `judge-claude-opus-5`) sees
the question, the reference answer, the rubric criteria and the model's answer.
It marks each criterion as met or not met. The judge never sees the media; the
rubrics are written to be checkable from the reference answer.

**Scoring.** An attempt counts as correct only if **every** rubric criterion is
met. The headline metric is **pass@1**: the per-task mean over attempts,
averaged over all 522 tasks. The paper uses 3 attempts per task (`-k 3`). An
attempt with no answer (an API failure or an empty response) counts as
incorrect.

**Blind control.** `--no-media` sends the question without its image or video,
using a short neutral system prompt that lets the model decline. It gives the
score reachable from the text alone. Results are written to a separate
`<label>+nomedia` file.

## Outputs

Each run writes to `results/<run>/`:

```
results/<run>/
├── run_config.json                    # settings of the first invocation
├── run_history.jsonl                  # one entry per invocation
├── responses/<model>@<effort>.jsonl   # one row per (task, attempt): answer, tokens, media details
├── judgments/<model>@<effort>__by__<judge>.jsonl   # per-criterion verdicts
└── reports/summary.json               # scores, breakdowns, per-task results
```

The `report` command prints a leaderboard for the run:

| column | meaning |
|---|---|
| `accuracy` | pass@1: fraction of attempts with every criterion met |
| `rubric` | mean fraction of criteria met (partial credit; diagnostic only) |
| `pass@k` | fraction of tasks solved by at least one of the k attempts |
| `out-tok` / `think-tok` / `total-tok` | mean tokens per attempt. `NA` means the endpoint does not report reasoning tokens |
| `coverage` | tasks evaluated / tasks in scope |
| `no-answer` | tasks with a failed or empty response, scored as incorrect |

`summary.json` also holds breakdowns by media type, domain and subdomain.

## Models

[`configs/models.yaml`](configs/models.yaml) lists the 24 models from the paper
and the judges. A `defaults` block is merged into every entry. The main fields
are:

| field | meaning |
|---|---|
| `model_id` | model name on your endpoint |
| `api` | `chat` (`/chat/completions`), `responses` (OpenAI Responses API) or `messages` (Anthropic Messages API) |
| `reasoning_effort` / `supported_efforts` | effort sent with each request, and the levels the model accepts |
| `native_video` | send whole video files instead of sampled frames |
| `max_output_tokens` | output budget, including reasoning tokens |
| `video.*` | frame rate, frame cap, frame size and per-request image limits |
| `extra_body` | additional provider-specific request fields |

To evaluate a new model, add an entry under `models` with at least `model_id`,
`display_name` and `reasoning_effort`, then pass its alias to `--models`.

## Command reference

`python -m hss_eval <command> [options]`

| command | what it does |
|---|---|
| `inspect-data` | dataset statistics; `--show N` prints the first N tasks |
| `list-models` | registry entries; `--remote` also lists the endpoint's models |
| `generate` | collect model answers (`--dry-run` builds requests without calling the API) |
| `judge` | grade the answers in a run |
| `report` | score a run and write `reports/summary.json` |
| `run` | `generate` + `judge` + `report` |
| `status` | per model: answered, judged, pending and errored trials |
| `sample-frames` | run video frame sampling only, without API calls |

Common options:

| option | meaning |
|---|---|
| `--dataset` | Hub id (default `ScaleAI/HSS`), local copy, or JSONL file |
| `--models` | comma-separated aliases, or `all` |
| `--run` | run name under `results/` (default: a timestamp) |
| `-k` | attempts per task |
| `--workers` / `--model-workers` | concurrent requests per model / models run in parallel |
| `--media-kinds`, `--domains`, `--sample-ids`, `--limit` | evaluate a subset |
| `--judge-model`, `--judge-workers` | judge alias and its concurrency |
| `--effort` | override reasoning effort (models that do not accept it are skipped) |
| `--max-frames`, `--fps` | override video frame sampling |
| `--system-prompt` | add a system prompt: a literal string, or `@file` |
| `--no-media` | blind control (question only) |
| `--no-resume` | redo work that is already complete |

## Repository layout

```
hss_eval/
  cli.py        command-line entry point
  dataset.py    loads the benchmark into tasks with parsed rubrics
  messages.py   builds requests: media preamble, image/video blocks, question
  media/        image encoding and uniform video frame sampling
  client.py     OpenAI-compatible client with retries (chat, Responses and Messages APIs)
  generate.py   stage 1: answers
  judge.py      stage 2: per-criterion grading
  report.py     stage 3: scores, breakdowns, token usage
  prompts.py    media preambles and the judge prompt
configs/models.yaml   model and judge registry
tests/                offline unit tests (no API calls)
```

Run the tests with `python -m pytest tests -q`.

## Citation

```bibtex
@misc{guo2026hss,
  title  = {Humanity's Sixth Sense: Benchmarking Intuitive Visual Reasoning in Multimodal Models},
  author = {Xingang Guo and Jing Gu and Brian Jang and Renxiong Wang and Utkarsh Tyagi and Daniel Quigley and Steven Li and David Yan and Daniel Yue Zhang and Darvin Yi and Forrest Huang and HiJae Kim and Tianyi Zhang and Jared Lichtarge and Jihua Huang and Le Xue and Manan Tomar and Qiuyi Richard Zhang and Ruofei Yu and Seth Neel and Yaning Hu and Marcella Valentine and Daniel Evans and Chenguang Wang and Dustin Tran and Tong Zhao and Yinfei Yang and Yunzhong He},
  year   = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). See the
[dataset card](https://huggingface.co/datasets/ScaleAI/HSS) for the dataset's
terms, including its media notice.
