"""Prompt templates for the answering model and the rubric judge."""

from __future__ import annotations

import hashlib
import json
from typing import List, Optional

from .dataset import Criterion, Sample


# --------------------------------------------------------------------------
# Answering model
# --------------------------------------------------------------------------
# The model under evaluation gets no system message by default: one user turn
# holding a short media preamble, the media, and the question verbatim. A custom
# system message can be supplied with `--system-prompt`.

VIDEO_FRAMES_PREAMBLE = (
    "The video below is provided as {num_frames} frames sampled uniformly at {fps:.4g} frames "
    "per second (video duration {duration:.2f}s), in chronological order. "
    "Frame timestamps in seconds: {timestamps}.\n"
    "Treat the frames as one continuous video and reason about motion and event order across "
    "them."
)

VIDEO_NATIVE_PREAMBLE = (
    "A video ({duration:.2f}s) is attached. Reason about motion and event order across time."
)

IMAGE_PREAMBLE = "An image is attached."

# Blind control (`--no-media`): the question alone. Deliberately generic, and it
# permits the model to say it cannot answer, so a refusal is a real measurement
# rather than an artefact of being told to guess.
BLIND_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question faithfully. "
    "If you do not have enough information to answer, do not guess -- say so honestly."
)


def render_answer_instruction(sample: Sample) -> str:
    return f"Question: {sample.prompt}"


def system_prompt_fingerprint(text: Optional[str]) -> str:
    """Stable id for the system message actually sent ("none" when there is none).

    Recorded on every response and compared on resume, so answers produced under
    a different system prompt are never silently reused.
    """
    if not text:
        return "none"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial grader for the Humanity's Sixth Sense (HSS) visual reasoning "
    "benchmark. You are given a question about a piece of media, a human-written reference "
    "answer that is authoritative and correct, a list of rubric criteria, and a candidate "
    "answer from a model under evaluation.\n\n"
    "Grade the candidate answer against each rubric criterion independently.\n\n"
    "Rules:\n"
    "1. A criterion is met only if the candidate answer actually asserts what the criterion "
    "requires. Do not infer or give credit for near-misses, hedges, or contradictions.\n"
    "2. Wording may differ from the reference; judge meaning, not phrasing. Synonyms, "
    "equivalent directions (e.g. 'right to left' vs 'leftward'), and equivalent units count.\n"
    "3. If the candidate answer states the required content but also states a contradicting "
    "alternative, the criterion is NOT met.\n"
    "4. Extra correct detail neither helps nor hurts. Extra incorrect detail only matters if it "
    "contradicts a criterion.\n"
    "5. Treat the reference answer as ground truth when the candidate disagrees with it.\n"
    "6. Judge only the criteria given. Do not invent criteria.\n\n"
    "Return ONLY a JSON object, no prose and no markdown fences, in exactly this form:\n"
    '{"criteria": [{"id": "<criterion id>", "met": true|false, '
    '"justification": "<one short sentence citing the candidate answer>"}], '
    '"overall_comment": "<one sentence overall assessment>"}'
)

JUDGE_USER_TEMPLATE = """\
## Question
{prompt}

## Media
type: {media_kind}

## Reference answer (ground truth)
{golden_response}

## Rubric criteria
{criteria_block}

## Candidate answer (from the model under evaluation)
<candidate_answer>
{answer}
</candidate_answer>

Grade each criterion. Output the JSON object described in the system prompt, with exactly \
{num_criteria} entries in "criteria", one per criterion id, in the same order.\
"""


def render_criteria_block(criteria: List[Criterion]) -> str:
    lines = [f"{i}. [id: {crit.id}] {crit.title}" for i, crit in enumerate(criteria, start=1)]
    return "\n".join(lines) if lines else "(no criteria provided)"


def render_judge_user_prompt(sample: Sample, answer: str) -> str:
    return JUDGE_USER_TEMPLATE.format(
        prompt=sample.prompt,
        media_kind=sample.media_kind,
        golden_response=sample.golden_response,
        criteria_block=render_criteria_block(sample.criteria),
        answer=answer.strip() or "(the model returned an empty answer)",
        num_criteria=len(sample.criteria),
    )


def criteria_ids_json(criteria: List[Criterion]) -> str:
    return json.dumps([c.id for c in criteria])
