"""
LLM-as-judge scoring for the Langfuse eval run.

Three semantic metrics that keyword matching cannot reliably cover:

  faithfulness          — does the response accurately reflect what tools returned?
  helpfulness           — is the response genuinely useful to the user?
  failure_explanation   — for edge/failure cases: does the agent explain *why*?

Each function accepts an optional `context` dict with eval metadata (current_time,
work_start, work_end, max_chunk_minutes) so the judge can verify scheduling math
independently of when the eval suite was actually run.

Each function returns (score: float 0–1, reason: str).
Calls use GPT-4.5 via the OpenAI API (OPENAI_API_KEY env var required).
"""

from __future__ import annotations

import json
import logging
import os

from openai import OpenAI

from evals.config import JUDGE_MODEL as _JUDGE_MODEL, JUDGE_MAX_TOKENS as _MAX_TOKENS

logger = logging.getLogger(__name__)


def _judge(system: str, user: str) -> tuple[float, str]:
    """Single judge call. Returns (score 0–1, reason str)."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # gpt-5.x and o-series models use max_completion_tokens; older models use
    # max_tokens. Try the new param first and fall back on the legacy error.
    kwargs: dict = {
        "model": _JUDGE_MODEL,
        "max_completion_tokens": _MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        if "max_completion_tokens" in str(e):
            # Older model — fall back to legacy parameter name
            kwargs.pop("max_completion_tokens")
            kwargs["max_tokens"] = _MAX_TOKENS
            response = client.chat.completions.create(**kwargs)
        else:
            raise
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        score  = float(parsed["score"])
        reason = str(parsed.get("reason", ""))
        return max(0.0, min(1.0, score)), reason
    except Exception:
        logger.warning("Judge parse error — raw: %r", raw[:200])
        return 0.0, f"parse error: {raw[:120]}"


def _format_context(context: dict | None) -> str:
    """Render eval context as a human-readable block for judge prompts."""
    if not context:
        return ""
    lines = []
    if context.get("current_time"):
        lines.append(f"  Simulated current time : {context['current_time']}")
    if context.get("work_start") and context.get("work_end"):
        lines.append(f"  Work window            : {context['work_start']} – {context['work_end']}")
    if context.get("max_chunk_minutes"):
        lines.append(f"  Max task chunk         : {context['max_chunk_minutes']} min")
    return "\n".join(lines)


# ── System prompts ────────────────────────────────────────────────────────────

_FAITHFULNESS_SYSTEM = """\
You are an evaluator checking whether an AI scheduling assistant's response accurately \
reflects what its tools actually returned. You will receive eval context (simulated \
current time, work window) — use it to verify scheduling math such as whether a task \
genuinely cannot fit before a deadline.

Focus on factual consistency: times, task names, durations, deadline constraints, \
and any other concrete values from the tool outputs.

Scoring guide:
1.0  All concrete details in the response match the tool outputs.
0.5  Minor discrepancy — one detail is wrong or omitted.
0.0  Response contradicts or significantly misrepresents tool outputs.

Reply with a JSON object only:
{"score": <number 0.0–1.0>, "reason": "<one sentence explaining your score>"}"""

_HELPFULNESS_SYSTEM = """\
You are an evaluator judging whether an AI scheduling assistant's response is genuinely \
helpful to the user. You will receive eval context (simulated current time, work window) \
— use it to verify that constraint explanations (e.g. "only 1 hour before the deadline \
but the task needs 8 hours") are mathematically accurate, not just plausible-sounding.

Consider clarity, completeness, and whether the response gives the user everything \
they need to act.

Scoring guide:
1.0  Clear, complete, directly addresses the user's need with all key information.
0.5  Partially helpful — missing a key detail or confusingly worded.
0.0  Unhelpful, off-topic, or so vague the user cannot act on it.

Reply with a JSON object only:
{"score": <number 0.0–1.0>, "reason": "<one sentence explaining your score>"}"""

_FAILURE_EXPLANATION_SYSTEM = """\
You are an evaluator judging whether an AI scheduling assistant explains *why* a task \
cannot be scheduled, rather than simply refusing.

You will receive eval context (simulated current time, work window). Use it to verify \
that the agent's stated reason is mathematically correct — e.g. if work_start=09:00, \
current_time=09:00, deadline=10:00, and the task needs 8 hours, the correct explanation \
is that the task duration far exceeds the 1-hour window before the deadline, NOT that \
the deadline has already passed.

A good explanation names the concrete constraint: deadline too close, no free window \
long enough, conflicting event, etc. A bare apology or generic "cannot schedule" with \
no specific reason scores 0.

Scoring guide:
1.0  Names the specific, mathematically correct constraint.
0.5  Acknowledges failure with some context but without the specific limiting detail, \
     or gives a plausible but imprecise reason.
0.0  Just says it cannot do it — no reason given, or states an incorrect reason.

Reply with a JSON object only:
{"score": <number 0.0–1.0>, "reason": "<one sentence explaining your score>"}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_conversation(turns: list[str], responses: list[str]) -> str:
    """Interleave user turns and assistant responses into a readable transcript."""
    lines = []
    for i, user_msg in enumerate(turns):
        lines.append(f"  User Turn {i + 1}: {user_msg}")
        if i < len(responses):
            lines.append(f"  Assistant Turn {i + 1}: {responses[i]}")
        else:
            lines.append(f"  Assistant Turn {i + 1}: (not reached)")
    return "\n".join(lines)


def _tool_section_for_turn(tool_output_log: list[dict], turn_index: int) -> str:
    """Return tool calls labelled by turn; entries for `turn_index` are marked FINAL TURN."""
    if not tool_output_log:
        return "  (no tool calls made)"
    lines = []
    for e in tool_output_log:
        t = e.get("turn_index", "?")
        label = f"Turn {t + 1}" + (" [FINAL]" if t == turn_index else "")
        lines.append(
            f"  [{label} - {e['tool']}] input={json.dumps(e['params'])} → output={json.dumps(e['result'])}"
        )
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def judge_faithfulness(
    turns: list[str],
    tool_output_log: list[dict],
    responses: list[str],
    context: dict | None = None,
) -> tuple[float, str]:
    """Score how faithfully the final-turn response reflects its tool outputs (0–1)."""
    last_turn_idx = len(turns) - 1
    tool_section = _tool_section_for_turn(tool_output_log, last_turn_idx)
    ctx = _format_context(context)
    user_prompt = (
        (f"Eval context:\n{ctx}\n\n" if ctx else "")
        + "Full conversation (interleaved user and assistant turns):\n"
        + _build_conversation(turns, responses)
        + f"\n\nTool calls across all turns (FINAL TURN = Turn {last_turn_idx + 1}):\n{tool_section}"
        + f"\n\nEvaluate the faithfulness of the FINAL assistant response (Turn {last_turn_idx + 1}) "
        + "against the [FINAL] tool outputs only."
    )
    return _judge(_FAITHFULNESS_SYSTEM, user_prompt)


def judge_helpfulness(
    turns: list[str],
    responses: list[str],
    context: dict | None = None,
) -> tuple[float, str]:
    """Score how helpful the final response is to the user (0–1)."""
    ctx = _format_context(context)
    last_turn_idx = len(turns) - 1
    user_prompt = (
        (f"Eval context:\n{ctx}\n\n" if ctx else "")
        + "Full conversation (interleaved user and assistant turns):\n"
        + _build_conversation(turns, responses)
        + f"\n\nEvaluate the helpfulness of the FINAL assistant response (Turn {last_turn_idx + 1})."
    )
    return _judge(_HELPFULNESS_SYSTEM, user_prompt)


def judge_failure_explanation(
    turns: list[str],
    responses: list[str],
    context: dict | None = None,
) -> tuple[float, str]:
    """Score quality of failure explanation for edge/failure cases (0–1)."""
    ctx = _format_context(context)
    last_turn_idx = len(turns) - 1
    user_prompt = (
        (f"Eval context:\n{ctx}\n\n" if ctx else "")
        + "Full conversation (interleaved user and assistant turns):\n"
        + _build_conversation(turns, responses)
        + f"\n\nEvaluate the failure explanation in the FINAL assistant response (Turn {last_turn_idx + 1})."
    )
    return _judge(_FAILURE_EXPLANATION_SYSTEM, user_prompt)