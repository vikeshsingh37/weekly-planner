"""
LLM-as-judge scoring for the Langfuse eval run.

Three semantic metrics that keyword matching cannot reliably cover:

  faithfulness          — does the response accurately reflect what tools returned?
  helpfulness           — is the response genuinely useful to the user?
  failure_explanation   — for edge/failure cases: does the agent explain *why*?

Each function returns (score: float 0–1, reason: str).
Calls use GPT-4.5 via the OpenAI API (OPENAI_API_KEY env var required).
"""

from __future__ import annotations

import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_JUDGE_MODEL = "gpt-4.5-preview"
_MAX_TOKENS = 256


def _judge(system: str, user: str) -> tuple[float, str]:
    """Single judge call. Returns (score 0–1, reason str)."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=_JUDGE_MODEL,
        max_tokens=_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        score  = float(parsed["score"])
        reason = str(parsed.get("reason", ""))
        return max(0.0, min(1.0, score)), reason
    except Exception:
        logger.warning("Judge parse error — raw: %r", raw[:200])
        return 0.0, f"parse error: {raw[:120]}"


# ── System prompts ────────────────────────────────────────────────────────────

_FAITHFULNESS_SYSTEM = """\
You are an evaluator checking whether an AI scheduling assistant's response accurately \
reflects what its tools actually returned. Focus on factual consistency: times, task \
names, durations, and any other concrete values from the tool outputs.

Scoring guide:
1.0  All concrete details in the response match the tool outputs.
0.5  Minor discrepancy — one detail is wrong or omitted.
0.0  Response contradicts or significantly misrepresents tool outputs.

Reply with a JSON object only:
{"score": <number 0.0–1.0>, "reason": "<one sentence explaining your score>"}"""

_HELPFULNESS_SYSTEM = """\
You are an evaluator judging whether an AI scheduling assistant's response is genuinely \
helpful to the user. Consider clarity, completeness, and whether it gives the user \
everything they need to act.

Scoring guide:
1.0  Clear, complete, directly addresses the user's need with all key information.
0.5  Partially helpful — missing a key detail or confusingly worded.
0.0  Unhelpful, off-topic, or so vague the user cannot act on it.

Reply with a JSON object only:
{"score": <number 0.0–1.0>, "reason": "<one sentence explaining your score>"}"""

_FAILURE_EXPLANATION_SYSTEM = """\
You are an evaluator judging whether an AI scheduling assistant explains *why* a task \
cannot be scheduled, rather than simply refusing.

A good explanation names the concrete constraint: deadline too close, no free window \
long enough, conflicting event, etc. A bare apology or generic "cannot schedule" with \
no specific reason scores 0.

Scoring guide:
1.0  Names the specific constraint (e.g. "only 30 min available before your 3 PM deadline but the task needs 2 hours").
0.5  Acknowledges failure with some context but without the specific limiting detail.
0.0  Just says it cannot do it — no reason given.

Reply with a JSON object only:
{"score": <number 0.0–1.0>, "reason": "<one sentence explaining your score>"}"""


# ── Public API ────────────────────────────────────────────────────────────────

def judge_faithfulness(
    turns: list[str],
    tool_output_log: list[dict],
    responses: list[str],
) -> tuple[float, str]:
    """Score how faithfully the response reflects actual tool outputs (0–1)."""
    tool_section = (
        "\n".join(
            f"  [{e['tool']}] input={json.dumps(e['params'])} → output={json.dumps(e['result'])}"
            for e in tool_output_log
        )
        or "  (no tool calls made)"
    )
    user_prompt = (
        "User conversation:\n"
        + "\n".join(f"  Turn {i + 1}: {t}" for i, t in enumerate(turns))
        + f"\n\nTool calls and their actual outputs:\n{tool_section}"
        + f"\n\nAssistant's final response:\n{responses[-1] if responses else '(no response)'}"
    )
    return _judge(_FAITHFULNESS_SYSTEM, user_prompt)


def judge_helpfulness(
    turns: list[str],
    responses: list[str],
) -> tuple[float, str]:
    """Score how helpful the final response is to the user (0–1)."""
    user_prompt = (
        "User conversation:\n"
        + "\n".join(f"  Turn {i + 1}: {t}" for i, t in enumerate(turns))
        + f"\n\nAssistant's final response:\n{responses[-1] if responses else '(no response)'}"
    )
    return _judge(_HELPFULNESS_SYSTEM, user_prompt)


def judge_failure_explanation(
    turns: list[str],
    responses: list[str],
) -> tuple[float, str]:
    """Score quality of failure explanation for edge/failure cases (0–1)."""
    user_prompt = (
        "User conversation:\n"
        + "\n".join(f"  Turn {i + 1}: {t}" for i, t in enumerate(turns))
        + f"\n\nAssistant's response:\n{responses[-1] if responses else '(no response)'}"
    )
    return _judge(_FAILURE_EXPLANATION_SYSTEM, user_prompt)