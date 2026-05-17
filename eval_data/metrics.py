"""
Metric definitions for the Weekly Planner eval dataset.

Four metric groups
──────────────────
  A) Tool-level    TSA, TPA, FTA  — did the right tools get called with the right params?
  B) Answer-level  AKR, AF        — does the response text reflect ground truth?
  C) State-level   SSA            — is session state correct after all turns?
  D) Robustness    GFR            — how gracefully does the agent handle failures?

Capturing tool calls
─────────────────────
  No agent instrumentation needed.  WeeklyPlannerAgent._run_tool() is already
  @observe-decorated, so every call lands in Langfuse as a child span:
    span.name  = tool name
    span.input = params dict passed to the tool

  After each run, query the Langfuse API for eval_user traces and build a
  tool_call_log list to pass to tool_selection_accuracy() / tool_param_accuracy()
  / first_tool_accuracy().  See eval_data/README.md for the fetch snippet.

Targets
────────
  TSA ≥ 0.95   TPA ≥ 0.90   FTA = 1.0
  AKR ≥ 0.90   AF  ≥ 0.95
  SSA ≥ 0.90   GFR = 1.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from eval_data.schemas import AnswerCheck, EvalDatapoint, ExpectedToolCall, SessionCheck


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    metric: str
    score: float    # always 0.0–1.0; higher is better
    details: str = ""


# ── A. Tool-level metrics ─────────────────────────────────────────────────────

def tool_selection_accuracy(
    expected: List[ExpectedToolCall],
    actual_log: List[Dict],
) -> MetricResult:
    """
    TSA — Tool Selection Accuracy

    Fraction of *required* expected tool calls that were actually made on the
    correct turn.  A call matches if tool name AND turn_index both agree.

    actual_log entries: {"tool": str, "turn_index": int, "params": dict}
    Target: ≥ 0.95
    """
    required = [e for e in expected if e.required]
    if not required:
        return MetricResult("tool_selection_accuracy", 1.0, "no required tool calls")
    hits = sum(
        1
        for exp in required
        if any(a["tool"] == exp.tool and a["turn_index"] == exp.turn_index for a in actual_log)
    )
    return MetricResult(
        "tool_selection_accuracy",
        hits / len(required),
        f"{hits}/{len(required)} required tools called on correct turn",
    )


def tool_param_accuracy(
    expected: List[ExpectedToolCall],
    actual_log: List[Dict],
) -> MetricResult:
    """
    TPA — Tool Parameter Accuracy

    Fraction of declared ParamChecks that pass against the actual tool input
    dict.  Only checks required calls.  Missing calls (covered by TSA) count
    all their param_checks as failures.

    Target: ≥ 0.90
    """
    total = 0
    passed = 0
    for exp in expected:
        if not exp.required or not exp.param_checks:
            continue
        actual = next(
            (
                a for a in actual_log
                if a["tool"] == exp.tool and a["turn_index"] == exp.turn_index
            ),
            None,
        )
        for pc in exp.param_checks:
            total += 1
            if actual is not None and pc.evaluate(actual["params"]):
                passed += 1
    if total == 0:
        return MetricResult("tool_param_accuracy", 1.0, "no param checks defined")
    return MetricResult(
        "tool_param_accuracy",
        passed / total,
        f"{passed}/{total} param checks passed",
    )


def first_tool_accuracy(
    expected: List[ExpectedToolCall],
    actual_log: List[Dict],
) -> MetricResult:
    """
    FTA — First-Turn Tool Accuracy

    Binary check: was the very first tool called (turn_index=0) the right one?
    High-signal proxy for intent recognition quality.

    Target: 1.0
    """
    exp0 = next((e for e in expected if e.turn_index == 0 and e.required), None)
    if exp0 is None:
        return MetricResult("first_tool_accuracy", 1.0, "no turn-0 required tool expected")
    act0 = next((a for a in actual_log if a["turn_index"] == 0), None)
    if act0 is None:
        return MetricResult("first_tool_accuracy", 0.0, "no tool called on turn 0")
    ok = act0["tool"] == exp0.tool
    return MetricResult(
        "first_tool_accuracy",
        1.0 if ok else 0.0,
        f"expected {exp0.tool!r}, got {act0['tool']!r}",
    )


# ── B. Answer-level metrics ───────────────────────────────────────────────────

def answer_keyword_recall(
    checks: List[AnswerCheck],
    responses: List[str],
) -> MetricResult:
    """
    AKR — Answer Keyword Recall

    Fraction of keyword constraints (contains_all, contains_any, excludes)
    that are satisfied across all AnswerChecks for this datapoint.

    Scoring:
      contains_all  → 1 point per keyword (all must appear)
      contains_any  → 1 point for the group (any one must appear)
      excludes      → 1 point per keyword (none must appear)

    Target: ≥ 0.90
    """
    total = 0
    satisfied = 0

    for ac in checks:
        try:
            resp = responses[ac.turn_index].lower()
        except IndexError:
            resp = ""

        for kw in ac.contains_all:
            total += 1
            if kw.lower() in resp:
                satisfied += 1

        if ac.contains_any:
            total += 1
            if any(kw.lower() in resp for kw in ac.contains_any):
                satisfied += 1

        for kw in ac.excludes:
            total += 1
            if kw.lower() not in resp:
                satisfied += 1

    if total == 0:
        return MetricResult("answer_keyword_recall", 1.0, "no answer checks defined")
    return MetricResult(
        "answer_keyword_recall",
        satisfied / total,
        f"{satisfied}/{total} keyword constraints satisfied",
    )


def answer_faithfulness(
    responses: List[str],
    session_tasks: List[Any],
) -> MetricResult:
    """
    AF — Answer Faithfulness (anti-hallucination heuristic)

    Checks that every HH:MM time mentioned in the response corresponds to a
    real scheduled_start or scheduled_end in the session.

    This is a lightweight deterministic proxy.  For deeper faithfulness scoring
    (response vs. full tool output text) use an LLM-as-judge evaluator in
    Langfuse — see README.md § Real-time Evals.

    Target: ≥ 0.95
    """
    real_times: set[str] = set()
    for t in session_tasks:
        if t.scheduled_start:
            real_times.add(t.scheduled_start)
        if t.scheduled_end:
            real_times.add(t.scheduled_end)

    combined = " ".join(responses)

    # Collect HH:MM patterns (both 24h and 12h without AM/PM suffix)
    mentioned: set[str] = set(re.findall(r"\b([01]?\d:[0-5]\d)\b", combined))
    # 12h patterns like "9:00 AM" or "2:30 PM"
    for m in re.finditer(r"\b(1?[0-9]:[0-5]\d)\s*(AM|PM)\b", combined, re.IGNORECASE):
        mentioned.add(m.group(1))

    if not mentioned or not real_times:
        return MetricResult("answer_faithfulness", 1.0, "no scheduled times to verify")

    phantom = sum(1 for t in mentioned if t not in real_times)
    score = max(0.0, 1.0 - phantom / len(mentioned))
    return MetricResult(
        "answer_faithfulness",
        score,
        f"{phantom} phantom time(s) in response out of {len(mentioned)} mentioned",
    )


# ── C. State-level metrics ────────────────────────────────────────────────────

def session_state_accuracy(
    checks: List[SessionCheck],
    session: Any,
) -> MetricResult:
    """
    SSA — Session State Accuracy

    Fraction of SessionChecks that pass after all turns complete.
    Ground-truth check on what actually persisted to the session store.

    Target: ≥ 0.90
    """
    if not checks:
        return MetricResult("session_state_accuracy", 1.0, "no session checks defined")
    passed = sum(1 for c in checks if c.fn(session))
    return MetricResult(
        "session_state_accuracy",
        passed / len(checks),
        f"{passed}/{len(checks)} session checks passed",
    )


# ── D. Robustness metrics ─────────────────────────────────────────────────────

def graceful_failure_rate(
    responses: List[str],
    failure_keywords: List[str],
) -> MetricResult:
    """
    GFR — Graceful Failure Rate

    For edge-case / impossible scenarios: does at least one failure keyword
    appear in any response?  Binary 0 or 1.

    Pass failure_keywords from the datapoint's AnswerCheck.contains_any list.

    Target: 1.0
    """
    combined = " ".join(responses).lower()
    found = any(kw.lower() in combined for kw in failure_keywords)
    return MetricResult(
        "graceful_failure_rate",
        1.0 if found else 0.0,
        f"looked for any of: {failure_keywords}",
    )


# ── Registry ──────────────────────────────────────────────────────────────────

METRIC_REGISTRY = {
    "tool_selection_accuracy": tool_selection_accuracy,
    "tool_param_accuracy": tool_param_accuracy,
    "first_tool_accuracy": first_tool_accuracy,
    "answer_keyword_recall": answer_keyword_recall,
    "answer_faithfulness": answer_faithfulness,
    "session_state_accuracy": session_state_accuracy,
    "graceful_failure_rate": graceful_failure_rate,
}

METRIC_DESCRIPTIONS: dict[str, str] = {
    "tool_selection_accuracy": (
        "TSA — Tool Selection Accuracy\n"
        "Fraction of required expected tool calls made on the correct turn.\n"
        "Penalises missing calls and wrong tool choices.\n"
        "Target ≥ 0.95."
    ),
    "tool_param_accuracy": (
        "TPA — Tool Parameter Accuracy\n"
        "Fraction of declared ParamChecks that pass against actual tool inputs.\n"
        "Catches silent misextractions (e.g. duration=60 when user said '2 hours').\n"
        "Target ≥ 0.90."
    ),
    "first_tool_accuracy": (
        "FTA — First-Turn Tool Accuracy\n"
        "Binary: was the first tool called on turn 0 the expected one?\n"
        "High-signal proxy for intent recognition.\n"
        "Target = 1.0."
    ),
    "answer_keyword_recall": (
        "AKR — Answer Keyword Recall\n"
        "Fraction of keyword constraints satisfied in the agent text responses.\n"
        "Covers contains_all, contains_any, and excludes constraints.\n"
        "Target ≥ 0.90."
    ),
    "answer_faithfulness": (
        "AF — Answer Faithfulness (anti-hallucination heuristic)\n"
        "Fraction of times mentioned in the response that correspond to real\n"
        "scheduled_start / scheduled_end values in the session.\n"
        "Use Langfuse LLM-as-judge for deeper faithfulness scoring.\n"
        "Target ≥ 0.95."
    ),
    "session_state_accuracy": (
        "SSA — Session State Accuracy\n"
        "Fraction of SessionChecks that pass after all turns complete.\n"
        "Ground-truth check on persistent state.\n"
        "Target ≥ 0.90."
    ),
    "graceful_failure_rate": (
        "GFR — Graceful Failure Rate\n"
        "Binary: for edge/failure cases, does the agent acknowledge the failure?\n"
        "Target = 1.0."
    ),
}