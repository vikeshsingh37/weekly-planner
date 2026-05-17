"""
Eval dataset schemas.

Three evaluation dimensions:
  1. Tool selection  — was the right tool called on the right turn?
  2. Tool params     — were the right arguments passed?
  3. Final answer    — does the response faithfully reflect tool output?

Capturing tool calls:
  No agent instrumentation needed.  WeeklyPlannerAgent._run_tool() is already
  @observe-decorated, so every call is recorded in Langfuse as a child span
  with the tool name as span.name and the input dict as span.input.  Query the
  Langfuse API for eval_user traces after each run and reconstruct a tool_call_log
  to pass to the metric functions in metrics.py.  See README.md for the snippet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from api.memory import AbstractSessionManager


@dataclass
class ParamCheck:
    """Assert a specific field value in a tool call's input dict.

    field: dot-path into the input dict, e.g.:
        "tasks.0.duration_minutes"  →  input["tasks"][0]["duration_minutes"]
        "work_start"                →  input["work_start"]
        "new_start_time"            →  input["new_start_time"]

    op:
        "eq"         — field == value
        "contains"   — value in str(field), case-insensitive
        "gte"        — field >= value
        "lte"        — field <= value
        "in"         — field in value  (value must be a collection)
        "startswith" — str(field).startswith(value), case-insensitive
    """

    field: str
    op: str
    value: Any
    description: str = ""

    def evaluate(self, params: dict) -> bool:
        obj: Any = params
        try:
            for part in self.field.split("."):
                obj = obj[int(part)] if part.isdigit() else obj[part]
        except (KeyError, IndexError, TypeError):
            return False

        if self.op == "eq":
            return obj == self.value
        if self.op == "contains":
            return self.value.lower() in str(obj).lower()
        if self.op == "gte":
            return obj >= self.value
        if self.op == "lte":
            return obj <= self.value
        if self.op == "in":
            return obj in self.value
        if self.op == "startswith":
            return str(obj).lower().startswith(str(self.value).lower())
        raise ValueError(f"Unknown op: {self.op!r}")


@dataclass
class ExpectedToolCall:
    """A tool call the agent should make in response to a specific user turn."""

    tool: str        # one of the 7 tool names in api/tools.py
    turn_index: int  # 0-indexed position in EvalDatapoint.turns that triggers this call
    param_checks: List[ParamCheck] = field(default_factory=list)
    required: bool = True  # False → nice-to-have; excluded from TSA denominator


@dataclass
class AnswerCheck:
    """Keyword-level assertions on the agent's text response for a specific turn.

    turn_index: index into the agent responses list; -1 means the last response.
    contains_all: every keyword must appear (case-insensitive).
    contains_any: at least one keyword must appear (case-insensitive).
    excludes:     none of these keywords may appear (case-insensitive).
    """

    turn_index: int = -1
    contains_all: List[str] = field(default_factory=list)
    contains_any: List[str] = field(default_factory=list)
    excludes: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class SessionCheck:
    """Assertion on session state after all conversation turns complete."""

    description: str
    fn: Callable[[AbstractSessionManager], bool]


@dataclass
class EvalDatapoint:
    """One evaluation example, covering up to N conversation turns."""

    id: str
    category: str  # tool_selection | tool_params | multi_turn | final_answer | edge_case
    description: str
    turns: List[str]  # ordered user messages

    expected_tool_calls: List[ExpectedToolCall] = field(default_factory=list)
    answer_checks: List[AnswerCheck] = field(default_factory=list)
    session_checks: List[SessionCheck] = field(default_factory=list)

    # Which metrics from metrics.py to compute for this datapoint
    metrics: List[str] = field(default_factory=list)

    # Optional preference overrides applied to the fresh session before any turns run
    preferences: Optional[dict] = None

    # Human-readable rationale / what edge case this covers
    notes: str = ""