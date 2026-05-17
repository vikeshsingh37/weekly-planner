"""Unit tests for eval_data.metrics."""
import pytest
from api.models import Task
from eval_data.metrics import (
    answer_faithfulness,
    answer_keyword_recall,
    first_tool_accuracy,
    graceful_failure_rate,
    session_state_accuracy,
    tool_param_accuracy,
    tool_selection_accuracy,
)
from eval_data.schemas import AnswerCheck, ExpectedToolCall, ParamCheck, SessionCheck


def _call(tool, turn, **params):
    return {"tool": tool, "turn_index": turn, "params": params}


def _task(start=None, end=None):
    t = Task(name="x", duration_minutes=60, priority=3)
    t.scheduled_start = start
    t.scheduled_end = end
    return t


# ── TSA — Tool Selection Accuracy ─────────────────────────────────────────────

class TestToolSelectionAccuracy:
    def test_perfect_match(self):
        r = tool_selection_accuracy(
            [ExpectedToolCall("add_task", 0)],
            [_call("add_task", 0)],
        )
        assert r.score == 1.0

    def test_wrong_tool_name(self):
        r = tool_selection_accuracy(
            [ExpectedToolCall("add_task", 0)],
            [_call("schedule_tasks", 0)],
        )
        assert r.score == 0.0

    def test_wrong_turn_index(self):
        r = tool_selection_accuracy(
            [ExpectedToolCall("add_task", 0)],
            [_call("add_task", 1)],
        )
        assert r.score == 0.0

    def test_optional_call_excluded_from_denominator(self):
        r = tool_selection_accuracy(
            [ExpectedToolCall("add_task", 0, required=False)],
            [],
        )
        assert r.score == 1.0

    def test_no_expected_calls(self):
        assert tool_selection_accuracy([], []).score == 1.0

    def test_partial_hit(self):
        expected = [
            ExpectedToolCall("add_task", 0),
            ExpectedToolCall("schedule_tasks", 1),
        ]
        r = tool_selection_accuracy(expected, [_call("add_task", 0)])
        assert r.score == 0.5


# ── TPA — Tool Parameter Accuracy ─────────────────────────────────────────────

class TestToolParamAccuracy:
    def test_all_checks_pass(self):
        expected = [
            ExpectedToolCall(
                "add_task", 0,
                param_checks=[ParamCheck("duration_minutes", "eq", 60)],
            )
        ]
        r = tool_param_accuracy(expected, [_call("add_task", 0, duration_minutes=60)])
        assert r.score == 1.0

    def test_check_fails(self):
        expected = [
            ExpectedToolCall(
                "add_task", 0,
                param_checks=[ParamCheck("duration_minutes", "eq", 60)],
            )
        ]
        r = tool_param_accuracy(expected, [_call("add_task", 0, duration_minutes=30)])
        assert r.score == 0.0

    def test_missing_call_fails_all_checks(self):
        expected = [
            ExpectedToolCall(
                "add_task", 0,
                param_checks=[ParamCheck("x", "eq", 1), ParamCheck("y", "eq", 2)],
            )
        ]
        assert tool_param_accuracy(expected, []).score == 0.0

    def test_no_param_checks_scores_one(self):
        expected = [ExpectedToolCall("add_task", 0)]
        r = tool_param_accuracy(expected, [_call("add_task", 0)])
        assert r.score == 1.0

    def test_optional_call_skipped(self):
        expected = [
            ExpectedToolCall(
                "add_task", 0, required=False,
                param_checks=[ParamCheck("x", "eq", 1)],
            )
        ]
        assert tool_param_accuracy(expected, []).score == 1.0


# ── FTA — First-Turn Tool Accuracy ───────────────────────────────────────────

class TestFirstToolAccuracy:
    def test_correct_first_tool(self):
        r = first_tool_accuracy(
            [ExpectedToolCall("add_task", 0)],
            [_call("add_task", 0)],
        )
        assert r.score == 1.0

    def test_wrong_first_tool(self):
        r = first_tool_accuracy(
            [ExpectedToolCall("add_task", 0)],
            [_call("schedule_tasks", 0)],
        )
        assert r.score == 0.0

    def test_no_tool_called_on_turn_zero(self):
        r = first_tool_accuracy([ExpectedToolCall("add_task", 0)], [])
        assert r.score == 0.0

    def test_no_turn_zero_expected(self):
        r = first_tool_accuracy([ExpectedToolCall("add_task", 1)], [])
        assert r.score == 1.0


# ── AKR — Answer Keyword Recall ───────────────────────────────────────────────

class TestAnswerKeywordRecall:
    def test_contains_any_found(self):
        checks = [AnswerCheck(0, contains_any=["scheduled", "added"])]
        r = answer_keyword_recall(checks, ["Task was added."])
        assert r.score == 1.0

    def test_contains_any_not_found(self):
        checks = [AnswerCheck(0, contains_any=["scheduled", "added"])]
        r = answer_keyword_recall(checks, ["Something else."])
        assert r.score == 0.0

    def test_contains_all_partial(self):
        checks = [AnswerCheck(0, contains_all=["gym", "scheduled"])]
        r = answer_keyword_recall(checks, ["gym is done"])
        assert r.score == 0.5

    def test_excludes_passes_when_absent(self):
        checks = [AnswerCheck(0, excludes=["error"])]
        r = answer_keyword_recall(checks, ["Task added successfully."])
        assert r.score == 1.0

    def test_excludes_fails_when_present(self):
        checks = [AnswerCheck(0, excludes=["error"])]
        r = answer_keyword_recall(checks, ["An error occurred."])
        assert r.score == 0.0

    def test_negative_index_uses_last_response(self):
        checks = [AnswerCheck(-1, contains_any=["done"])]
        r = answer_keyword_recall(checks, ["first", "second", "done!"])
        assert r.score == 1.0

    def test_case_insensitive(self):
        checks = [AnswerCheck(0, contains_any=["SCHEDULED"])]
        r = answer_keyword_recall(checks, ["task scheduled"])
        assert r.score == 1.0

    def test_no_checks(self):
        assert answer_keyword_recall([], ["anything"]).score == 1.0


# ── AF — Answer Faithfulness ──────────────────────────────────────────────────

class TestAnswerFaithfulness:
    def test_no_times_in_response(self):
        r = answer_faithfulness(["No times mentioned."], [])
        assert r.score == 1.0

    def test_all_mentioned_times_are_real(self):
        t = _task(start="09:00", end="10:00")
        r = answer_faithfulness(["Scheduled 09:00 to 10:00."], [t])
        assert r.score == 1.0

    def test_phantom_time_lowers_score(self):
        t = _task(start="09:00", end="10:00")
        # 11:00 is not a real scheduled time
        r = answer_faithfulness(["09:00 to 11:00"], [t])
        assert r.score < 1.0

    def test_no_tasks_scheduled(self):
        r = answer_faithfulness(["Nothing scheduled yet."], [])
        assert r.score == 1.0


# ── SSA — Session State Accuracy ──────────────────────────────────────────────

class TestSessionStateAccuracy:
    def test_all_checks_pass(self):
        checks = [
            SessionCheck("always true", lambda s: True),
            SessionCheck("also true",   lambda s: True),
        ]
        assert session_state_accuracy(checks, object()).score == 1.0

    def test_partial_pass(self):
        checks = [
            SessionCheck("true",  lambda s: True),
            SessionCheck("false", lambda s: False),
        ]
        assert session_state_accuracy(checks, object()).score == 0.5

    def test_all_fail(self):
        checks = [SessionCheck("false", lambda s: False)]
        assert session_state_accuracy(checks, object()).score == 0.0

    def test_no_checks(self):
        assert session_state_accuracy([], object()).score == 1.0


# ── GFR — Graceful Failure Rate ───────────────────────────────────────────────

class TestGracefulFailureRate:
    def test_keyword_found(self):
        r = graceful_failure_rate(["Sorry, cannot schedule that."], ["cannot"])
        assert r.score == 1.0

    def test_keyword_not_found(self):
        r = graceful_failure_rate(["Task scheduled."], ["cannot", "impossible"])
        assert r.score == 0.0

    def test_case_insensitive(self):
        r = graceful_failure_rate(["CANNOT fit this task."], ["cannot"])
        assert r.score == 1.0

    def test_empty_responses(self):
        r = graceful_failure_rate([], ["cannot"])
        assert r.score == 0.0