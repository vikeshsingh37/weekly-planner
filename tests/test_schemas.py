"""Unit tests for eval_data.schemas.ParamCheck.evaluate."""
import pytest
from eval_data.schemas import ParamCheck


class TestParamCheckEvaluate:
    # ── Operators ─────────────────────────────────────────────────────────────

    def test_eq_match(self):
        assert ParamCheck("dur", "eq", 60).evaluate({"dur": 60}) is True

    def test_eq_no_match(self):
        assert ParamCheck("dur", "eq", 60).evaluate({"dur": 30}) is False

    def test_contains_found(self):
        assert ParamCheck("name", "contains", "gym").evaluate({"name": "Morning Gym"}) is True

    def test_contains_not_found(self):
        assert ParamCheck("name", "contains", "gym").evaluate({"name": "running"}) is False

    def test_contains_case_insensitive(self):
        assert ParamCheck("name", "contains", "GYM").evaluate({"name": "gym session"}) is True

    def test_gte_equal(self):
        assert ParamCheck("p", "gte", 4).evaluate({"p": 4}) is True

    def test_gte_greater(self):
        assert ParamCheck("p", "gte", 4).evaluate({"p": 5}) is True

    def test_gte_less(self):
        assert ParamCheck("p", "gte", 4).evaluate({"p": 3}) is False

    def test_lte_equal(self):
        assert ParamCheck("p", "lte", 3).evaluate({"p": 3}) is True

    def test_lte_greater(self):
        assert ParamCheck("p", "lte", 3).evaluate({"p": 4}) is False

    def test_in_member(self):
        assert ParamCheck("p", "in", [1, 2]).evaluate({"p": 1}) is True

    def test_in_not_member(self):
        assert ParamCheck("p", "in", [1, 2]).evaluate({"p": 3}) is False

    def test_startswith_match(self):
        assert ParamCheck("name", "startswith", "gym").evaluate({"name": "Gym session"}) is True

    def test_startswith_no_match(self):
        assert ParamCheck("name", "startswith", "gym").evaluate({"name": "morning gym"}) is False

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError, match="Unknown op"):
            ParamCheck("x", "bogus", 1).evaluate({"x": 1})

    # ── Dot-path traversal ────────────────────────────────────────────────────

    def test_dot_path_list_index(self):
        pc = ParamCheck("tasks.0.duration_minutes", "eq", 120)
        assert pc.evaluate({"tasks": [{"duration_minutes": 120}]}) is True

    def test_dot_path_nested_dict(self):
        pc = ParamCheck("a.b.c", "eq", 42)
        assert pc.evaluate({"a": {"b": {"c": 42}}}) is True

    def test_missing_key_returns_false(self):
        assert ParamCheck("missing", "eq", 1).evaluate({}) is False

    def test_out_of_range_index_returns_false(self):
        assert ParamCheck("tasks.5.name", "eq", "x").evaluate({"tasks": []}) is False

    def test_none_intermediate_returns_false(self):
        assert ParamCheck("a.b", "eq", 1).evaluate({"a": None}) is False