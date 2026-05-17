"""Unit tests for api.models Pydantic validators."""
import pytest
from pydantic import ValidationError
from api.models import Preferences, Task


class TestTask:
    def test_valid_task(self):
        t = Task(name="gym", duration_minutes=60, priority=3)
        assert t.name == "gym"
        assert t.status == "pending"

    def test_valid_deadline(self):
        t = Task(name="x", duration_minutes=60, priority=3, deadline="11:00")
        assert t.deadline == "11:00"

    def test_invalid_time_format_rejected(self):
        with pytest.raises(ValidationError):
            Task(name="x", duration_minutes=60, priority=3, deadline="9am")

    def test_invalid_date_format_rejected(self):
        with pytest.raises(ValidationError):
            Task(name="x", duration_minutes=60, priority=3, date="today")

    def test_valid_date(self):
        t = Task(name="x", duration_minutes=60, priority=3, date="2026-05-17")
        assert t.date == "2026-05-17"

    def test_duration_zero_rejected(self):
        with pytest.raises(ValidationError):
            Task(name="x", duration_minutes=0, priority=3)

    def test_duration_over_max_rejected(self):
        with pytest.raises(ValidationError):
            Task(name="x", duration_minutes=601, priority=3)

    def test_priority_too_low_rejected(self):
        with pytest.raises(ValidationError):
            Task(name="x", duration_minutes=60, priority=0)

    def test_priority_too_high_rejected(self):
        with pytest.raises(ValidationError):
            Task(name="x", duration_minutes=60, priority=6)

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            Task(name="", duration_minutes=60, priority=3)

    def test_id_auto_generated(self):
        t1 = Task(name="a", duration_minutes=60, priority=3)
        t2 = Task(name="b", duration_minutes=60, priority=3)
        assert t1.id != t2.id


class TestPreferences:
    def test_defaults(self):
        p = Preferences()
        assert p.work_start == "09:00"
        assert p.work_end == "18:00"
        assert p.break_minutes == 5

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError):
            Preferences(work_start="18:00", work_end="09:00")

    def test_equal_start_end_rejected(self):
        with pytest.raises(ValidationError):
            Preferences(work_start="09:00", work_end="09:00")

    def test_invalid_time_format_rejected(self):
        with pytest.raises(ValidationError):
            Preferences(work_start="9am")

    def test_invalid_timezone_rejected(self):
        with pytest.raises(ValidationError):
            Preferences(timezone="Not/ATimezone")

    def test_valid_timezone(self):
        p = Preferences(timezone="America/New_York")
        assert p.timezone == "America/New_York"

    def test_break_minutes_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Preferences(break_minutes=61)

    def test_max_chunk_minutes_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Preferences(max_chunk_minutes=481)