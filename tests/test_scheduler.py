"""Unit tests for the EDF scheduler."""
import pytest
from api.models import Preferences, Task
from impl.scheduler import EDFScheduler, _ceil15, _to_ampm, _to_hhmm, _to_min


def make_task(name, duration, priority=3, deadline=None,
              pinned=False, scheduled_start=None, scheduled_end=None):
    t = Task(name=name, duration_minutes=duration, priority=priority)
    if deadline:
        t.deadline = deadline
    if pinned:
        t.pinned = True
        t.scheduled_start = scheduled_start
        t.scheduled_end = scheduled_end
        t.status = "scheduled"
    return t


def prefs(start="09:00", end="18:00", break_min=0, max_chunk=0):
    return Preferences(
        work_start=start,
        work_end=end,
        break_minutes=break_min,
        max_chunk_minutes=max_chunk,
    )


scheduler = EDFScheduler()


# ── Conversion helpers ────────────────────────────────────────────────────────

class TestHelpers:
    def test_to_min(self):
        assert _to_min("09:00") == 540
        assert _to_min("00:00") == 0
        assert _to_min("23:59") == 1439

    def test_to_hhmm(self):
        assert _to_hhmm(540) == "09:00"
        assert _to_hhmm(0) == "00:00"
        assert _to_hhmm(1439) == "23:59"

    def test_to_ampm_morning(self):
        assert _to_ampm("09:00") == "9:00 AM"

    def test_to_ampm_noon(self):
        assert _to_ampm("12:00") == "12:00 PM"

    def test_to_ampm_afternoon(self):
        assert _to_ampm("13:30") == "1:30 PM"

    def test_to_ampm_midnight(self):
        assert _to_ampm("00:00") == "12:00 AM"

    def test_ceil15_already_aligned(self):
        assert _ceil15(0) == 0
        assert _ceil15(15) == 15
        assert _ceil15(60) == 60

    def test_ceil15_rounds_up(self):
        assert _ceil15(1) == 15
        assert _ceil15(16) == 30
        assert _ceil15(31) == 45


# ── Basic scheduling ──────────────────────────────────────────────────────────

class TestScheduler:
    def test_single_task_fits(self):
        result = scheduler.schedule([make_task("work", 60)], prefs())
        assert len(result.scheduled) == 1
        assert result.scheduled[0].name == "work"
        assert result.scheduled[0].start == "09:00"
        assert result.scheduled[0].end == "10:00"

    def test_two_tasks_sequential(self):
        tasks = [make_task("A", 60), make_task("B", 60)]
        result = scheduler.schedule(tasks, prefs())
        assert {s.name for s in result.scheduled} == {"A", "B"}
        assert result.unschedulable == []

    def test_empty_tasks(self):
        result = scheduler.schedule([], prefs())
        assert result.scheduled == []
        assert result.unschedulable == []
        assert result.conflicts == []

    def test_tasks_fill_window_exactly(self):
        tasks = [make_task("A", 240), make_task("B", 240)]
        result = scheduler.schedule(tasks, prefs("09:00", "17:00"))
        assert len(result.scheduled) == 2
        assert result.unschedulable == []

    def test_partial_scheduling_when_overflow(self):
        tasks = [make_task("A", 180), make_task("B", 180), make_task("C", 180)]
        result = scheduler.schedule(tasks, prefs("09:00", "14:00"))
        assert len(result.unschedulable) >= 1

    def test_task_exceeds_full_window_unschedulable(self):
        result = scheduler.schedule(
            [make_task("big", 540)], prefs("09:00", "17:00")
        )
        assert len(result.unschedulable) == 1
        assert result.unschedulable[0].name == "big"

    def test_now_min_clamps_past_slots(self):
        result = scheduler.schedule(
            [make_task("work", 60)], prefs(), now_min=_to_min("12:00")
        )
        assert len(result.scheduled) == 1
        assert result.scheduled[0].start >= "12:00"


# ── Deadlines ─────────────────────────────────────────────────────────────────

class TestDeadlines:
    def test_deadline_respected(self):
        task = make_task("report", 60, deadline="11:00")
        result = scheduler.schedule([task], prefs())
        assert len(result.scheduled) == 1
        assert result.scheduled[0].end <= "11:00"

    def test_deadline_impossible_unschedulable(self):
        # 5h task, work starts 09:00, deadline 11:00 — only 2h available
        task = make_task("report", 300, deadline="11:00")
        result = scheduler.schedule([task], prefs())
        assert len(result.unschedulable) == 1

    def test_edf_orders_by_deadline(self):
        # earlier deadline wins regardless of priority
        late  = make_task("late",  60, priority=5, deadline="17:00")
        early = make_task("early", 60, priority=1, deadline="10:00")
        result = scheduler.schedule([late, early], prefs())
        assert len(result.scheduled) == 2
        starts = {s.name: s.start for s in result.scheduled}
        assert starts["early"] < starts["late"]


# ── Pinned tasks ──────────────────────────────────────────────────────────────

class TestPinnedTasks:
    def test_pinned_task_not_moved(self):
        pinned = make_task(
            "meeting", 60, pinned=True,
            scheduled_start="10:00", scheduled_end="11:00",
        )
        result = scheduler.schedule([pinned], prefs())
        assert len(result.scheduled) == 1
        assert result.scheduled[0].start == "10:00"

    def test_free_task_does_not_overlap_pinned(self):
        pinned = make_task(
            "meeting", 60, pinned=True,
            scheduled_start="10:00", scheduled_end="11:00",
        )
        free = make_task("work", 60)
        result = scheduler.schedule([pinned, free], prefs())
        slots = {s.name: s for s in result.scheduled}
        assert "work" in slots
        # no overlap with [10:00, 11:00]
        assert not (slots["work"].start < "11:00" and slots["work"].end > "10:00")


# ── Chunking ──────────────────────────────────────────────────────────────────

class TestChunking:
    def test_task_split_into_two_chunks(self):
        task = make_task("deep", 180)
        result = scheduler.schedule([task], prefs(max_chunk=90))
        assert len(result.scheduled) == 1
        assert len(result.scheduled[0].chunks) == 2

    def test_chunk_fits_within_deadline(self):
        task = make_task("focus", 120, deadline="18:00")
        result = scheduler.schedule([task], prefs(max_chunk=60))
        assert len(result.scheduled) == 1

    def test_chunked_task_unschedulable_when_impossible(self):
        # 4h task, 90-min chunks, deadline 11:00 — can't place blocks before deadline
        task = make_task("huge", 240, deadline="11:00")
        result = scheduler.schedule([task], prefs(max_chunk=90))
        assert len(result.unschedulable) == 1


# ── Conflict detection ─────────────────────────────────────────────────────────

class TestConflicts:
    def test_no_conflict_for_sequential_tasks(self):
        tasks = [make_task("A", 60), make_task("B", 60)]
        result = scheduler.schedule(tasks, prefs())
        assert result.conflicts == []

    def test_conflict_between_overlapping_pinned_tasks(self):
        a = make_task("A", 60, pinned=True, scheduled_start="09:00", scheduled_end="10:00")
        b = make_task("B", 60, pinned=True, scheduled_start="09:30", scheduled_end="10:30")
        conflicts = scheduler.check_conflicts([a, b])
        assert len(conflicts) == 1
        assert conflicts[0].overlap_minutes == 30

    def test_no_conflict_for_adjacent_tasks(self):
        a = make_task("A", 60, pinned=True, scheduled_start="09:00", scheduled_end="10:00")
        b = make_task("B", 60, pinned=True, scheduled_start="10:00", scheduled_end="11:00")
        assert scheduler.check_conflicts([a, b]) == []