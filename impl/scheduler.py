"""
EDF (Earliest-Deadline-First) scheduler — implements api.scheduler.AbstractScheduler.

Rules:
  1. Pinned tasks (user-placed) block time and are never moved.
  2. Remaining tasks are sorted: deadline ascending, then priority descending.
  3. Free slots are the complement of pinned-task intervals within the work window.
  4. Tasks are placed greedily into the first slot that fits AND honours the deadline.
  5. Unschedulable tasks get a reason string — they are never silently dropped.
"""

from api.models import ConflictItem, Preferences, ScheduleResult, ScheduledSlot, Task, UnschedulableItem
from api.scheduler import AbstractScheduler


def _to_min(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def _to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _deadline_min(task: Task, work_end: int) -> int:
    return _to_min(task.deadline) if task.deadline else work_end


def _sort_key(task: Task, work_end: int) -> tuple:
    return (_deadline_min(task, work_end), 6 - task.priority)


class EDFScheduler(AbstractScheduler):
    def schedule(self, tasks: list[Task], preferences: Preferences) -> ScheduleResult:
        ws = _to_min(preferences.work_start)
        we = _to_min(preferences.work_end)
        gap = preferences.break_minutes
        total = we - ws

        # Reset non-pinned tasks
        for t in tasks:
            if not t.pinned:
                t.scheduled_start = None
                t.scheduled_end = None
                t.status = "pending"

        # Build free slots from work window minus pinned intervals
        blocked = sorted(
            (_to_min(t.scheduled_start), _to_min(t.scheduled_end))
            for t in tasks
            if t.pinned and t.scheduled_start and t.scheduled_end
        )
        free: list[list[int]] = []
        cursor = ws
        for b_start, b_end in blocked:
            if cursor < b_start:
                free.append([cursor, b_start])
            cursor = max(cursor, b_end)
        if cursor < we:
            free.append([cursor, we])

        pending = sorted(
            (t for t in tasks if not t.pinned),
            key=lambda t: _sort_key(t, we),
        )

        unschedulable: list[UnschedulableItem] = []

        for task in pending:
            dur = task.duration_minutes
            deadline = _deadline_min(task, we)

            if dur > total:
                task.status = "unschedulable"
                unschedulable.append(UnschedulableItem(
                    name=task.name,
                    reason=f"Duration ({dur}m) exceeds entire work window ({total}m)",
                ))
                continue

            placed = False
            for slot in free:
                avail = slot[1] - slot[0]
                if avail >= dur and slot[0] + dur <= deadline:
                    task.scheduled_start = _to_hhmm(slot[0])
                    task.scheduled_end = _to_hhmm(slot[0] + dur)
                    task.status = "scheduled"
                    slot[0] = slot[0] + dur + gap
                    placed = True
                    break

            if not placed:
                task.status = "unschedulable"
                dl_str = task.deadline or preferences.work_end
                unschedulable.append(UnschedulableItem(
                    name=task.name,
                    reason=f"No free slot of {dur}m before deadline {dl_str}",
                ))

        scheduled = [
            ScheduledSlot(name=t.name, start=t.scheduled_start, end=t.scheduled_end)
            for t in tasks
            if t.status == "scheduled"
        ]
        return ScheduleResult(
            scheduled=scheduled,
            unschedulable=unschedulable,
            conflicts=self.check_conflicts(tasks),
        )

    def check_conflicts(self, tasks: list[Task]) -> list[ConflictItem]:
        active = sorted(
            (t for t in tasks if t.scheduled_start and t.scheduled_end),
            key=lambda t: _to_min(t.scheduled_start),
        )
        conflicts: list[ConflictItem] = []
        for i in range(len(active) - 1):
            a, b = active[i], active[i + 1]
            a_end = _to_min(a.scheduled_end)
            b_start = _to_min(b.scheduled_start)
            if a_end > b_start:
                conflicts.append(ConflictItem(
                    task_a=a.name,
                    task_b=b.name,
                    overlap_minutes=a_end - b_start,
                ))
        return conflicts
