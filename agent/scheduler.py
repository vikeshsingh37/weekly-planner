"""
Deterministic task scheduler using Earliest-Deadline-First (EDF).

Rules:
  1. Pinned tasks (user-placed) are fixed; they block time and are not moved.
  2. Remaining tasks are sorted by deadline (earliest first), then priority (highest first).
  3. Free slots are computed from work hours minus pinned tasks.
  4. Tasks are greedily assigned to the first slot that fits.
  5. Tasks that cannot fit are returned as unschedulable with a reason.
"""

from datetime import datetime, timedelta
from typing import List, Tuple

from .memory import Task


def _to_minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def _to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _parse_deadline_minutes(deadline: str | None, work_end_minutes: int) -> int:
    """Return deadline as minutes-since-midnight. None → work_end."""
    if deadline is None:
        return work_end_minutes
    return _to_minutes(deadline)


def _sort_key(task: Task, work_end_minutes: int):
    deadline_min = _parse_deadline_minutes(task.deadline, work_end_minutes)
    priority_score = 6 - task.priority  # lower is better (sort ascending)
    return (deadline_min, priority_score)


def schedule(
    tasks: List[Task],
    work_start: str,
    work_end: str,
    break_minutes: int = 5,
) -> Tuple[List[Task], List[dict]]:
    """
    Schedule tasks and return (updated_tasks, unschedulable_list).

    unschedulable_list entries: {"task": Task, "reason": str}
    """
    ws = _to_minutes(work_start)
    we = _to_minutes(work_end)
    total_work = we - ws

    if total_work <= 0:
        unschedulable = [
            {"task": t, "reason": "Work window is zero or negative"}
            for t in tasks
            if not t.pinned
        ]
        return tasks, unschedulable

    # Reset status for non-pinned tasks so we reschedule from scratch
    for t in tasks:
        if not t.pinned:
            t.scheduled_start = None
            t.scheduled_end = None
            t.status = "pending"

    # Build blocked intervals from pinned tasks
    blocked: List[Tuple[int, int]] = []
    for t in tasks:
        if t.pinned and t.scheduled_start and t.scheduled_end:
            blocked.append((_to_minutes(t.scheduled_start), _to_minutes(t.scheduled_end)))
    blocked.sort()

    # Derive free slots (list of [start, end] in minutes)
    free: List[List[int]] = []
    cursor = ws
    for b_start, b_end in blocked:
        if cursor < b_start:
            free.append([cursor, b_start])
        cursor = max(cursor, b_end)
    if cursor < we:
        free.append([cursor, we])

    # Sort non-pinned tasks by EDF then priority
    pending = [t for t in tasks if not t.pinned]
    pending.sort(key=lambda t: _sort_key(t, we))

    unschedulable: List[dict] = []

    for task in pending:
        duration = task.duration_minutes
        deadline_min = _parse_deadline_minutes(task.deadline, we)

        if duration > total_work:
            task.status = "unschedulable"
            unschedulable.append({
                "task": task,
                "reason": f"Task duration ({duration}m) exceeds total work window ({total_work}m)",
            })
            continue

        placed = False
        for slot in free:
            slot_start, slot_end = slot
            available = slot_end - slot_start
            if available >= duration:
                task_end = slot_start + duration
                # Enforce deadline: task must finish by deadline
                if task_end > deadline_min:
                    continue
                task.scheduled_start = _to_hhmm(slot_start)
                task.scheduled_end = _to_hhmm(task_end)
                task.status = "scheduled"
                # Consume the slot
                if task_end + break_minutes < slot_end:
                    slot[0] = task_end + break_minutes
                else:
                    slot[0] = slot_end  # exhausted
                placed = True
                break

        if not placed:
            task.status = "unschedulable"
            reason = (
                f"No free slot of {duration}m before deadline {task.deadline or work_end}"
            )
            unschedulable.append({"task": task, "reason": reason})

    return tasks, unschedulable


def check_conflicts(tasks: List[Task]) -> List[dict]:
    """Return list of conflict dicts for any overlapping scheduled tasks."""
    scheduled = [
        t for t in tasks
        if t.scheduled_start and t.scheduled_end
    ]
    scheduled.sort(key=lambda t: _to_minutes(t.scheduled_start))

    conflicts = []
    for i in range(len(scheduled) - 1):
        a, b = scheduled[i], scheduled[i + 1]
        a_end = _to_minutes(a.scheduled_end)
        b_start = _to_minutes(b.scheduled_start)
        if a_end > b_start:
            conflicts.append({
                "task_a": a.name,
                "task_b": b.name,
                "overlap_minutes": a_end - b_start,
            })
    return conflicts
