"""
EDF (Earliest-Deadline-First) scheduler — implements api.scheduler.AbstractScheduler.

Rules:
  1. Pinned tasks (user-placed) block time and are never moved.
  2. Remaining tasks are sorted: deadline ascending, then priority descending.
  3. Free slots are the complement of pinned-task intervals within the work window.
  4. Free slots are clamped to now_min so past time is never used.
  5. Short tasks (duration <= max_chunk_minutes) are placed whole or marked unschedulable.
  6. Long tasks (duration > max_chunk_minutes) are split into focus blocks of at most
     max_chunk_minutes each, with break_minutes gaps between blocks.
  7. Unschedulable tasks get a reason string — they are never silently dropped.
"""

from api.models import ConflictItem, Preferences, ScheduleResult, ScheduledSlot, Task, UnschedulableItem
from api.scheduler import AbstractScheduler


def _to_min(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def _to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _to_ampm(hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    period = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {period}"


def _ceil15(m: int) -> int:
    """Round minutes up to the nearest 15-minute boundary."""
    return ((m + 14) // 15) * 15


def _deadline_min(task: Task, work_end: int) -> int:
    return _to_min(task.deadline) if task.deadline else work_end


def _sort_key(task: Task, work_end: int) -> tuple:
    return (_deadline_min(task, work_end), 6 - task.priority)


def _place_chunked(
    dur: int,
    free: list[list[int]],
    deadline: int,
    gap: int,
    max_chunk: int,
) -> list[tuple[int, int]] | None:
    """
    Greedily place `dur` minutes across free slots in blocks of at most `max_chunk` minutes,
    with `gap` minutes between consecutive blocks. Mutates free slot cursors in place.
    Returns a list of (start_min, end_min) pairs, or None if unschedulable.
    """
    remaining = dur
    chunks: list[tuple[int, int]] = []

    for slot in free:
        while remaining > 0:
            slot[0] = _ceil15(slot[0])  # align to 15-min grid
            avail = slot[1] - slot[0]
            if avail <= 0:
                break  # slot exhausted, move to next
            chunk_dur = min(remaining, max_chunk, avail)
            chunk_start = slot[0]
            chunk_end = chunk_start + chunk_dur
            if chunk_end > deadline:
                return None  # can't meet deadline
            chunks.append((chunk_start, chunk_end))
            remaining -= chunk_dur
            slot[0] = chunk_end + gap

    return chunks if remaining == 0 else None


class EDFScheduler(AbstractScheduler):
    def schedule(
        self,
        tasks: list[Task],
        preferences: Preferences,
        now_min: int | None = None,
    ) -> ScheduleResult:
        ws = _to_min(preferences.work_start)
        we = _to_min(preferences.work_end)
        gap = preferences.break_minutes
        max_chunk = preferences.max_chunk_minutes  # 0 = no limit
        total = we - ws

        # Reset non-pinned tasks
        for t in tasks:
            if not t.pinned:
                t.scheduled_start = None
                t.scheduled_end = None
                t.chunks = []
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

        # Clamp free slots to current time so we never plan in the past
        if now_min is not None:
            free = [[max(s, now_min), e] for s, e in free if e > now_min]

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

            needs_chunking = max_chunk > 0 and dur > max_chunk

            if needs_chunking:
                # Split into focus blocks of at most max_chunk minutes
                chunk_pairs = _place_chunked(dur, free, deadline, gap, max_chunk)
                if chunk_pairs is None:
                    task.status = "unschedulable"
                    n_chunks = -(-dur // max_chunk)  # ceiling division
                    dl_str = _to_ampm(task.deadline or preferences.work_end)
                    unschedulable.append(UnschedulableItem(
                        name=task.name,
                        reason=(
                            f"Cannot fit {n_chunks} focus blocks ({max_chunk}m each) "
                            f"for {dur}m task before deadline {dl_str}"
                        ),
                    ))
                else:
                    task.chunks = [{"start": _to_hhmm(s), "end": _to_hhmm(e)} for s, e in chunk_pairs]
                    task.scheduled_start = task.chunks[0]["start"]
                    task.scheduled_end = task.chunks[-1]["end"]
                    task.status = "scheduled"
            else:
                # Short task: fit whole or mark unschedulable
                placed = False
                for slot in free:
                    start = _ceil15(slot[0])  # align to 15-min grid
                    avail = slot[1] - start
                    if avail >= dur and start + dur <= deadline:
                        task.scheduled_start = _to_hhmm(start)
                        task.scheduled_end = _to_hhmm(start + dur)
                        task.chunks = []
                        task.status = "scheduled"
                        slot[0] = start + dur + gap
                        placed = True
                        break
                if not placed:
                    task.status = "unschedulable"
                    dl_str = _to_ampm(task.deadline or preferences.work_end)
                    unschedulable.append(UnschedulableItem(
                        name=task.name,
                        reason=f"No free slot of {dur}m before deadline {dl_str}",
                    ))

        scheduled = [
            ScheduledSlot(
                name=t.name,
                start=t.scheduled_start,
                end=t.scheduled_end,
                chunks=t.chunks,
            )
            for t in tasks
            if t.status == "scheduled"
        ]
        return ScheduleResult(
            scheduled=scheduled,
            unschedulable=unschedulable,
            conflicts=self.check_conflicts(tasks),
        )

    def check_conflicts(self, tasks: list[Task]) -> list[ConflictItem]:
        # Collect actual occupied intervals (chunk-aware: chunked tasks don't own break time)
        intervals: list[tuple[int, int, str]] = []
        for t in tasks:
            if t.chunks:
                for c in t.chunks:
                    intervals.append((_to_min(c["start"]), _to_min(c["end"]), t.name))
            elif t.scheduled_start and t.scheduled_end:
                intervals.append((_to_min(t.scheduled_start), _to_min(t.scheduled_end), t.name))

        intervals.sort()
        conflicts: list[ConflictItem] = []
        for i in range(len(intervals) - 1):
            s1, e1, n1 = intervals[i]
            s2, e2, n2 = intervals[i + 1]
            if n1 != n2 and e1 > s2:
                conflicts.append(ConflictItem(
                    task_a=n1,
                    task_b=n2,
                    overlap_minutes=e1 - s2,
                ))
        return conflicts
