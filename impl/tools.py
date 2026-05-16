"""
Tool runner — implements api.tools.AbstractToolRunner.

Each tool function receives Pydantic-validated inputs and returns a Pydantic
output model serialised to dict. Validation errors are caught and returned
as {"error": "..."} so Claude always gets a usable tool_result.
"""

from pydantic import ValidationError

from api.memory import AbstractSessionManager
from api.models import (
    GetScheduleOutput,
    MoveTaskInput,
    MoveTaskOutput,
    ParseAndAddTasksInput,
    ParseAndAddTasksOutput,
    RemoveTaskInput,
    RemoveTaskOutput,
    ScheduleTasksOutput,
    Task,
    TaskDetail,
    UpdatePreferencesInput,
    UpdatePreferencesOutput,
)
from api.scheduler import AbstractScheduler
from api.tools import AbstractToolRunner
from impl.scheduler import EDFScheduler, _to_hhmm, _to_min


class ToolRunner(AbstractToolRunner):
    def __init__(self, scheduler: AbstractScheduler | None = None):
        self._scheduler = scheduler or EDFScheduler()

    def run(self, name: str, inputs: dict, session: AbstractSessionManager) -> dict:
        handlers = {
            "parse_and_add_tasks": self._parse_and_add_tasks,
            "schedule_tasks": self._schedule_tasks,
            "move_task": self._move_task,
            "remove_task": self._remove_task,
            "get_schedule": self._get_schedule,
            "update_preferences": self._update_preferences,
        }
        if name not in handlers:
            return {"error": f"Unknown tool: {name}"}
        try:
            return handlers[name](inputs, session)
        except ValidationError as e:
            return {"error": f"Input validation failed: {e.errors(include_url=False)}"}
        except Exception as e:
            return {"error": str(e)}

    # ── Tool implementations ───────────────────────────────────────────────────

    def _parse_and_add_tasks(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        validated = ParseAndAddTasksInput.model_validate(inputs)
        added = []
        for ti in validated.tasks:
            task = Task(**ti.model_dump())
            session.add_task(task)
            added.append(task.name)
        session.save()
        return ParseAndAddTasksOutput(
            added_tasks=added,
            total_tasks=len(session.state.tasks),
            note="Call schedule_tasks to assign time slots.",
        ).model_dump()

    def _schedule_tasks(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        result = self._scheduler.schedule(
            session.state.tasks,
            session.state.preferences,
        )
        session.replace_tasks(session.state.tasks)
        session.save()
        prefs = session.state.preferences
        return ScheduleTasksOutput(
            scheduled=result.scheduled,
            unschedulable=result.unschedulable,
            conflicts=result.conflicts,
            work_window=f"{prefs.work_start} – {prefs.work_end}",
        ).model_dump()

    def _move_task(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        validated = MoveTaskInput.model_validate(inputs)
        task = session.find_task(validated.task_name)
        if task is None:
            return {"error": f"Task '{validated.task_name}' not found."}

        start_min = _to_min(validated.new_start_time)
        end_min = start_min + task.duration_minutes
        task.scheduled_start = validated.new_start_time
        task.scheduled_end = _to_hhmm(end_min)
        task.pinned = True
        task.status = "scheduled"

        warnings: list[str] = []
        prefs = session.state.preferences
        ws = _to_min(prefs.work_start)
        we = _to_min(prefs.work_end)
        if start_min < ws or end_min > we:
            warnings.append(
                f"Task placed outside work window ({prefs.work_start}–{prefs.work_end})"
            )

        # Reschedule non-pinned tasks around the newly pinned one
        self._schedule_tasks({}, session)

        return MoveTaskOutput(
            moved=task.name,
            new_slot=f"{task.scheduled_start} – {task.scheduled_end}",
            warnings=warnings,
        ).model_dump()

    def _remove_task(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        validated = RemoveTaskInput.model_validate(inputs)
        removed = session.remove_task(validated.task_name)
        if not removed:
            return {"error": f"Task '{validated.task_name}' not found."}
        session.save()
        return RemoveTaskOutput(
            removed=validated.task_name,
            remaining_tasks=len(session.state.tasks),
        ).model_dump()

    def _get_schedule(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        prefs = session.state.preferences
        tasks_sorted = sorted(
            session.state.tasks,
            key=lambda t: (t.scheduled_start or "99:99", t.name),
        )
        return GetScheduleOutput(
            date=prefs.date,
            work_window=f"{prefs.work_start} – {prefs.work_end}",
            tasks=[
                TaskDetail(
                    name=t.name,
                    duration_minutes=t.duration_minutes,
                    priority=t.priority,
                    deadline=t.deadline,
                    scheduled_start=t.scheduled_start,
                    scheduled_end=t.scheduled_end,
                    pinned=t.pinned,
                    status=t.status,
                    notes=t.notes,
                )
                for t in tasks_sorted
            ],
        ).model_dump()

    def _update_preferences(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        validated = UpdatePreferencesInput.model_validate(inputs)
        updated_prefs = session.update_preferences(validated.model_dump(exclude_none=True))
        session.save()
        return UpdatePreferencesOutput(
            updated_preferences=validated.model_dump(exclude_none=True),
            current_preferences=updated_prefs.model_dump(),
        ).model_dump()
