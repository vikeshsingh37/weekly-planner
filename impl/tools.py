"""
Tool runner — implements api.tools.AbstractToolRunner.

Each tool function receives Pydantic-validated inputs and returns a Pydantic
output model serialised to dict. Validation errors are caught and returned
as {"error": "..."} so Claude always gets a usable tool_result.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_MAX_PLANNING_DAYS = 7  # tasks beyond this many days from today are rejected

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
    WeatherInput,
)
from api.scheduler import AbstractScheduler
from api.tools import AbstractToolRunner
from impl.scheduler import EDFScheduler, _to_ampm, _to_hhmm, _to_min


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
            "get_weather": self._get_weather,
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
        base_date = session.state.preferences.date
        today = date.today()
        max_date = today + timedelta(days=_MAX_PLANNING_DAYS - 1)

        added, rejected = [], []
        for ti in validated.tasks:
            task_date = date.fromisoformat(ti.date) if ti.date else today
            if task_date > max_date:
                rejected.append(
                    f"{ti.name} (date {ti.date} is beyond the "
                    f"{_MAX_PLANNING_DAYS}-day planning window; max allowed: {max_date})"
                )
                continue
            task = Task(**{**ti.model_dump(exclude={"start_time"}), "date": ti.date or base_date})
            if ti.start_time:
                start_min = _to_min(ti.start_time)
                task.scheduled_start = ti.start_time
                task.scheduled_end = _to_hhmm(start_min + task.duration_minutes)
                task.pinned = True
                task.status = "scheduled"
            session.add_task(task)
            added.append(task.name)

        session.save()
        return ParseAndAddTasksOutput(
            added_tasks=added,
            rejected_tasks=rejected,
            total_tasks=len(session.state.tasks),
            note="Call schedule_tasks to assign time slots for any unpinned tasks.",
        ).model_dump()

    def _schedule_tasks(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        prefs = session.state.preferences
        today = prefs.date

        # Group tasks by their planning date
        tasks_by_date: dict[str, list[Task]] = defaultdict(list)
        for task in session.state.tasks:
            tasks_by_date[task.date or today].append(task)

        all_scheduled = []
        all_unschedulable = []
        all_conflicts = []

        for date_str in sorted(tasks_by_date):
            day_tasks = tasks_by_date[date_str]
            # Only clamp to current time for today's tasks
            now_min = self._compute_now_min(prefs) if date_str == today else None
            result = self._scheduler.schedule(day_tasks, prefs, now_min=now_min)
            # Tag scheduled slots with their date
            for slot in result.scheduled:
                slot.date = date_str
            all_scheduled.extend(result.scheduled)
            all_unschedulable.extend(result.unschedulable)
            all_conflicts.extend(result.conflicts)

        session.replace_tasks(session.state.tasks)
        session.save()
        return ScheduleTasksOutput(
            scheduled=all_scheduled,
            unschedulable=all_unschedulable,
            conflicts=all_conflicts,
            work_window=f"{_to_ampm(prefs.work_start)} – {_to_ampm(prefs.work_end)}",
        ).model_dump()

    @staticmethod
    def _compute_now_min(prefs) -> int | None:
        """Return current minute-of-day in user's timezone, or None if not planning for today."""
        if prefs.current_time:
            # Eval mode: use the fixed simulated time so scheduler output is deterministic.
            h, m = map(int, prefs.current_time.split(":"))
            return h * 60 + m
        try:
            tz = ZoneInfo(prefs.timezone)
            now = datetime.now(tz)
            if str(now.date()) != prefs.date:
                return None
            return now.hour * 60 + now.minute
        except (ZoneInfoNotFoundError, KeyError, Exception):
            return None

    def _move_task(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        validated = MoveTaskInput.model_validate(inputs)
        task = (
            session.find_task_by_id(validated.task_id)
            if validated.task_id
            else session.find_task(validated.task_name)
        )
        if task is None:
            return {"error": f"Task '{validated.task_id or validated.task_name}' not found."}

        # Optionally change the task's date — enforce the same 7-day cap
        if validated.date:
            today = date.today()
            max_date = today + timedelta(days=_MAX_PLANNING_DAYS - 1)
            if date.fromisoformat(validated.date) > max_date:
                return {
                    "error": (
                        f"Date {validated.date} is beyond the {_MAX_PLANNING_DAYS}-day "
                        f"planning window (max allowed: {max_date})."
                    )
                }
            task.date = validated.date

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
                f"Task placed outside work window ({_to_ampm(prefs.work_start)}–{_to_ampm(prefs.work_end)})"
            )

        self._schedule_tasks({}, session)

        return MoveTaskOutput(
            moved=task.name,
            new_slot=f"{_to_ampm(task.scheduled_start)} – {_to_ampm(task.scheduled_end)}",
            warnings=warnings,
        ).model_dump()

    def _remove_task(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        validated = RemoveTaskInput.model_validate(inputs)
        if validated.task_id:
            task = session.find_task_by_id(validated.task_id)
            if task is None:
                return {"error": f"No task with ID '{validated.task_id}'."}
            removed_name = task.name
            session.remove_task_by_id(validated.task_id)
        else:
            task = session.find_task(validated.task_name)
            if task is None:
                return {"error": f"Task '{validated.task_name}' not found."}
            removed_name = task.name
            session.remove_task(validated.task_name)
        return RemoveTaskOutput(
            removed=removed_name,
            remaining_tasks=len(session.state.tasks),
        ).model_dump()

    def _get_schedule(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        prefs = session.state.preferences
        today = prefs.date
        days = prefs.planning_days

        # Build planning horizon label
        if days == 1:
            horizon = today
        else:
            from datetime import date, timedelta
            end_date = str(date.fromisoformat(today) + timedelta(days=days - 1))
            horizon = f"{today} to {end_date} ({days} days)"

        tasks_sorted = sorted(
            session.state.tasks,
            key=lambda t: (t.date or today, t.scheduled_start or "99:99", t.name),
        )
        return GetScheduleOutput(
            planning_horizon=horizon,
            work_window=f"{_to_ampm(prefs.work_start)} – {_to_ampm(prefs.work_end)}",
            tasks=[
                TaskDetail(
                    id=t.id,
                    name=t.name,
                    date=t.date or today,
                    duration_minutes=t.duration_minutes,
                    priority=t.priority,
                    deadline=t.deadline,
                    scheduled_start=t.scheduled_start,
                    scheduled_end=t.scheduled_end,
                    pinned=t.pinned,
                    status=t.status,
                    notes=t.notes,
                    chunks=t.chunks,
                )
                for t in tasks_sorted
            ],
        ).model_dump()

    def _update_preferences(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        validated = UpdatePreferencesInput.model_validate(inputs)
        updates = validated.model_dump(exclude_none=True)

        # If a location_name was given but no coordinates, geocode it server-side
        if "location_name" in updates and updates.get("latitude") is None:
            try:
                from impl.weather import geocode_city
                geo = geocode_city(updates["location_name"])
                if geo:
                    updates["latitude"] = geo["latitude"]
                    updates["longitude"] = geo["longitude"]
                    parts = [geo["name"]]
                    if geo.get("admin1"):
                        parts.append(geo["admin1"])
                    if geo.get("country"):
                        parts.append(geo["country"])
                    updates["location_name"] = ", ".join(parts)
            except Exception:
                pass

        updated_prefs = session.update_preferences(updates)
        session.save()
        return UpdatePreferencesOutput(
            updated_preferences=updates,
            current_preferences=updated_prefs.model_dump(),
        ).model_dump()

    def _get_weather(
        self, inputs: dict, session: AbstractSessionManager
    ) -> dict:
        prefs = session.state.preferences
        if prefs.latitude is None or prefs.longitude is None:
            return {
                "error": "Location not set.",
                "action": (
                    "Ask the user for their city or location so weather can be fetched. "
                    "They can also set it in the Settings panel in the UI."
                ),
            }
        try:
            from impl.weather import fetch_forecast, summarize_forecast
            date = inputs.get("date") or prefs.date
            raw = fetch_forecast(
                latitude=prefs.latitude,
                longitude=prefs.longitude,
                timezone=prefs.timezone,
                date=date,
            )
            return summarize_forecast(
                data=raw,
                work_start=prefs.work_start,
                work_end=prefs.work_end,
                location_name=prefs.location_name or "",
            )
        except Exception as exc:
            return {"error": f"Weather fetch failed: {exc}"}
