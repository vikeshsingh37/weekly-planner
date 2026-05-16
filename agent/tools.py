"""
Tool implementations — deterministic Python functions the agent calls.

Each function returns a plain dict that gets serialized to JSON for Claude.
No LLM logic here: inputs are validated, state is mutated, results are returned.
"""

from dataclasses import asdict
from typing import Optional

from .memory import SessionManager, Task
from .scheduler import schedule, check_conflicts


# ── Tool definitions (Claude tool_use schema) ──────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "parse_and_add_tasks",
        "description": (
            "Add one or more tasks to the session. "
            "Call this whenever the user mentions tasks they need to do. "
            "Estimate duration and priority from context if not explicit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short descriptive task name",
                            },
                            "duration_minutes": {
                                "type": "integer",
                                "description": "Estimated duration in minutes",
                            },
                            "priority": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "1=low, 5=critical",
                            },
                            "deadline": {
                                "type": "string",
                                "description": "Hard deadline in HH:MM (24h). Omit if none.",
                            },
                            "notes": {
                                "type": "string",
                                "description": "Optional extra context",
                            },
                        },
                        "required": ["name", "duration_minutes", "priority"],
                    },
                }
            },
            "required": ["tasks"],
        },
    },
    {
        "name": "schedule_tasks",
        "description": (
            "Run the scheduler on all pending tasks and assign them to time slots. "
            "Call this after adding or removing tasks, or when the user asks to (re)schedule."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move_task",
        "description": (
            "Pin a task to a specific start time chosen by the user, "
            "then reschedule the remaining tasks around it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_name": {
                    "type": "string",
                    "description": "Name of the task to move (case-insensitive match)",
                },
                "new_start_time": {
                    "type": "string",
                    "description": "New start time in HH:MM (24h)",
                },
            },
            "required": ["task_name", "new_start_time"],
        },
    },
    {
        "name": "remove_task",
        "description": "Remove a task from the session entirely.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_name": {
                    "type": "string",
                    "description": "Name of the task to remove (case-insensitive match)",
                }
            },
            "required": ["task_name"],
        },
    },
    {
        "name": "get_schedule",
        "description": (
            "Return the current tasks and schedule. "
            "Call this to check state before reporting to the user."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "update_preferences",
        "description": "Update work-hours or break preferences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "work_start": {
                    "type": "string",
                    "description": "Start of workday HH:MM",
                },
                "work_end": {
                    "type": "string",
                    "description": "End of workday HH:MM",
                },
                "break_minutes": {
                    "type": "integer",
                    "description": "Gap between tasks in minutes",
                },
            },
            "required": [],
        },
    },
]


# ── Tool executors ─────────────────────────────────────────────────────────────

def run_tool(name: str, inputs: dict, session: SessionManager) -> dict:
    """Dispatch tool call and return result dict."""
    handlers = {
        "parse_and_add_tasks": _parse_and_add_tasks,
        "schedule_tasks": _schedule_tasks,
        "move_task": _move_task,
        "remove_task": _remove_task,
        "get_schedule": _get_schedule,
        "update_preferences": _update_preferences,
    }
    if name not in handlers:
        return {"error": f"Unknown tool: {name}"}
    return handlers[name](inputs, session)


def _parse_and_add_tasks(inputs: dict, session: SessionManager) -> dict:
    added = []
    for raw in inputs.get("tasks", []):
        task = Task(
            name=raw["name"],
            duration_minutes=int(raw["duration_minutes"]),
            priority=int(raw["priority"]),
            deadline=raw.get("deadline"),
            notes=raw.get("notes"),
        )
        session.add_task(task)
        added.append(task.name)
    session.save()
    return {
        "added_tasks": added,
        "total_tasks": len(session.state.tasks),
        "note": "Call schedule_tasks to assign time slots.",
    }


def _schedule_tasks(inputs: dict, session: SessionManager) -> dict:
    prefs = session.state.preferences
    updated_tasks, unschedulable = schedule(
        session.state.tasks,
        work_start=prefs["work_start"],
        work_end=prefs["work_end"],
        break_minutes=prefs.get("break_minutes", 5),
    )
    session.state.tasks = updated_tasks

    conflicts = check_conflicts(session.state.tasks)
    session.save()

    scheduled = [t for t in session.state.tasks if t.status == "scheduled"]
    result = {
        "scheduled": [
            {"name": t.name, "start": t.scheduled_start, "end": t.scheduled_end}
            for t in scheduled
        ],
        "unschedulable": [
            {"name": u["task"].name, "reason": u["reason"]}
            for u in unschedulable
        ],
        "conflicts": conflicts,
        "work_window": f"{prefs['work_start']} – {prefs['work_end']}",
    }
    return result


def _move_task(inputs: dict, session: SessionManager) -> dict:
    task_name = inputs["task_name"]
    new_start = inputs["new_start_time"]
    task = session.find_task(task_name)
    if task is None:
        return {"error": f"Task '{task_name}' not found."}

    from .scheduler import _to_minutes, _to_hhmm
    start_min = _to_minutes(new_start)
    end_min = start_min + task.duration_minutes
    prefs = session.state.preferences

    task.scheduled_start = new_start
    task.scheduled_end = _to_hhmm(end_min)
    task.pinned = True
    task.status = "scheduled"

    # Warn if outside work window
    warnings = []
    ws = _to_minutes(prefs["work_start"])
    we = _to_minutes(prefs["work_end"])
    if start_min < ws or end_min > we:
        warnings.append(
            f"Task placed outside work window ({prefs['work_start']}–{prefs['work_end']})"
        )

    # Reschedule the rest around the pinned task
    _schedule_tasks({}, session)

    result = {
        "moved": task_name,
        "new_slot": f"{task.scheduled_start} – {task.scheduled_end}",
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _remove_task(inputs: dict, session: SessionManager) -> dict:
    task_name = inputs["task_name"]
    removed = session.remove_task(task_name)
    if not removed:
        return {"error": f"Task '{task_name}' not found."}
    session.save()
    return {"removed": task_name, "remaining_tasks": len(session.state.tasks)}


def _get_schedule(inputs: dict, session: SessionManager) -> dict:
    prefs = session.state.preferences
    tasks_out = []
    for t in sorted(
        session.state.tasks,
        key=lambda t: (t.scheduled_start or "99:99", t.name),
    ):
        tasks_out.append(
            {
                "name": t.name,
                "duration_minutes": t.duration_minutes,
                "priority": t.priority,
                "deadline": t.deadline,
                "scheduled_start": t.scheduled_start,
                "scheduled_end": t.scheduled_end,
                "pinned": t.pinned,
                "status": t.status,
                "notes": t.notes,
            }
        )
    return {
        "date": prefs.get("date"),
        "work_window": f"{prefs['work_start']} – {prefs['work_end']}",
        "tasks": tasks_out,
    }


def _update_preferences(inputs: dict, session: SessionManager) -> dict:
    prefs = session.state.preferences
    changed = {}
    for key in ("work_start", "work_end", "break_minutes"):
        if key in inputs and inputs[key] is not None:
            prefs[key] = inputs[key]
            changed[key] = inputs[key]
    session.save()
    return {"updated_preferences": changed, "current_preferences": prefs}
