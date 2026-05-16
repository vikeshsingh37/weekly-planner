"""
Tool interface: abstract runner + Claude tool schema definitions.

The schema definitions live here (not in impl) because they are part of the
contract — what tools exist, their names, and their input shapes are determined
at the interface level. Impl provides the execution logic.
"""

from abc import ABC, abstractmethod

from .memory import AbstractSessionManager

# ── Claude tool_use schema definitions ────────────────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
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
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "duration_minutes": {
                                "type": "integer",
                                "description": "Estimated duration in minutes",
                            },
                            "priority": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "1 = low, 5 = critical",
                            },
                            "deadline": {
                                "type": "string",
                                "description": "Hard deadline HH:MM (24h). Omit if none.",
                            },
                            "notes": {"type": "string"},
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
                    "description": "Name of the task to move (case-insensitive)",
                },
                "new_start_time": {
                    "type": "string",
                    "description": "New start time HH:MM (24h)",
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
                    "description": "Name of the task to remove (case-insensitive)",
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
        "description": "Update work-hours or break-gap preferences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "work_start": {"type": "string", "description": "Start of workday HH:MM"},
                "work_end": {"type": "string", "description": "End of workday HH:MM"},
                "break_minutes": {
                    "type": "integer",
                    "description": "Gap between tasks in minutes",
                },
            },
            "required": [],
        },
    },
]


# ── Abstract runner ────────────────────────────────────────────────────────────

class AbstractToolRunner(ABC):
    @abstractmethod
    def run(self, name: str, inputs: dict, session: AbstractSessionManager) -> dict:
        """
        Execute the named tool with validated inputs against the given session.

        Returns a plain dict that is JSON-serialised and sent back to Claude
        as a tool_result. Must never raise — return {"error": "..."} instead.
        """
