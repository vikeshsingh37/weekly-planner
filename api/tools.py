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
            "Estimate duration and priority from context if not explicit. "
            "Use the 'date' field when the user specifies a particular day (e.g. tomorrow, Wednesday)."
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
                            "date": {
                                "type": "string",
                                "description": (
                                    "Planning date YYYY-MM-DD. Omit for today. "
                                    "Use for tasks on a specific future day."
                                ),
                            },
                            "start_time": {
                                "type": "string",
                                "description": (
                                    "Fixed start time HH:MM (24h). "
                                    "Set this when the user specifies an exact time (e.g. standup at 09:30). "
                                    "The task will be pinned at this time; schedule_tasks will fit remaining tasks around it."
                                ),
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
            "Tasks are grouped by date and scheduled independently within each day's work window. "
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
                "task_id": {
                    "type": "string",
                    "description": (
                        "UUID of the task from get_schedule. "
                        "Use this when multiple tasks share the same name."
                    ),
                },
                "new_start_time": {
                    "type": "string",
                    "description": "New start time HH:MM (24h)",
                },
                "date": {
                    "type": "string",
                    "description": "Move task to a different day YYYY-MM-DD (optional).",
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
                },
                "task_id": {
                    "type": "string",
                    "description": (
                        "UUID of the task from get_schedule. "
                        "Use this when multiple tasks share the same name."
                    ),
                },
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
        "description": (
            "Update work-hours, break-gap, timezone, or location preferences. "
            "Pass location_name (city/region string) to set the user's location for weather — "
            "the server will geocode it automatically, so coordinates are never needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "work_start": {"type": "string", "description": "Start of workday HH:MM"},
                "work_end": {"type": "string", "description": "End of workday HH:MM"},
                "break_minutes": {
                    "type": "integer",
                    "description": "Gap between tasks in minutes",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. America/New_York, Asia/Kolkata",
                },
                "max_chunk_minutes": {
                    "type": "integer",
                    "description": (
                        "Maximum continuous focus block in minutes before a break is inserted. "
                        "Tasks longer than this are split automatically. 0 = no limit. Default 90."
                    ),
                },
                "location_name": {
                    "type": "string",
                    "description": (
                        "City or region name for weather lookups, e.g. 'Mumbai', 'San Francisco, CA'. "
                        "Server geocodes this automatically — do not ask for coordinates."
                    ),
                },
                "planning_days": {
                    "type": "integer",
                    "description": (
                        "Number of days to plan (1–7). 1 = today only, 7 = full week. "
                        "Update when the user wants to switch between daily and weekly planning."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Fetch the weather forecast for the user's location via Open-Meteo (up to 16 days ahead). "
            "Call this before scheduling any outdoor activity (running, walking, cycling, "
            "outdoor lunch, sports, gardening, etc.) to check conditions and find the best window. "
            "Returns hourly forecast for work hours, an overall outdoor conditions rating "
            "(good/moderate/poor), and the best outdoor time window. "
            "If location is not set, the tool returns an error — ask the user for their city."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date to fetch forecast for in YYYY-MM-DD format. Defaults to today if omitted.",
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
