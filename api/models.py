"""
Canonical Pydantic data models — the source of truth for every data shape
that crosses a boundary in this system (tools, scheduler, session state).
"""

import os
import re
import uuid
from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _detect_local_timezone() -> str:
    """Return the IANA timezone name for the local system, falling back to UTC."""
    try:
        local_tz = datetime.now().astimezone().tzinfo
        if hasattr(local_tz, "key"):
            return local_tz.key
    except Exception:
        pass
    try:
        link = os.path.realpath("/etc/localtime")
        if "/zoneinfo/" in link:
            return link.split("/zoneinfo/", 1)[-1]
    except Exception:
        pass
    return "UTC"

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_hhmm(v: Optional[str]) -> Optional[str]:
    if v is not None and not _TIME_RE.match(v):
        raise ValueError(f"Expected HH:MM format, got '{v}'")
    return v


def _validate_date(v: Optional[str]) -> Optional[str]:
    if v is not None and not _DATE_RE.match(v):
        raise ValueError(f"Expected YYYY-MM-DD format, got '{v}'")
    return v


# ── Core domain models ─────────────────────────────────────────────────────────

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1, max_length=120)
    duration_minutes: int = Field(gt=0, le=600, description="Max 10 h per task")
    priority: int = Field(ge=1, le=5, description="1 = lowest, 5 = highest")
    deadline: Optional[str] = Field(default=None, description="HH:MM hard deadline")
    notes: Optional[str] = Field(default=None, max_length=500)
    # Which planning day this task belongs to (YYYY-MM-DD).
    # None is treated as the session's base date at scheduling time.
    date: Optional[str] = None
    scheduled_start: Optional[str] = None
    scheduled_end: Optional[str] = None
    pinned: bool = False
    status: Literal["pending", "scheduled", "unschedulable"] = "pending"
    chunks: list[dict] = Field(default_factory=list)

    @field_validator("deadline", "scheduled_start", "scheduled_end", mode="before")
    @classmethod
    def validate_time_field(cls, v: Optional[str]) -> Optional[str]:
        return _validate_hhmm(v)

    @field_validator("date", mode="before")
    @classmethod
    def validate_date_field(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date(v)


class Preferences(BaseModel):
    work_start: str = "09:00"
    work_end: str = "18:00"
    break_minutes: int = Field(default=5, ge=0, le=60)
    max_chunk_minutes: int = Field(default=90, ge=0, le=480)
    # Base date for the planning session (today by default).
    date: str = Field(default_factory=lambda: str(date.today()))
    timezone: str = Field(default_factory=_detect_local_timezone)
    # How many days to plan ahead (1 = today only, up to 7 for a full week).
    planning_days: int = Field(default=7, ge=1, le=7)
    # Location for weather forecasts
    location_name: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @field_validator("work_start", "work_end", mode="before")
    @classmethod
    def validate_time_field(cls, v: str) -> str:
        return _validate_hhmm(v)  # type: ignore[return-value]

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, KeyError):
            raise ValueError(f"Unknown timezone '{v}'")
        return v

    @model_validator(mode="after")
    def end_after_start(self) -> "Preferences":
        if self.work_start >= self.work_end:
            raise ValueError("work_end must be after work_start")
        return self


class SessionState(BaseModel):
    tasks: list[Task] = Field(default_factory=list)
    preferences: Preferences = Field(default_factory=Preferences)
    conversation_history: list[dict] = Field(default_factory=list)


# ── Scheduler I/O models ───────────────────────────────────────────────────────

class ScheduledSlot(BaseModel):
    name: str
    date: Optional[str] = None
    start: str
    end: str
    chunks: list[dict] = Field(default_factory=list)


class UnschedulableItem(BaseModel):
    name: str
    reason: str


class ConflictItem(BaseModel):
    task_a: str
    task_b: str
    overlap_minutes: int


class ScheduleResult(BaseModel):
    scheduled: list[ScheduledSlot]
    unschedulable: list[UnschedulableItem]
    conflicts: list[ConflictItem]


# ── Tool I/O models ────────────────────────────────────────────────────────────

class TaskInput(BaseModel):
    """Single task as provided by the LLM in parse_and_add_tasks."""
    name: str = Field(min_length=1, max_length=120)
    duration_minutes: int = Field(gt=0, le=600)
    priority: int = Field(ge=1, le=5)
    deadline: Optional[str] = None
    notes: Optional[str] = None
    date: Optional[str] = None  # YYYY-MM-DD; None = session base date

    @field_validator("deadline", mode="before")
    @classmethod
    def validate_deadline(cls, v: Optional[str]) -> Optional[str]:
        return _validate_hhmm(v)

    @field_validator("date", mode="before")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date(v)


class ParseAndAddTasksInput(BaseModel):
    tasks: list[TaskInput] = Field(min_length=1)


class ParseAndAddTasksOutput(BaseModel):
    added_tasks: list[str]
    rejected_tasks: list[str] = Field(default_factory=list)
    total_tasks: int
    note: str


class ScheduleTasksOutput(BaseModel):
    scheduled: list[ScheduledSlot]
    unschedulable: list[UnschedulableItem]
    conflicts: list[ConflictItem]
    work_window: str


class MoveTaskInput(BaseModel):
    task_name: str = Field(min_length=1)
    task_id: Optional[str] = None  # preferred when names are not unique
    new_start_time: str
    date: Optional[str] = None  # Move task to a different day (optional)

    @field_validator("new_start_time", mode="before")
    @classmethod
    def validate_start(cls, v: str) -> str:
        return _validate_hhmm(v)  # type: ignore[return-value]

    @field_validator("date", mode="before")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date(v)


class MoveTaskOutput(BaseModel):
    moved: str
    new_slot: str
    warnings: list[str] = Field(default_factory=list)


class RemoveTaskInput(BaseModel):
    task_name: str = Field(min_length=1)
    task_id: Optional[str] = None  # preferred when names are not unique


class RemoveTaskOutput(BaseModel):
    removed: str
    remaining_tasks: int


class TaskDetail(BaseModel):
    id: str
    name: str
    date: Optional[str]
    duration_minutes: int
    priority: int
    deadline: Optional[str]
    scheduled_start: Optional[str]
    scheduled_end: Optional[str]
    pinned: bool
    status: str
    notes: Optional[str]
    chunks: list[dict] = Field(default_factory=list)


class GetScheduleOutput(BaseModel):
    planning_horizon: str   # e.g. "2026-05-16 to 2026-05-20 (5 days)"
    work_window: str
    tasks: list[TaskDetail]


# WeatherInput is intentionally empty — location comes from session preferences.
class WeatherInput(BaseModel):
    pass


class UpdatePreferencesInput(BaseModel):
    work_start: Optional[str] = None
    work_end: Optional[str] = None
    break_minutes: Optional[int] = Field(default=None, ge=0, le=60)
    max_chunk_minutes: Optional[int] = Field(default=None, ge=0, le=480)
    timezone: Optional[str] = None
    planning_days: Optional[int] = Field(default=None, ge=1, le=7)
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @field_validator("work_start", "work_end", mode="before")
    @classmethod
    def validate_times(cls, v: Optional[str]) -> Optional[str]:
        return _validate_hhmm(v)

    @field_validator("timezone", mode="before")
    @classmethod
    def validate_tz(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, KeyError):
            raise ValueError(f"Unknown timezone '{v}'")
        return v


class UpdatePreferencesOutput(BaseModel):
    updated_preferences: dict
    current_preferences: dict