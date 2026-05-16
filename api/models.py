"""
Canonical Pydantic data models — the source of truth for every data shape
that crosses a boundary in this system (tools, scheduler, session state).
"""

import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _validate_hhmm(v: Optional[str]) -> Optional[str]:
    if v is not None and not _TIME_RE.match(v):
        raise ValueError(f"Expected HH:MM format, got '{v}'")
    return v


# ── Core domain models ─────────────────────────────────────────────────────────

class Task(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    duration_minutes: int = Field(gt=0, le=600, description="Max 10 h per task")
    priority: int = Field(ge=1, le=5, description="1 = lowest, 5 = highest")
    deadline: Optional[str] = Field(default=None, description="HH:MM hard deadline")
    notes: Optional[str] = Field(default=None, max_length=500)
    scheduled_start: Optional[str] = None
    scheduled_end: Optional[str] = None
    pinned: bool = False
    status: Literal["pending", "scheduled", "unschedulable"] = "pending"

    @field_validator("deadline", "scheduled_start", "scheduled_end", mode="before")
    @classmethod
    def validate_time_field(cls, v: Optional[str]) -> Optional[str]:
        return _validate_hhmm(v)


class Preferences(BaseModel):
    work_start: str = "09:00"
    work_end: str = "18:00"
    break_minutes: int = Field(default=5, ge=0, le=60)
    date: str = Field(default_factory=lambda: str(date.today()))

    @field_validator("work_start", "work_end", mode="before")
    @classmethod
    def validate_time_field(cls, v: str) -> str:
        return _validate_hhmm(v)  # type: ignore[return-value]

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
    start: str
    end: str


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

    @field_validator("deadline", mode="before")
    @classmethod
    def validate_deadline(cls, v: Optional[str]) -> Optional[str]:
        return _validate_hhmm(v)


class ParseAndAddTasksInput(BaseModel):
    tasks: list[TaskInput] = Field(min_length=1)


class ParseAndAddTasksOutput(BaseModel):
    added_tasks: list[str]
    total_tasks: int
    note: str


class ScheduleTasksOutput(BaseModel):
    scheduled: list[ScheduledSlot]
    unschedulable: list[UnschedulableItem]
    conflicts: list[ConflictItem]
    work_window: str


class MoveTaskInput(BaseModel):
    task_name: str = Field(min_length=1)
    new_start_time: str

    @field_validator("new_start_time", mode="before")
    @classmethod
    def validate_start(cls, v: str) -> str:
        return _validate_hhmm(v)  # type: ignore[return-value]


class MoveTaskOutput(BaseModel):
    moved: str
    new_slot: str
    warnings: list[str] = Field(default_factory=list)


class RemoveTaskInput(BaseModel):
    task_name: str = Field(min_length=1)


class RemoveTaskOutput(BaseModel):
    removed: str
    remaining_tasks: int


class TaskDetail(BaseModel):
    name: str
    duration_minutes: int
    priority: int
    deadline: Optional[str]
    scheduled_start: Optional[str]
    scheduled_end: Optional[str]
    pinned: bool
    status: str
    notes: Optional[str]


class GetScheduleOutput(BaseModel):
    date: Optional[str]
    work_window: str
    tasks: list[TaskDetail]


class UpdatePreferencesInput(BaseModel):
    work_start: Optional[str] = None
    work_end: Optional[str] = None
    break_minutes: Optional[int] = Field(default=None, ge=0, le=60)

    @field_validator("work_start", "work_end", mode="before")
    @classmethod
    def validate_times(cls, v: Optional[str]) -> Optional[str]:
        return _validate_hhmm(v)


class UpdatePreferencesOutput(BaseModel):
    updated_preferences: dict
    current_preferences: dict
