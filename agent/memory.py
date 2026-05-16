"""Session state management — persists tasks, schedule, and preferences across turns."""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional


@dataclass
class Task:
    name: str
    duration_minutes: int
    priority: int  # 1–5, 5 = highest
    deadline: Optional[str] = None  # "HH:MM" or None
    notes: Optional[str] = None
    scheduled_start: Optional[str] = None  # "HH:MM"
    scheduled_end: Optional[str] = None    # "HH:MM"
    pinned: bool = False  # True = user manually placed, skip in auto-scheduler
    status: str = "pending"  # pending | scheduled | unschedulable


@dataclass
class SessionState:
    tasks: list = field(default_factory=list)
    preferences: dict = field(default_factory=lambda: {
        "work_start": "09:00",
        "work_end": "18:00",
        "break_minutes": 5,
        "date": str(date.today()),
    })
    # Full Claude message history — drives multi-turn context
    conversation_history: list = field(default_factory=list)


class SessionManager:
    def __init__(self, session_file: Optional[str] = None):
        self._file = session_file
        self.state = SessionState()
        if session_file and os.path.exists(session_file):
            self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self):
        with open(self._file) as f:
            raw = json.load(f)
        self.state.tasks = [Task(**t) for t in raw.get("tasks", [])]
        self.state.preferences = raw.get("preferences", self.state.preferences)
        self.state.conversation_history = raw.get("conversation_history", [])

    def save(self):
        if not self._file:
            return
        os.makedirs(os.path.dirname(self._file), exist_ok=True)
        with open(self._file, "w") as f:
            json.dump(
                {
                    "tasks": [asdict(t) for t in self.state.tasks],
                    "preferences": self.state.preferences,
                    "conversation_history": self.state.conversation_history,
                },
                f,
                indent=2,
            )

    # ── Task helpers ───────────────────────────────────────────────────────────

    def add_task(self, task: Task):
        # Replace if same name already exists
        self.state.tasks = [t for t in self.state.tasks if t.name.lower() != task.name.lower()]
        self.state.tasks.append(task)

    def find_task(self, name: str) -> Optional[Task]:
        name_l = name.lower()
        return next((t for t in self.state.tasks if t.name.lower() == name_l), None)

    def remove_task(self, name: str) -> bool:
        before = len(self.state.tasks)
        self.state.tasks = [t for t in self.state.tasks if t.name.lower() != name.lower()]
        return len(self.state.tasks) < before

    # ── Conversation history ───────────────────────────────────────────────────

    def add_message(self, role: str, content):
        self.state.conversation_history.append({"role": role, "content": content})

    def get_history(self) -> list:
        return self.state.conversation_history
