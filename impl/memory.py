"""
JSON-backed session manager — implements api.memory.AbstractSessionManager.

Swap this for a Redis or Postgres-backed implementation without touching
the agent or tools — they depend only on AbstractSessionManager.
"""

import json
import os
from typing import Optional

from api.memory import AbstractSessionManager
from api.models import Preferences, SessionState, Task


class JSONSessionManager(AbstractSessionManager):
    def __init__(self, session_file: Optional[str] = None):
        self._file = session_file
        self._state = SessionState()
        if session_file and os.path.exists(session_file):
            self._load()

    # ── AbstractSessionManager ─────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        return self._state

    def add_task(self, task: Task) -> None:
        self._state.tasks = [
            t for t in self._state.tasks if t.name.lower() != task.name.lower()
        ]
        self._state.tasks.append(task)

    def find_task(self, name: str) -> Optional[Task]:
        nl = name.lower()
        return next((t for t in self._state.tasks if t.name.lower() == nl), None)

    def remove_task(self, name: str) -> bool:
        before = len(self._state.tasks)
        self._state.tasks = [
            t for t in self._state.tasks if t.name.lower() != name.lower()
        ]
        return len(self._state.tasks) < before

    def replace_tasks(self, tasks: list[Task]) -> None:
        self._state.tasks = tasks

    def update_preferences(self, updates: dict) -> Preferences:
        current = self._state.preferences.model_dump()
        current.update({k: v for k, v in updates.items() if v is not None})
        self._state.preferences = Preferences(**current)
        return self._state.preferences

    def add_message(self, role: str, content: object) -> None:
        self._state.conversation_history.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        return self._state.conversation_history

    def save(self) -> None:
        if not self._file:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._file)), exist_ok=True)
        with open(self._file, "w") as f:
            json.dump(self._state.model_dump(), f, indent=2)

    def reset(self) -> None:
        self._state = SessionState()
        self.save()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        with open(self._file) as f:
            raw = json.load(f)
        self._state = SessionState.model_validate(raw)
