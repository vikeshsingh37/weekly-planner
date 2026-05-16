"""
Session managers — implements api.memory.AbstractSessionManager.

_BaseSessionManager   : in-memory state + threading.Lock (shared logic)
JSONSessionManager    : persists to a local JSON file
PostgresSessionManager: persists to a Postgres JSONB column (see postgres_memory.py)
"""

import json
import os
import threading
from abc import abstractmethod
from typing import Optional

from api.memory import AbstractSessionManager
from api.models import Preferences, SessionState, Task


class _BaseSessionManager(AbstractSessionManager):
    """Thread-safe in-memory state. Subclasses implement _persist() and _load()."""

    def __init__(self):
        self._state = SessionState()
        self._lock = threading.Lock()

    # ── AbstractSessionManager ─────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        return self._state

    def add_task(self, task: Task) -> None:
        with self._lock:
            self._state.tasks = [t for t in self._state.tasks if t.id != task.id]
            self._state.tasks.append(task)

    def find_task(self, name: str) -> Optional[Task]:
        nl = name.lower()
        return next((t for t in self._state.tasks if t.name.lower() == nl), None)

    def find_task_by_id(self, task_id: str) -> Optional[Task]:
        return next((t for t in self._state.tasks if t.id == task_id), None)

    def remove_task(self, name: str) -> bool:
        with self._lock:
            nl = name.lower()
            for i, t in enumerate(self._state.tasks):
                if t.name.lower() == nl:
                    self._state.tasks.pop(i)
                    return True
        return False

    def remove_task_by_id(self, task_id: str) -> bool:
        with self._lock:
            before = len(self._state.tasks)
            self._state.tasks = [t for t in self._state.tasks if t.id != task_id]
            return len(self._state.tasks) < before

    def replace_tasks(self, tasks: list[Task]) -> None:
        with self._lock:
            self._state.tasks = tasks

    def update_preferences(self, updates: dict) -> Preferences:
        with self._lock:
            current = self._state.preferences.model_dump()
            current.update({k: v for k, v in updates.items() if v is not None})
            self._state.preferences = Preferences(**current)
            return self._state.preferences

    def add_message(self, role: str, content: object) -> None:
        with self._lock:
            self._state.conversation_history.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        return self._state.conversation_history

    def reset(self) -> None:
        with self._lock:
            self._state = SessionState()
        self.save()

    # ── Persistence — subclasses override these ────────────────────────────────

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def reload(self) -> None: ...

    # ── Shared serialisation helper ────────────────────────────────────────────

    def _snapshot(self) -> dict:
        """Return a serialisable snapshot of current state (lock-safe)."""
        with self._lock:
            return self._state.model_dump()

    def _restore(self, raw: dict) -> None:
        self._state = SessionState.model_validate(raw)


class JSONSessionManager(_BaseSessionManager):
    def __init__(self, session_file: Optional[str] = None):
        super().__init__()
        self._file = session_file
        if session_file and os.path.exists(session_file):
            self._load()

    def save(self) -> None:
        if not self._file:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._file)), exist_ok=True)
        with open(self._file, "w") as f:
            json.dump(self._snapshot(), f, indent=2)

    def reload(self) -> None:
        if self._file and os.path.exists(self._file):
            self._load()

    def _load(self) -> None:
        with open(self._file) as f:
            self._restore(json.load(f))