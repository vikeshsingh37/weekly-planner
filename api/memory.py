"""Abstract session manager interface."""

from abc import ABC, abstractmethod
from typing import Optional

from .models import Preferences, SessionState, Task


class AbstractSessionManager(ABC):
    @property
    @abstractmethod
    def state(self) -> SessionState:
        """Current session state (read-only view — mutate via methods below)."""

    @abstractmethod
    def add_task(self, task: Task) -> None:
        """Add or replace a task. Deduplicates by task.id, not by name."""

    @abstractmethod
    def find_task(self, name: str) -> Optional[Task]:
        """Return first task matching name (case-insensitive), or None."""

    @abstractmethod
    def find_task_by_id(self, task_id: str) -> Optional[Task]:
        """Return task with the given UUID, or None."""

    @abstractmethod
    def remove_task(self, name: str) -> bool:
        """Remove first task matching name (case-insensitive). Returns True if found."""

    @abstractmethod
    def remove_task_by_id(self, task_id: str) -> bool:
        """Remove task with the given UUID. Returns True if found."""

    @abstractmethod
    def replace_tasks(self, tasks: list[Task]) -> None:
        """Replace the full task list (used after the scheduler runs)."""

    @abstractmethod
    def update_preferences(self, updates: dict) -> Preferences:
        """Merge updates into preferences and return the updated Preferences."""

    @abstractmethod
    def add_message(self, role: str, content: object) -> None:
        """Append a message to conversation history."""

    @abstractmethod
    def get_history(self) -> list[dict]:
        """Return full conversation history."""

    @abstractmethod
    def save(self) -> None:
        """Persist state to durable storage."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all state (used in tests and /reset commands)."""

    def reload(self) -> None:
        """Re-read state from durable storage. No-op for in-memory implementations."""