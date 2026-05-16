"""Abstract scheduler interface."""

from abc import ABC, abstractmethod

from .models import ConflictItem, Preferences, ScheduleResult, Task


class AbstractScheduler(ABC):
    @abstractmethod
    def schedule(
        self,
        tasks: list[Task],
        preferences: Preferences,
    ) -> ScheduleResult:
        """
        Assign time slots to tasks in-place and return a ScheduleResult.

        Implementations must:
        - Not move tasks with pinned=True
        - Update task.scheduled_start, task.scheduled_end, task.status
        - Return every unschedulable task with a human-readable reason
        """

    @abstractmethod
    def check_conflicts(self, tasks: list[Task]) -> list[ConflictItem]:
        """Return overlap details for any two scheduled tasks that share time."""
