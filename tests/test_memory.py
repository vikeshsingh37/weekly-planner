"""Unit tests for impl.memory.JSONSessionManager."""
import pytest
from api.models import Task
from impl.memory import JSONSessionManager


def _task(name="gym", duration=60, priority=3):
    return Task(name=name, duration_minutes=duration, priority=priority)


class TestJSONSessionManager:
    def setup_method(self):
        self.session = JSONSessionManager(session_file=None)

    def test_initial_state_is_empty(self):
        assert self.session.state.tasks == []
        assert self.session.state.conversation_history == []

    def test_add_and_find_task(self):
        self.session.add_task(_task("gym"))
        found = self.session.find_task("gym")
        assert found is not None
        assert found.name == "gym"

    def test_find_task_case_insensitive(self):
        self.session.add_task(_task("Gym"))
        assert self.session.find_task("gym") is not None

    def test_find_missing_task_returns_none(self):
        assert self.session.find_task("nonexistent") is None

    def test_remove_existing_task(self):
        self.session.add_task(_task("gym"))
        assert self.session.remove_task("gym") is True
        assert self.session.find_task("gym") is None

    def test_remove_missing_task_returns_false(self):
        assert self.session.remove_task("ghost") is False

    def test_add_task_deduplicates_by_id(self):
        t = _task("gym")
        self.session.add_task(t)
        t.duration_minutes = 90
        self.session.add_task(t)
        assert len(self.session.state.tasks) == 1
        assert self.session.state.tasks[0].duration_minutes == 90

    def test_multiple_tasks_stored(self):
        self.session.add_task(_task("gym"))
        self.session.add_task(_task("reading"))
        assert len(self.session.state.tasks) == 2

    def test_update_preferences_single_field(self):
        self.session.update_preferences({"work_start": "08:00"})
        assert self.session.state.preferences.work_start == "08:00"

    def test_update_preferences_preserves_other_fields(self):
        original_end = self.session.state.preferences.work_end
        self.session.update_preferences({"work_start": "10:00"})
        assert self.session.state.preferences.work_end == original_end

    def test_update_preferences_none_values_ignored(self):
        original_start = self.session.state.preferences.work_start
        self.session.update_preferences({"work_start": None})
        assert self.session.state.preferences.work_start == original_start

    def test_add_message_and_get_history(self):
        self.session.add_message("user", "Hello")
        self.session.add_message("assistant", "Hi there")
        history = self.session.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there"}

    def test_replace_tasks(self):
        self.session.add_task(_task("old"))
        self.session.replace_tasks([_task("new")])
        assert len(self.session.state.tasks) == 1
        assert self.session.state.tasks[0].name == "new"

    def test_find_task_by_id(self):
        t = _task("gym")
        self.session.add_task(t)
        found = self.session.find_task_by_id(t.id)
        assert found is not None
        assert found.id == t.id

    def test_remove_task_by_id(self):
        t = _task("gym")
        self.session.add_task(t)
        assert self.session.remove_task_by_id(t.id) is True
        assert self.session.find_task("gym") is None