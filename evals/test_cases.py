"""
Eval test cases — each is a self-contained scenario the runner executes.

Case format:
    turns:   list of user messages (in order)
    checks:  list of assertion functions applied to the session after all turns
    name:    short identifier
    category: one of task_completion | hallucination | graceful_failure | memory
"""

from dataclasses import dataclass
from typing import Callable, List

from agent.memory import SessionManager


@dataclass
class Check:
    description: str
    fn: Callable[[SessionManager, List[str]], bool]  # (session, agent_responses) → pass


@dataclass
class EvalCase:
    name: str
    category: str
    turns: List[str]
    checks: List[Check]
    # Optional preference overrides applied before running
    preferences: dict = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _scheduled_count(session: SessionManager) -> int:
    return sum(1 for t in session.state.tasks if t.status == "scheduled")


def _task_names(session: SessionManager) -> List[str]:
    return [t.name.lower() for t in session.state.tasks]


def _no_conflicts(session: SessionManager) -> bool:
    from agent.scheduler import check_conflicts
    return len(check_conflicts(session.state.tasks)) == 0


def _agent_mentions(responses: List[str], keyword: str) -> bool:
    combined = " ".join(responses).lower()
    return keyword.lower() in combined


# ── Category 1: Task Completion Rate ──────────────────────────────────────────

TASK_COMPLETION_CASES = [
    EvalCase(
        name="tc_01_three_tasks_fit",
        category="task_completion",
        turns=[
            "I need to study ML for 2 hours, do gym for 1 hour, and write a report for 30 minutes.",
            "Schedule everything please.",
        ],
        checks=[
            Check(
                "All 3 tasks added to session",
                lambda s, r: len(s.state.tasks) >= 3,
            ),
            Check(
                "At least 3 tasks scheduled (8-hour window fits 3.5h of work)",
                lambda s, r: _scheduled_count(s) >= 3,
            ),
            Check(
                "No scheduling conflicts",
                lambda s, r: _no_conflicts(s),
            ),
        ],
    ),
    EvalCase(
        name="tc_02_five_tasks_fit",
        category="task_completion",
        turns=[
            "Add these tasks: email replies (30min), team standup (30min), "
            "code review (1 hour), lunch (45min, priority 3), deep work (2 hours, priority 5).",
            "Schedule them.",
        ],
        checks=[
            Check("5 tasks added", lambda s, r: len(s.state.tasks) >= 5),
            Check("All 5 scheduled (4h45m fits in 9h window)", lambda s, r: _scheduled_count(s) == 5),
            Check("No conflicts", lambda s, r: _no_conflicts(s)),
        ],
    ),
    EvalCase(
        name="tc_03_deadline_respected",
        category="task_completion",
        turns=[
            "Add: prepare presentation (1 hour, deadline 11:00) and reading (2 hours).",
            "Schedule everything.",
        ],
        checks=[
            Check(
                "Presentation ends by 11:00",
                lambda s, r: next(
                    (t for t in s.state.tasks if "presentation" in t.name.lower()),
                    None,
                ) is not None and next(
                    (t for t in s.state.tasks if "presentation" in t.name.lower()),
                ).scheduled_end is not None and next(
                    (t for t in s.state.tasks if "presentation" in t.name.lower()),
                ).scheduled_end <= "11:00",
            ),
        ],
    ),
    EvalCase(
        name="tc_04_partial_completion_reported",
        category="task_completion",
        turns=[
            "I have: task A (3h), task B (3h), task C (3h), task D (2h). Schedule all.",
        ],
        preferences={"work_start": "09:00", "work_end": "17:00"},  # 8h window, 11h tasks
        checks=[
            Check(
                "Some tasks scheduled and some reported unschedulable (11h > 8h window)",
                lambda s, r: _scheduled_count(s) > 0
                and any(t.status == "unschedulable" for t in s.state.tasks),
            ),
            Check(
                "Agent mentions tasks that couldn't fit",
                lambda s, r: any(
                    word in " ".join(r).lower()
                    for word in ["cannot", "couldn't", "unschedulable", "doesn't fit", "not fit", "no room"]
                ),
            ),
        ],
    ),
]


# ── Category 2: Hallucination on Tool Outputs ──────────────────────────────────

HALLUCINATION_CASES = [
    EvalCase(
        name="hal_01_schedule_reflects_state",
        category="hallucination",
        turns=[
            "Add: morning run (45min), breakfast (30min).",
            "What's my schedule?",
        ],
        checks=[
            Check(
                "Agent's schedule response cites only tasks that exist in state",
                lambda s, r: all(
                    any(t.name.lower() in resp.lower() for t in s.state.tasks)
                    or resp == r[0]  # first response might not mention tasks
                    for resp in r
                ),
            ),
            Check(
                "No tasks in session that weren't requested",
                lambda s, r: len(s.state.tasks) <= 2,
            ),
        ],
    ),
    EvalCase(
        name="hal_02_no_phantom_slots",
        category="hallucination",
        turns=[
            "I only have tasks: writing (4h) and coding (4h). Schedule them.",
            "What time does writing end?",
        ],
        preferences={"work_start": "09:00", "work_end": "18:00"},
        checks=[
            Check(
                "Writing task exists with a real scheduled_end time",
                lambda s, r: any(
                    t.name.lower() == "writing" and t.scheduled_end is not None
                    for t in s.state.tasks
                ),
            ),
            Check(
                "Scheduled times are within work window",
                lambda s, r: all(
                    t.scheduled_start is None or "09:00" <= t.scheduled_start <= "18:00"
                    for t in s.state.tasks
                ),
            ),
        ],
    ),
    EvalCase(
        name="hal_03_tool_output_matches_response",
        category="hallucination",
        turns=[
            "Schedule: meeting (1h, deadline 10:00), review (2h).",
            "Confirm the schedule.",
        ],
        checks=[
            Check(
                "Meeting appears in session scheduled before 10:00",
                lambda s, r: any(
                    t.name.lower() == "meeting"
                    and t.scheduled_end is not None
                    and t.scheduled_end <= "10:00"
                    for t in s.state.tasks
                ),
            ),
        ],
    ),
]


# ── Category 3: Graceful Failure ───────────────────────────────────────────────

GRACEFUL_FAILURE_CASES = [
    EvalCase(
        name="gf_01_impossible_schedule",
        category="graceful_failure",
        turns=[
            "I need to do: task1 (4h), task2 (4h), task3 (4h), task4 (4h). Schedule all.",
        ],
        preferences={"work_start": "09:00", "work_end": "13:00"},  # only 4h available
        checks=[
            Check(
                "Agent does not claim all 16h of tasks fit in 4h",
                lambda s, r: any(t.status == "unschedulable" for t in s.state.tasks),
            ),
            Check(
                "Agent communicates scheduling failure",
                lambda s, r: any(
                    word in " ".join(r).lower()
                    for word in ["cannot", "can't", "unable", "impossible", "not fit", "unschedulable", "no room"]
                ),
            ),
        ],
    ),
    EvalCase(
        name="gf_02_deadline_impossible",
        category="graceful_failure",
        turns=[
            "Add: long report (5 hours, deadline 11:00). Schedule it.",
        ],
        preferences={"work_start": "09:00", "work_end": "18:00"},
        checks=[
            Check(
                "Report is marked unschedulable (can't finish 5h report by 11:00 starting at 09:00)",
                lambda s, r: any(
                    "report" in t.name.lower() and t.status == "unschedulable"
                    for t in s.state.tasks
                ),
            ),
            Check(
                "Agent explains the deadline conflict",
                lambda s, r: any(
                    word in " ".join(r).lower()
                    for word in ["deadline", "11:00", "cannot", "can't", "fit"]
                ),
            ),
        ],
    ),
    EvalCase(
        name="gf_03_move_to_conflicting_slot",
        category="graceful_failure",
        turns=[
            "Add: standup (30min), deep work (2h). Schedule them.",
            "Move deep work to 09:00.",
            "Move standup to 09:15.",  # conflicts with deep work
        ],
        checks=[
            Check(
                "Agent warns about overlap or resolves conflict",
                lambda s, r: _agent_mentions(r[2:], "conflict")
                or _agent_mentions(r[2:], "overlap")
                or _no_conflicts(s),
            ),
        ],
    ),
    EvalCase(
        name="gf_04_empty_day",
        category="graceful_failure",
        turns=["What's my schedule for today?"],
        checks=[
            Check(
                "Agent handles empty schedule gracefully (no crash, no phantom tasks)",
                lambda s, r: len(s.state.tasks) == 0,
            ),
            Check(
                "Agent acknowledges nothing scheduled",
                lambda s, r: any(
                    word in " ".join(r).lower()
                    for word in ["no tasks", "nothing", "empty", "add", "start"]
                ),
            ),
        ],
    ),
]


# ── Category 4: Session Memory ──────────────────────────────────────────────────

MEMORY_CASES = [
    EvalCase(
        name="mem_01_task_persists_across_turns",
        category="memory",
        turns=[
            "Add gym (1 hour, priority 4).",
            "What tasks do I have?",
        ],
        checks=[
            Check(
                "Gym task present in session after 2 turns",
                lambda s, r: any("gym" in t.name.lower() for t in s.state.tasks),
            ),
            Check(
                "Agent mentions gym in second response",
                lambda s, r: "gym" in r[1].lower(),
            ),
        ],
    ),
    EvalCase(
        name="mem_02_schedule_update_persists",
        category="memory",
        turns=[
            "Add: study (2h), coding (1.5h). Schedule them.",
            "Move gym to 17:00.",   # gym doesn't exist — agent should flag it
            "Show my schedule.",
        ],
        checks=[
            Check(
                "Study and coding remain in session after all turns",
                lambda s, r: any("study" in t.name.lower() for t in s.state.tasks)
                and any("coding" in t.name.lower() for t in s.state.tasks),
            ),
        ],
    ),
    EvalCase(
        name="mem_03_remove_then_reschedule",
        category="memory",
        turns=[
            "Add: meeting (1h), lunch (45min), exercise (1h). Schedule.",
            "Remove lunch.",
            "Reschedule.",
        ],
        checks=[
            Check(
                "Lunch removed from session",
                lambda s, r: not any("lunch" in t.name.lower() for t in s.state.tasks),
            ),
            Check(
                "Meeting and exercise still present",
                lambda s, r: any("meeting" in t.name.lower() for t in s.state.tasks)
                and any("exercise" in t.name.lower() for t in s.state.tasks),
            ),
        ],
    ),
    EvalCase(
        name="mem_04_preference_change_affects_schedule",
        category="memory",
        turns=[
            "Change my work hours to 10:00–14:00.",
            "Add: analysis (3h). Schedule it.",
        ],
        checks=[
            Check(
                "Work start updated to 10:00",
                lambda s, r: s.state.preferences.get("work_start") == "10:00",
            ),
            Check(
                "Analysis starts at or after 10:00",
                lambda s, r: any(
                    "analysis" in t.name.lower()
                    and t.scheduled_start is not None
                    and t.scheduled_start >= "10:00"
                    for t in s.state.tasks
                ),
            ),
        ],
    ),
]


ALL_CASES = (
    TASK_COMPLETION_CASES
    + HALLUCINATION_CASES
    + GRACEFUL_FAILURE_CASES
    + MEMORY_CASES
)
