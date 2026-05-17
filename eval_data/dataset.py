"""
25-case eval dataset for the Weekly Planner agent.

Categories
──────────
  tool_selection  (7)  Was the right tool called on the right turn?
  tool_params     (6)  Were the right arguments extracted from natural language?
  multi_turn      (5)  Does the agent maintain correct state across N turns?
  final_answer    (4)  Does the response text faithfully reflect tool outputs?
  edge_case       (3)  Graceful handling of impossible / ambiguous inputs.

Coverage
────────
  Single-turn:  ts_01–07, tp_01–06, ec_01–03          (16 cases)
  Multi-turn :  mt_01–05, fa_01–04 + ts_02/03/04/05   (12 cases contain ≥2 turns)
"""

from __future__ import annotations

import datetime

from eval_data.schemas import (
    AnswerCheck,
    EvalDatapoint,
    ExpectedToolCall,
    ParamCheck,
    SessionCheck,
)

TODAY: str = datetime.date.today().isoformat()
TOMORROW: str = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()


# ── TOOL SELECTION (ts_) ────────────────────────────────────────────────────────
# These cases verify that the agent picks the *right tool* for a given intent.
# They do not deeply test parameter values — that's tp_*.

TS_01 = EvalDatapoint(
    id="ts_01",
    category="tool_selection",
    description="Single add: agent should call parse_and_add_tasks (not schedule_tasks)",
    turns=["Add a 2-hour gym session to my schedule today."],
    expected_tool_calls=[
        ExpectedToolCall(
            tool="parse_and_add_tasks",
            turn_index=0,
            param_checks=[
                ParamCheck("tasks.0.duration_minutes", "eq", 120, "2 hours → 120 minutes"),
                ParamCheck("tasks.0.name", "contains", "gym", "name contains 'gym'"),
            ],
        )
    ],
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["added", "gym", "created"],
            description="response acknowledges the add",
        )
    ],
    session_checks=[
        SessionCheck(
            "Gym task present in session",
            lambda s: any("gym" in t.name.lower() for t in s.state.tasks),
        )
    ],
    metrics=["tool_selection_accuracy", "tool_param_accuracy", "answer_keyword_recall", "session_state_accuracy"],
    notes="Baseline single-turn add. Add intent must NOT trigger schedule_tasks automatically.",
)

TS_02 = EvalDatapoint(
    id="ts_02",
    category="tool_selection",
    description="Explicit schedule request after add: agent should call schedule_tasks on turn 2",
    turns=[
        "Add reading (1h) and cooking (30 min).",
        "Now schedule everything for today.",
    ],
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="schedule_tasks", turn_index=1),
    ],
    session_checks=[
        SessionCheck(
            "Both tasks scheduled",
            lambda s: sum(1 for t in s.state.tasks if t.status == "scheduled") >= 2,
        )
    ],
    metrics=["tool_selection_accuracy", "session_state_accuracy"],
    notes="Agent must not auto-schedule on add; schedule_tasks fires only on explicit request.",
)

TS_03 = EvalDatapoint(
    id="ts_03",
    category="tool_selection",
    description="Move a task to a specific time: agent should call move_task (not schedule_tasks)",
    turns=[
        "Add standup (30 min, priority 3).",
        "Schedule it.",
        "Move standup to 2pm.",
    ],
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="schedule_tasks", turn_index=1),
        ExpectedToolCall(
            tool="move_task",
            turn_index=2,
            param_checks=[
                ParamCheck("new_start_time", "eq", "14:00", "2pm → 14:00"),
                ParamCheck("task_name", "contains", "standup", "task_name references standup"),
            ],
        ),
    ],
    session_checks=[
        SessionCheck(
            "Standup pinned at 14:00",
            lambda s: any(
                "standup" in t.name.lower() and t.scheduled_start == "14:00"
                for t in s.state.tasks
            ),
        )
    ],
    metrics=["tool_selection_accuracy", "tool_param_accuracy", "session_state_accuracy"],
    notes="Move intent must pick move_task, not reschedule via schedule_tasks. Tests 12h→24h time parsing.",
)

TS_04 = EvalDatapoint(
    id="ts_04",
    category="tool_selection",
    description="Remove a task: agent should call remove_task (not move_task or add)",
    turns=[
        "Add yoga (1h, priority 2).",
        "Actually, cancel yoga — remove it from my schedule.",
    ],
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(
            tool="remove_task",
            turn_index=1,
            param_checks=[
                ParamCheck("task_name", "contains", "yoga", "task_name references yoga"),
            ],
        ),
    ],
    session_checks=[
        SessionCheck(
            "Yoga absent from session",
            lambda s: not any("yoga" in t.name.lower() for t in s.state.tasks),
        )
    ],
    metrics=["tool_selection_accuracy", "tool_param_accuracy", "session_state_accuracy"],
    notes="Delete intent ('cancel', 'remove') must route to remove_task, not move_task.",
)

TS_05 = EvalDatapoint(
    id="ts_05",
    category="tool_selection",
    description="View schedule on empty session: agent should call get_schedule",
    turns=["What's my schedule for today?"],
    expected_tool_calls=[
        ExpectedToolCall(tool="get_schedule", turn_index=0),
    ],
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["no tasks", "nothing", "empty", "haven't added", "add"],
            description="response acknowledges there is nothing scheduled",
        )
    ],
    session_checks=[
        SessionCheck("Session remains empty", lambda s: len(s.state.tasks) == 0),
    ],
    metrics=["tool_selection_accuracy", "answer_keyword_recall", "graceful_failure_rate"],
    notes="Empty schedule must not crash or hallucinate tasks. Agent must call get_schedule, not return from memory.",
)

TS_06 = EvalDatapoint(
    id="ts_06",
    category="tool_selection",
    description="Update working hours: agent should call update_preferences",
    turns=["Change my working hours to 8am until 6pm."],
    expected_tool_calls=[
        ExpectedToolCall(
            tool="update_preferences",
            turn_index=0,
            param_checks=[
                ParamCheck("work_start", "eq", "08:00", "8am → 08:00"),
                ParamCheck("work_end", "eq", "18:00", "6pm → 18:00"),
            ],
        )
    ],
    session_checks=[
        SessionCheck(
            "work_start = 08:00",
            lambda s: s.state.preferences.work_start == "08:00",
        ),
        SessionCheck(
            "work_end = 18:00",
            lambda s: s.state.preferences.work_end == "18:00",
        ),
    ],
    metrics=["tool_selection_accuracy", "tool_param_accuracy", "session_state_accuracy"],
    notes="Preference update intent. Verifies 12h→24h conversion for both boundaries.",
)

TS_07 = EvalDatapoint(
    id="ts_07",
    category="tool_selection",
    description="Outdoor activity: agent should proactively call get_weather before answering",
    turns=["I want to go for a run today. What's the best time based on the weather?"],
    expected_tool_calls=[
        ExpectedToolCall(
            tool="get_weather",
            turn_index=0,
            param_checks=[
                ParamCheck("date", "eq", TODAY, "date defaults to today"),
            ],
        ),
    ],
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["weather", "temperature", "conditions", "forecast", "rain", "wind", "sunny"],
            description="response references actual weather conditions from the forecast",
        )
    ],
    metrics=["tool_selection_accuracy", "tool_param_accuracy", "answer_keyword_recall"],
    notes=(
        "Agent must proactively fetch weather for outdoor activities (run, cycle, walk, garden). "
        "If location is not set, agent may ask for it — that's acceptable."
    ),
)


# ── TOOL PARAMETERS (tp_) ────────────────────────────────────────────────────────
# These cases stress-test natural language → tool argument extraction.

TP_01 = EvalDatapoint(
    id="tp_01",
    category="tool_params",
    description="Fractional hour duration: '2.5 hours' must become duration_minutes=150",
    turns=["Add: deep focus session (2.5 hours, priority 4)."],
    expected_tool_calls=[
        ExpectedToolCall(
            tool="parse_and_add_tasks",
            turn_index=0,
            param_checks=[
                ParamCheck("tasks.0.duration_minutes", "eq", 150, "2.5h = 150 min"),
                ParamCheck("tasks.0.priority", "eq", 4, "priority 4"),
            ],
        )
    ],
    session_checks=[
        SessionCheck(
            "Task with duration_minutes=150 in session",
            lambda s: any(t.duration_minutes == 150 for t in s.state.tasks),
        )
    ],
    metrics=["tool_param_accuracy", "session_state_accuracy"],
    notes="Common parsing failure: 2.5h → 2 min, 25 min, or 250 min. Must be exactly 150.",
)

TP_02 = EvalDatapoint(
    id="tp_02",
    category="tool_params",
    description="Natural language priority signals: 'critical / top priority' → priority=5",
    turns=["Add: critical production hotfix (1 hour). This is top priority — must happen first today."],
    expected_tool_calls=[
        ExpectedToolCall(
            tool="parse_and_add_tasks",
            turn_index=0,
            param_checks=[
                ParamCheck("tasks.0.priority", "eq", 5, "'critical'/'top priority' → 5"),
                ParamCheck("tasks.0.duration_minutes", "eq", 60, "1 hour = 60 min"),
            ],
        )
    ],
    session_checks=[
        SessionCheck(
            "Task with priority=5 added",
            lambda s: any(t.priority == 5 for t in s.state.tasks),
        )
    ],
    metrics=["tool_param_accuracy", "session_state_accuracy"],
    notes="Agent must map 'critical'/'top priority' to 5, not 3 or 4. Borderline: 'urgent' → 4 or 5 are both acceptable.",
)

TP_03 = EvalDatapoint(
    id="tp_03",
    category="tool_params",
    description="Deadline from natural language: 'due by noon' → deadline='12:00'",
    turns=["Add: submit quarterly report (1.5 hours). It's due by noon today."],
    expected_tool_calls=[
        ExpectedToolCall(
            tool="parse_and_add_tasks",
            turn_index=0,
            param_checks=[
                ParamCheck("tasks.0.deadline", "eq", "12:00", "'noon' → deadline='12:00'"),
                ParamCheck("tasks.0.duration_minutes", "eq", 90, "1.5h = 90 min"),
            ],
        )
    ],
    session_checks=[
        SessionCheck(
            "Task with deadline=12:00",
            lambda s: any(t.deadline == "12:00" for t in s.state.tasks),
        )
    ],
    metrics=["tool_param_accuracy", "session_state_accuracy"],
    notes="'Noon', '12pm', and '12:00' should all produce deadline='12:00'. Also tests fractional hours (90 min).",
)

TP_04 = EvalDatapoint(
    id="tp_04",
    category="tool_params",
    description="Bulk add: 3 tasks with distinct durations, priorities, and a deadline — all extracted correctly",
    turns=[
        "Add these tasks: standup call (15 min, priority 3), code review (45 min, priority 4), "
        "write deployment notes (30 min, low priority, deadline 5pm)."
    ],
    expected_tool_calls=[
        ExpectedToolCall(
            tool="parse_and_add_tasks",
            turn_index=0,
            param_checks=[
                ParamCheck("tasks.0.duration_minutes", "eq", 15, "standup = 15 min"),
                ParamCheck("tasks.0.priority", "eq", 3, "standup priority = 3"),
                ParamCheck("tasks.1.duration_minutes", "eq", 45, "code review = 45 min"),
                ParamCheck("tasks.1.priority", "eq", 4, "code review priority = 4"),
                ParamCheck("tasks.2.duration_minutes", "eq", 30, "deployment notes = 30 min"),
                ParamCheck("tasks.2.deadline", "eq", "17:00", "5pm → 17:00"),
                ParamCheck("tasks.2.priority", "in", [1, 2], "low priority → 1 or 2"),
            ],
        )
    ],
    session_checks=[
        SessionCheck("3 tasks in session", lambda s: len(s.state.tasks) >= 3),
    ],
    metrics=["tool_param_accuracy", "session_state_accuracy"],
    notes=(
        "Bulk extraction. Array ordering must match input order. "
        "Failure mode: tasks merged, durations swapped, or deadline missed."
    ),
)

TP_05 = EvalDatapoint(
    id="tp_05",
    category="tool_params",
    description="Move to tomorrow at a specific time: move_task must include both new_start_time AND date",
    turns=[
        "Add: team planning (1h).",
        "Schedule it.",
        "Move team planning to tomorrow at 3pm.",
    ],
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="schedule_tasks", turn_index=1),
        ExpectedToolCall(
            tool="move_task",
            turn_index=2,
            param_checks=[
                ParamCheck("new_start_time", "eq", "15:00", "3pm → 15:00"),
                ParamCheck("date", "eq", TOMORROW, f"'tomorrow' → {TOMORROW}"),
                ParamCheck("task_name", "contains", "planning", "task_name references planning"),
            ],
        ),
    ],
    session_checks=[
        SessionCheck(
            "Planning task date is tomorrow",
            lambda s: any(
                "planning" in t.name.lower() and t.date == TOMORROW
                for t in s.state.tasks
            ),
        )
    ],
    metrics=["tool_param_accuracy", "session_state_accuracy"],
    notes=(
        "Cross-day move: agent must pass BOTH new_start_time='15:00' AND date=TOMORROW. "
        "Omitting date leaves the task on today — a silent correctness bug."
    ),
)

TP_06 = EvalDatapoint(
    id="tp_06",
    category="tool_params",
    description="Timezone from city name: 'I'm in Tokyo' → timezone='Asia/Tokyo' (IANA)",
    turns=["I'm working from Tokyo today. Update my timezone accordingly."],
    expected_tool_calls=[
        ExpectedToolCall(
            tool="update_preferences",
            turn_index=0,
            param_checks=[
                ParamCheck("timezone", "eq", "Asia/Tokyo", "Tokyo → Asia/Tokyo IANA"),
            ],
        )
    ],
    session_checks=[
        SessionCheck(
            "timezone = Asia/Tokyo",
            lambda s: s.state.preferences.timezone == "Asia/Tokyo",
        )
    ],
    metrics=["tool_param_accuracy", "session_state_accuracy"],
    notes=(
        "Must produce a valid IANA name ('Asia/Tokyo'), not a display name like "
        "'Japan Standard Time'. Also tests that other prefs are untouched."
    ),
)


# ── MULTI-TURN (mt_) ─────────────────────────────────────────────────────────────
# State must be consistent and correctly updated across every turn.

MT_01 = EvalDatapoint(
    id="mt_01",
    category="multi_turn",
    description="3-turn flow: add tasks → schedule → move one; other task must be unaffected",
    turns=[
        "Add: code review (1h, priority 4) and testing (45 min, priority 3).",
        "Schedule both.",
        "Move testing to 3pm.",
    ],
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="schedule_tasks", turn_index=1),
        ExpectedToolCall(
            tool="move_task",
            turn_index=2,
            param_checks=[
                ParamCheck("new_start_time", "eq", "15:00", "3pm → 15:00"),
                ParamCheck("task_name", "contains", "testing", "targets testing"),
            ],
        ),
    ],
    session_checks=[
        SessionCheck(
            "Testing pinned at 15:00",
            lambda s: any(
                "testing" in t.name.lower() and t.scheduled_start == "15:00"
                for t in s.state.tasks
            ),
        ),
        SessionCheck(
            "Code review still in session and scheduled",
            lambda s: any(
                "code review" in t.name.lower() and t.status == "scheduled"
                for t in s.state.tasks
            ),
        ),
    ],
    metrics=["tool_selection_accuracy", "tool_param_accuracy", "session_state_accuracy"],
    notes="Moving one task must not evict or corrupt the other. Key regression: move wipes the full task list.",
)

MT_02 = EvalDatapoint(
    id="mt_02",
    category="multi_turn",
    description="Add → remove one task → reschedule remaining; removed task must not reappear",
    turns=[
        "Add: meeting (1h), lunch (45 min), exercise (1h). Schedule all.",
        "Remove lunch.",
        "Reschedule everything.",
    ],
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="schedule_tasks", turn_index=0),   # agent may inline both
        ExpectedToolCall(
            tool="remove_task",
            turn_index=1,
            param_checks=[
                ParamCheck("task_name", "contains", "lunch", "targets lunch"),
            ],
        ),
        ExpectedToolCall(tool="schedule_tasks", turn_index=2),
    ],
    session_checks=[
        SessionCheck(
            "Lunch absent from session",
            lambda s: not any("lunch" in t.name.lower() for t in s.state.tasks),
        ),
        SessionCheck(
            "Meeting and exercise remain scheduled",
            lambda s: all(
                t.status == "scheduled"
                for t in s.state.tasks
                if "meeting" in t.name.lower() or "exercise" in t.name.lower()
            ),
        ),
    ],
    metrics=["tool_selection_accuracy", "tool_param_accuracy", "session_state_accuracy"],
    notes="Remove must not cascade. Reschedule on turn 3 must pick up the correct two-task set.",
)

MT_03 = EvalDatapoint(
    id="mt_03",
    category="multi_turn",
    description="Preference change on turn 1 must affect scheduling on turn 2",
    turns=[
        "Set my work hours to 10:00 am to 2:00 pm.",
        "Add: analysis (2h) and report writing (1h). Schedule both.",
    ],
    expected_tool_calls=[
        ExpectedToolCall(
            tool="update_preferences",
            turn_index=0,
            param_checks=[
                ParamCheck("work_start", "eq", "10:00"),
                ParamCheck("work_end", "eq", "14:00"),
            ],
        ),
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=1),
        ExpectedToolCall(tool="schedule_tasks", turn_index=1),
    ],
    preferences={"work_start": "09:00", "work_end": "17:00"},
    session_checks=[
        SessionCheck(
            "work_start updated to 10:00",
            lambda s: s.state.preferences.work_start == "10:00",
        ),
        SessionCheck(
            "All scheduled tasks start at or after 10:00",
            lambda s: all(
                t.scheduled_start >= "10:00"
                for t in s.state.tasks
                if t.status == "scheduled" and t.scheduled_start
            ),
        ),
        SessionCheck(
            "No task scheduled after 14:00",
            lambda s: all(
                t.scheduled_end <= "14:00"
                for t in s.state.tasks
                if t.status == "scheduled" and t.scheduled_end
            ),
        ),
    ],
    metrics=["tool_selection_accuracy", "tool_param_accuracy", "session_state_accuracy"],
    notes="Multi-turn state coupling: preferences saved in turn 1 must be read by scheduler in turn 2.",
)

MT_04 = EvalDatapoint(
    id="mt_04",
    category="multi_turn",
    description="Move a task that doesn't exist: agent should flag it without corrupting session",
    turns=[
        "Add: deep work (2h, priority 5). Schedule.",
        "Move 'standup' to 11am.",  # standup was never added
    ],
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="schedule_tasks", turn_index=0),
    ],
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["not found", "don't have", "no task", "doesn't exist", "can't find", "cannot find"],
            description="agent tells the user the task wasn't found",
        )
    ],
    session_checks=[
        SessionCheck(
            "Deep work still in session (not lost)",
            lambda s: any("deep work" in t.name.lower() for t in s.state.tasks),
        )
    ],
    metrics=["answer_keyword_recall", "graceful_failure_rate", "session_state_accuracy"],
    notes="remove_task / move_task should return a clear error; agent must relay it, not silently succeed.",
)

MT_05 = EvalDatapoint(
    id="mt_05",
    category="multi_turn",
    description="5-turn conversation: add, status check, add more, schedule, move — full coherence test",
    turns=[
        "Add: email replies (30 min) and team sync (1h).",
        "What tasks do I have so far?",
        "Also add: documentation (2h, priority 4).",
        "Schedule everything.",
        "Move email replies to 9am.",
    ],
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="get_schedule", turn_index=1),
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=2),
        ExpectedToolCall(tool="schedule_tasks", turn_index=3),
        ExpectedToolCall(
            tool="move_task",
            turn_index=4,
            param_checks=[
                ParamCheck("new_start_time", "eq", "09:00", "9am → 09:00"),
                ParamCheck("task_name", "contains", "email", "targets email replies"),
            ],
        ),
    ],
    session_checks=[
        SessionCheck(
            "3 tasks in session",
            lambda s: len(s.state.tasks) == 3,
        ),
        SessionCheck(
            "Email replies pinned at 09:00",
            lambda s: any(
                "email" in t.name.lower() and t.scheduled_start == "09:00"
                for t in s.state.tasks
            ),
        ),
        SessionCheck(
            "All 3 tasks scheduled",
            lambda s: sum(1 for t in s.state.tasks if t.status == "scheduled") == 3,
        ),
    ],
    metrics=["tool_selection_accuracy", "tool_param_accuracy", "session_state_accuracy"],
    notes="Longest conversation in the dataset. State from turn 0 must remain correct through turn 4.",
)


# ── FINAL ANSWER (fa_) ─────────────────────────────────────────────────────────
# Correctness and faithfulness of the agent's text output.

FA_01 = EvalDatapoint(
    id="fa_01",
    category="final_answer",
    description="Schedule output must use 12h AM/PM format as specified in the system prompt",
    turns=[
        "Add: morning stand-up (30 min) and deep focus (2h).",
        "Schedule them.",
        "Show me the full schedule.",
    ],
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="schedule_tasks", turn_index=1),
        ExpectedToolCall(tool="get_schedule", turn_index=2),
    ],
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["AM", "PM"],
            excludes=[],
            description="response uses 12h AM/PM, not bare 24h HH:MM",
        )
    ],
    metrics=["tool_selection_accuracy", "answer_keyword_recall"],
    notes="System prompt requires 12h output. Regression: agent reverts to 24h after a model update.",
)

FA_02 = EvalDatapoint(
    id="fa_02",
    category="final_answer",
    description="Partial scheduling: response must name which tasks couldn't be scheduled",
    turns=["Add: taskA (3h), taskB (3h), taskC (3h). Schedule all."],
    preferences={"work_start": "09:00", "work_end": "14:00"},   # 5h window, 9h total
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["taskC", "taskB", "taskA", "couldn't", "cannot", "not fit", "unschedulable", "left out"],
            description="response explicitly names task(s) that couldn't fit",
        )
    ],
    session_checks=[
        SessionCheck(
            "At least one task is unschedulable",
            lambda s: any(t.status == "unschedulable" for t in s.state.tasks),
        )
    ],
    metrics=["answer_keyword_recall", "graceful_failure_rate", "session_state_accuracy"],
    notes="Agent must not silently drop tasks. User must be told which specific tasks were left out.",
)

FA_03 = EvalDatapoint(
    id="fa_03",
    category="final_answer",
    description="Deadline impossible: response must explain WHY (deadline + duration mismatch)",
    turns=["Add: marathon report (6 hours, deadline 11:00). Schedule it."],
    preferences={"work_start": "09:00", "work_end": "18:00"},
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["deadline", "11:00", "11 AM", "6 hours", "cannot", "can't", "impossible", "not fit"],
            description="response references both the deadline and why the task doesn't fit",
        )
    ],
    session_checks=[
        SessionCheck(
            "Report marked unschedulable",
            lambda s: any(
                "report" in t.name.lower() and t.status == "unschedulable"
                for t in s.state.tasks
            ),
        )
    ],
    metrics=["answer_keyword_recall", "graceful_failure_rate", "session_state_accuracy"],
    notes=(
        "6h task starting at 09:00 finishes at 15:00 — 4h past the 11:00 deadline. "
        "Response must explain the WHY, not just say 'unschedulable'."
    ),
)

FA_04 = EvalDatapoint(
    id="fa_04",
    category="final_answer",
    description="Move confirmation: response must state the correct new time (2:30 PM)",
    turns=[
        "Add: product demo (1h).",
        "Schedule it.",
        "Move the product demo to 2:30pm.",
    ],
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="schedule_tasks", turn_index=1),
        ExpectedToolCall(
            tool="move_task",
            turn_index=2,
            param_checks=[
                ParamCheck("new_start_time", "eq", "14:30", "2:30pm → 14:30"),
                ParamCheck("task_name", "contains", "demo", "targets product demo"),
            ],
        ),
    ],
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["2:30", "14:30", "2:30 PM"],
            description="confirmation explicitly states the new start time",
        )
    ],
    session_checks=[
        SessionCheck(
            "Demo scheduled at 14:30",
            lambda s: any(
                "demo" in t.name.lower() and t.scheduled_start == "14:30"
                for t in s.state.tasks
            ),
        )
    ],
    metrics=["tool_param_accuracy", "answer_keyword_recall", "session_state_accuracy"],
    notes="Tests both correct param extraction (14:30) AND that the confirmation echoes the actual time.",
)


# ── EDGE CASES (ec_) ──────────────────────────────────────────────────────────
# Unusual or adversarial inputs the agent must handle gracefully.

EC_01 = EvalDatapoint(
    id="ec_01",
    category="edge_case",
    description="Task longer than max_chunk_minutes auto-splits; response should mention blocks/sessions",
    turns=[
        "Add: deep research (4 hours, priority 5).",
        "Schedule it.",
    ],
    preferences={"work_start": "09:00", "work_end": "18:00", "max_chunk_minutes": 90},
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="schedule_tasks", turn_index=1),
    ],
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["block", "chunk", "split", "session", "focus", "break"],
            description="response mentions that the task was divided into blocks",
        )
    ],
    session_checks=[
        SessionCheck(
            "Research task exists and is scheduled",
            lambda s: any(
                "research" in t.name.lower() and t.status == "scheduled"
                for t in s.state.tasks
            ),
        )
    ],
    metrics=["answer_keyword_recall", "session_state_accuracy"],
    notes=(
        "EDFScheduler splits tasks > max_chunk_minutes into focus blocks. "
        "4h / 90min = ~3 blocks. Agent must communicate splitting, not just report one start time."
    ),
)

EC_02 = EvalDatapoint(
    id="ec_02",
    category="edge_case",
    description="Remove a task that doesn't exist: graceful error, session state unchanged",
    turns=["Remove the 'budget review' from my schedule."],
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["not found", "don't have", "no task", "doesn't exist", "can't find", "not in"],
            description="response tells the user the task wasn't found",
        )
    ],
    session_checks=[
        SessionCheck("Session remains empty", lambda s: len(s.state.tasks) == 0),
    ],
    metrics=["answer_keyword_recall", "graceful_failure_rate", "session_state_accuracy"],
    notes=(
        "remove_task tool returns an error for unknown task names. "
        "Agent must surface the error — not silently succeed or add a placeholder."
    ),
)

EC_03 = EvalDatapoint(
    id="ec_03",
    category="edge_case",
    description="Mathematically impossible deadline: 8h task with 10:00 deadline in 9h window",
    turns=["Add: very long report (8 hours, deadline 10:00). Schedule it."],
    preferences={"work_start": "09:00", "work_end": "18:00"},
    expected_tool_calls=[
        ExpectedToolCall(tool="parse_and_add_tasks", turn_index=0),
        ExpectedToolCall(tool="schedule_tasks", turn_index=0),
    ],
    answer_checks=[
        AnswerCheck(
            turn_index=-1,
            contains_any=["8 hours", "deadline", "10:00", "10 AM", "cannot", "can't", "impossible", "not fit"],
            description="response explains the impossibility clearly",
        )
    ],
    session_checks=[
        SessionCheck(
            "Report marked unschedulable",
            lambda s: any(
                "report" in t.name.lower() and t.status == "unschedulable"
                for t in s.state.tasks
            ),
        )
    ],
    metrics=["answer_keyword_recall", "graceful_failure_rate", "session_state_accuracy"],
    notes=(
        "Mathematically impossible: 8h task starting at 09:00 ends at 17:00 — 7h after the 10:00 deadline. "
        "Stresses both scheduler correctness and agent communication quality."
    ),
)


# ── Full dataset ───────────────────────────────────────────────────────────────

ALL_DATAPOINTS: list[EvalDatapoint] = [
    # tool_selection (7)
    TS_01, TS_02, TS_03, TS_04, TS_05, TS_06, TS_07,
    # tool_params (6)
    TP_01, TP_02, TP_03, TP_04, TP_05, TP_06,
    # multi_turn (5)
    MT_01, MT_02, MT_03, MT_04, MT_05,
    # final_answer (4)
    FA_01, FA_02, FA_03, FA_04,
    # edge_case (3)
    EC_01, EC_02, EC_03,
]

assert len(ALL_DATAPOINTS) == 25, f"Expected 25 datapoints, got {len(ALL_DATAPOINTS)}"