# Daily Planner Agent — Architecture Decision Document

## 1. What Was Built

A conversational AI agent that helps users plan their workday through natural language. The agent collects tasks, schedules them deterministically, and updates the plan interactively across a session.

**Core capabilities:**
- Parse tasks (name, duration, priority, deadline) from freeform text
- Schedule all tasks using Earliest-Deadline-First into available work-hour slots
- Move individual tasks to user-specified times and reschedule around them
- Remove tasks and compact the schedule
- Persist session state across turns so the user never has to repeat themselves

---

## 2. Framework Decision: Plain Python + Claude API

### Why not LangGraph?

| Criterion | LangGraph | Plain Python |
|-----------|-----------|--------------|
| Debugging | Requires understanding graph state machine | Linear call stack, trivial to trace |
| Reliability | Extra abstraction layer between tool call and result | Direct function calls, no intermediaries |
| Eval coverage | Graph nodes harder to unit test | Each tool is a pure function — fully testable |
| Dependency surface | Large; opinionated about state shape | `anthropic` + `python-dotenv` only |
| Overhead at 100k users | Graph serialization, node checkpointing | Stateless HTTP handler + JSON session |

**Decision: plain Python.** The agentic loop is 40 lines. Every tool is a pure `(inputs, session) → dict` function. There is no magic.

The one thing LangGraph would have added is built-in human-in-the-loop breakpoints. For this scope, a simple `confirm_before=True` flag would be enough if needed.

### Why not function calling via OpenAI?

Claude's tool_use API (structured tool inputs, `tool_result` messages) is better suited here because:
- Tool call results are typed JSON, not freeform text
- Claude is more conservative about hallucinating tool outputs when it knows the result is authoritative
- The multi-turn agentic loop is first-class in the API (not a workaround)

---

## 3. Tool Design

Five tools, all deterministic:

### `parse_and_add_tasks`
Accepts structured task data that Claude extracts from natural language. Claude handles the NLP (duration inference, priority estimation); this tool just writes to session state. Separating LLM extraction from state mutation means the extraction can be wrong without corrupting the schedule.

### `schedule_tasks`
Runs the EDF scheduler over all non-pinned tasks. Always called *after* adding/removing tasks. Deterministic: same inputs → same schedule every time. No LLM involvement in scheduling logic.

### `move_task`
Pins a task to a user-specified time slot, marks it `pinned=True`, then calls `schedule_tasks` to fill the remaining time around it. Pinned tasks are never moved by the auto-scheduler.

### `remove_task`
Removes a task by name (case-insensitive). Does not auto-reschedule — agent calls `schedule_tasks` afterward if needed.

### `get_schedule`
Read-only snapshot of session state. The agent is instructed to call this before reporting to the user, which prevents it from reporting stale or invented schedule data.

### `update_preferences`
Modifies work_start, work_end, break_minutes. These feed directly into the scheduler.

---

## 4. Scheduling Algorithm: Earliest Deadline First (EDF)

**Why EDF over other approaches:**

| Algorithm | Optimal? | Complexity | Preemptive? |
|-----------|----------|------------|-------------|
| Random | No | O(n) | No |
| Priority-only | No | O(n log n) | No |
| EDF (chosen) | Yes (for single resource) | O(n log n) | No |
| Integer Linear Programming | Yes (multi-constraint) | Exponential | N/A |

EDF is provably optimal for single-machine scheduling when all tasks must complete before their deadline and preemption isn't allowed — which matches the daily planning problem exactly.

**Implementation details:**
1. Pinned tasks block time first (user-explicit placement always wins)
2. Remaining tasks sorted: deadline ascending, then priority descending for ties
3. Free slots computed as complement of pinned-task intervals within work window
4. Tasks placed greedily into first slot that fits *and* allows deadline to be met
5. Unschedulable tasks are collected with a reason — never silently dropped

**Break handling:** a configurable gap (default 5 minutes) is inserted after each task before the next slot begins. This prevents back-to-back scheduling that looks good on paper but fails in practice.

---

## 5. Session Memory

```python
SessionState:
    tasks: List[Task]                  # source of truth for all tasks + schedule
    preferences: dict                  # work hours, break gap, date
    conversation_history: List[dict]   # full Claude message history
```

**Why store the full conversation history?** Claude's context window is the memory. No vector DB, no summarization pipeline, no embedding. The session state stores the *history* in the format the API expects — so multi-turn context is free. The cost is that very long sessions eventually hit the token limit (~200k tokens for claude-sonnet-4-6 ≈ 40–60 turns before truncation risk).

**Persistence:** serialized to JSON. For production, swap to Redis or Postgres — the `SessionManager` interface is the only thing that changes.

---

## 6. Production Considerations (100k+ Users)

### What breaks first

**Token budget per session.** At 100k users, long sessions (>30 turns) will start hitting limits or getting expensive. Fix: implement rolling window compression — summarize the first N turns into a compact state snapshot and drop the raw history.

**Cold starts.** Each new session re-sends the full system prompt + conversation history. At scale, prompt caching (Anthropic's cache_control feature) dramatically reduces cost for the stable system prompt.

**Scheduler correctness under concurrent edits.** The current scheduler is not thread-safe. If the same session is modified by two concurrent requests (unlikely in single-user scenario, but possible via API), race conditions can corrupt state. Fix: optimistic locking on session writes.

### What to monitor

| Metric | Alert threshold | Why |
|--------|----------------|-----|
| Tool call error rate | > 5% | Tool failure = broken schedule |
| Unschedulable task rate | > 20% | Users setting impossible plans |
| Session token count | > 150k | Approaching context limit |
| `schedule_tasks` latency | > 200ms | Scheduler performance regression |
| Hallucination check rate | > 2% | Agent citing non-existent tasks |

### What to fix first at scale

1. **Prompt caching** — biggest cost reduction, zero behavior change
2. **Session compression** — enables long-running sessions
3. **Persistent storage** — swap JSON files for a real DB
4. **Structured logging** — per-turn tool call traces for debugging

---

## 7. What Was Deliberately Not Built

| Feature | Why Not |
|---------|---------|
| Calendar API (Google Calendar, Outlook) | Requires OAuth per user, adds infra complexity, not needed to demo agent quality |
| Long-term memory across days | Out of scope — this is a single-day planner |
| Vector memory / semantic search | Overkill for <50 tasks per session |
| Multi-user sessions | Not required by the spec |
| Natural language time parsing ("tomorrow morning") | Dates scoped to today; times extracted by Claude's NLP |
| UI (web/mobile) | CLI sufficient to demonstrate agent quality |
| Constraint solver (ILP) | EDF is optimal for the single-machine case; ILP adds complexity with no benefit here |

---

## 8. Where the Agent Breaks

### Ambiguous task duration
*"I need to prepare for the meeting"* — Claude will estimate a duration, but it may be wrong. No feedback loop to refine the estimate. Fix: always ask when duration is absent (the system prompt instructs this, but Claude sometimes proceeds with an estimate anyway).

### Repeatedly shifting constraints
If the user says "move X to 2pm", then "move X to 3pm", then "move X back to 2pm", the scheduler runs three times. Each reschedule is globally consistent, but the user experience is choppy. Fix: batch constraint changes before rescheduling.

### Priority ties with no deadline
Two tasks with the same priority and no deadline get scheduled in arbitrary order (Python sort is stable, so insertion order breaks ties). Fix: expose a secondary sort key like "effort" or allow the user to manually rank within a priority level.

### No cross-day planning
The work window is a single day. If the user says "I also need to prep for Monday's meeting", the agent treats it as today's task. Fix: extend the session state to hold a date-keyed map of task lists.

### Context window exhaustion
After ~50 turns, the conversation history exceeds safe token limits. Currently no truncation logic exists. Fix: implement a rolling-window summarizer that compresses the first 20 turns into a structured summary whenever the total token count exceeds a threshold.

### Greedy scheduler suboptimality
EDF is optimal for "can I fit everything before its deadline?" but it does not optimize for user preferences like "I prefer deep work in the morning." Fix: add a soft-constraint scoring function that picks among equally-valid EDF orderings based on preferences.

---

## 9. What It Would Take to Fix the Major Gaps

| Gap | Fix | Effort |
|-----|-----|--------|
| Context exhaustion | Rolling-window summarizer with structured state | Medium (2–3 days) |
| Ambiguous durations | Clarification turn before tool call | Low (1 day — already half-done by system prompt) |
| Cross-day planning | Date-keyed session state, "plan week" mode | Medium (2–3 days) |
| User preference optimization | Soft-constraint scoring on schedule output | Medium-High |
| Production persistence | Redis session store + Postgres task store | Low-Medium (mostly infra) |
| Calendar API integration | OAuth flow + CalDAV/Graph API adapter | High (auth infra is the hard part) |
