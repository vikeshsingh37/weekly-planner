# Weekly Planner Agent — Architecture Decision Document

## 1. What Was Built

A conversational AI agent that helps users plan their workday through natural language. The agent collects tasks, schedules them deterministically, and updates the plan interactively across a session.

**Core capabilities:**
- Parse tasks (name, duration, priority, deadline) from freeform text
- Schedule tasks using Earliest-Deadline-First into available work-hour slots
- Automatically split long tasks into focus blocks (configurable max, default 90 min) with breaks between them so users are never scheduled for continuous multi-hour stretches
- Clamp scheduling to current time in the user's timezone — never plan tasks in the past
- Move individual tasks to user-specified times and reschedule around them
- Remove tasks and compact the schedule
- Fetch weather forecasts (Open-Meteo) to inform outdoor activity scheduling
- Persist session state (tasks, preferences, conversation history) across turns
- **Multi-user auth** — JWT-based login/register; each user's data is fully isolated in its own session directory, derived from their email address
- Web UI: login screen, then a two-panel layout with real-time chat on the left and a live day-view calendar + settings on the right
- REST endpoints for reading the schedule and saving preferences outside the chat loop
- CLI with animated thinking display

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

**Decision: plain Python.** The agentic loop is ~50 lines. Every tool is a pure `(inputs, session) → dict` function. There is no magic.

### Layered abstractions (api/ vs impl/)

The codebase is split into two layers:

- `api/` defines abstract interfaces (`AbstractScheduler`, `AbstractSessionManager`, `AbstractToolRunner`) and the Pydantic data models that flow across boundaries.
- `impl/` provides one concrete implementation of each: `EDFScheduler`, `JSONSessionManager`, `ToolRunner`, plus `auth` (user management) and `weather` (Open-Meteo integration).

The agent depends only on `api/`. Swapping JSON sessions for Redis, or EDF for an ILP-based scheduler, is a one-file change with no edits to `agent/agent.py`. This is also what makes the eval suite trivial — each case runs against a fresh in-memory `JSONSessionManager(session_file=None)` with no I/O.

### Configuration and dynamic system prompt

Model ID, max tokens, and the base system prompt live in `agent/config.py` / `agent/system_prompt.txt` — separated so the prompt can be iterated without touching the loop code.

The system prompt is **not static**: `DailyPlannerAgent._build_system_prompt()` appends the user's current date and time (resolved via `zoneinfo` from `Preferences.timezone`) on every `chat()` call. This gives the agent real-time awareness without requiring the user to state the time explicitly, and provides the ground truth that prevents scheduling tasks in the past.

---

## 3. Authentication

### Design

Authentication uses short-lived JWTs (7-day expiry) signed with a server-side `SECRET_KEY`. Passwords are hashed with bcrypt. User accounts are stored in `data/users.json`.

```
POST /auth/register  { email, password }  →  { token, user_id }
POST /auth/login     { email, password }  →  { token, user_id }
```

All other API endpoints require `Authorization: Bearer <token>`. The dependency `_require_user` extracts and verifies the token, returning the `user_id` that scopes every database/file operation.

### User ID derivation

The `user_id` is derived deterministically from the email address by replacing all non-alphanumeric characters (except `.` and `-`) with `_`:

```
alice@example.com   →  alice_example.com
bob+tag@corp.io     →  bob_tag_corp.io
```

This string is used as the session directory: `sessions/{user_id}/state.json`. Different users never share a directory, so isolation is enforced at the filesystem level with no additional access-control logic.

### WebSocket authentication

Browsers cannot set HTTP headers on WebSocket connections. The JWT is therefore passed as a query parameter: `/ws?token=<jwt>`. The server verifies the token before accepting the connection and closes with code `4401` if invalid. The client treats `4401` as a sign-out signal rather than a reconnect trigger.

### Why not sessions/cookies?

JWTs are stateless — no server-side session store required. The token payload carries the user's email; `verify_token()` re-reads `users.json` to confirm the user still exists. For a personal-scale app this is fine. At scale, move to a signed session cookie or a Redis-backed session store to support token revocation.

---

## 4. Tool Design

Seven tools, all deterministic. Each receives Pydantic-validated input and returns a Pydantic-typed output dict. Validation errors are caught in `ToolRunner.run` and returned as `{"error": "..."}` so the agent always gets a usable `tool_result` rather than a raised exception.

### `parse_and_add_tasks`
Accepts structured task data the AI agent extracts from natural language. The agent handles NLP (duration inference, priority estimation); this tool just writes to session state. Separating LLM extraction from state mutation means an extraction error doesn't corrupt the schedule — the tool either accepts valid data or returns an error.

### `schedule_tasks`
Runs the EDF scheduler over all non-pinned tasks. Always called *after* adding/removing tasks. Deterministic: same inputs → same schedule every time. Computes `now_min` (current minute of day in the user's timezone) before invoking the scheduler so past time slots are never used. Long tasks are automatically split into focus blocks here; the `Task.chunks` field is updated in place.

### `move_task`
Pins a task to a user-specified time slot, marks it `pinned=True`, then calls `schedule_tasks` to fill the remaining time around it. Pinned tasks are never moved or split by the auto-scheduler.

### `remove_task`
Removes a task by name (case-insensitive). Does not auto-reschedule — the agent calls `schedule_tasks` afterward if needed.

### `get_schedule`
Read-only snapshot of session state. The agent is instructed to call this before reporting to the user, which prevents it from reporting stale or invented slot times. Returns `chunks` per task so the agent can report all focus-block times accurately.

### `update_preferences`
Modifies any combination of: `work_start`, `work_end`, `break_minutes`, `max_chunk_minutes`, `timezone`, `planning_days`, `location_name`, `latitude`, `longitude`.

- `break_minutes`: gap inserted after every task *and* between consecutive focus blocks of the same long task.
- `max_chunk_minutes`: maximum continuous focus block in minutes. `0` = no limit. Default 90.
- `timezone`: IANA timezone (e.g. `America/New_York`, `Asia/Kolkata`). Validated via `zoneinfo.ZoneInfo`. Used by the scheduler's `now_min` computation and by `_build_system_prompt()`.
- `location_name`: if the user provides a city name without coordinates, `impl/weather.py` geocodes it via Open-Meteo's geocoding API before saving.

All preference changes are written to disk immediately. The WebSocket agent's in-memory session picks up changes on the next `chat()` call via `session.reload()`.

### `get_weather`
Fetches an hourly weather forecast from Open-Meteo (free, no API key) for the user's saved coordinates. Returns temperature range, precipitation, an outdoor conditions rating (good / moderate / poor), and the best outdoor window of the day. The agent calls this tool before scheduling outdoor tasks.

---

## 5. Scheduling Algorithm: Earliest Deadline First (EDF)

**Why EDF over other approaches:**

| Algorithm | Optimal? | Complexity | Preemptive? |
|-----------|----------|------------|-------------|
| Random | No | O(n) | No |
| Priority-only | No | O(n log n) | No |
| EDF (chosen) | Yes (for single resource) | O(n log n) | No |
| Integer Linear Programming | Yes (multi-constraint) | Exponential | N/A |

EDF is provably optimal for single-machine scheduling when all tasks must complete before their deadline and preemption isn't allowed — which matches the daily planning problem exactly.

**Implementation details (`impl/scheduler.py`):**

1. **Pinned tasks block time first** — user-explicit placement always wins; pinned tasks are never moved or split.
2. **Remaining tasks sorted** — deadline ascending, then priority descending for ties.
3. **Free slots computed** — complement of pinned-task intervals within the work window.
4. **`now_min` clamping** — free slots are advanced to `max(slot_start, now_min)` so past time is never offered to the scheduler. `now_min` is `None` when the plan date is not today, allowing the full window.
5. **Short tasks** (`duration <= max_chunk_minutes`, or `max_chunk_minutes == 0`): placed whole into the first slot that fits and meets the deadline. If no slot fits, the task is marked unschedulable.
6. **Long tasks** (`duration > max_chunk_minutes`): `_place_chunked()` greedily places blocks of at most `max_chunk_minutes` across consecutive free slots, advancing each slot cursor by `break_minutes` after each block. The task's `chunks` field records every `{start, end}` pair. `scheduled_start` / `scheduled_end` hold the first block's start and last block's end. If not all blocks can fit before the deadline, the task is marked unschedulable with a human-readable reason.
7. **Unschedulable tasks** always get a human-readable reason — never silently dropped.

**Conflict detection** (`check_conflicts`) is chunk-aware: for split tasks it inspects each individual block's interval rather than the coarse `scheduled_start–scheduled_end` span. This prevents falsely flagging a task scheduled during a focus-block break as a conflict.

**Break handling:** `break_minutes` serves double duty — gap between different tasks *and* gap between consecutive blocks of the same split task. This keeps the model simple and consistent.

**Time representation:** all times stored internally as 24-hour `HH:MM` strings (enforced by Pydantic validators). Converted to 12-hour AM/PM at display time.

---

## 6. Session Memory

```python
SessionState:
    tasks: List[Task]                  # source of truth for all tasks + schedule
    preferences: Preferences           # work hours, timezone, location, planning_days, etc.
    conversation_history: List[dict]   # full LLM message history
```

**`Task` fields relevant to scheduling:**

```python
Task:
    date:           Optional[str]    # YYYY-MM-DD; None = session base date
    scheduled_start: Optional[str]   # HH:MM of first (or only) block
    scheduled_end:   Optional[str]   # HH:MM of last (or only) block
    pinned: bool                     # True = user-placed, never auto-moved
    status: "pending" | "scheduled" | "unschedulable"
    chunks: List[dict]               # [{"start": "HH:MM", "end": "HH:MM"}, ...] — empty for single-block tasks
```

`Task.chunks` is persisted to the session JSON so the REST `/api/schedule` endpoint can serve chunk data to the calendar without re-running the scheduler.

**Why store the full conversation history?** The model's context window is the memory. No vector DB, no summarisation pipeline, no embedding. The session state stores the history in the format the API expects — multi-turn context is free. The cost: very long sessions eventually hit the token limit.

**Preference sync across WebSocket and REST:** The WebSocket handler holds a `JSONSessionManager` in memory for the duration of the connection. The REST `POST /api/preferences` endpoint creates a separate instance, writes to disk, and returns. On the agent's next `chat()` call, `session.reload()` re-reads state from disk, picking up the new preferences before any tool call runs.

**User isolation:** each user's session file lives in `sessions/{user_id}/state.json`. The `user_id` is derived from the user's email (see §3). The server always extracts `user_id` from the verified JWT — it is never accepted as a query parameter.

**Persistence:** serialised to JSON. For production, swap `JSONSessionManager` for a Redis or Postgres-backed implementation — `AbstractSessionManager` is the only interface that changes.

---

## 7. Web UI and Event Streaming

The web UI is built on FastAPI with a WebSocket endpoint (`/ws`) and REST endpoints. Because the agent loop is synchronous (blocking API calls), it runs in a thread-pool executor (`loop.run_in_executor`) so it doesn't block the async event loop.

Events flow from the agent thread to the WebSocket sender via `asyncio.Queue` and `loop.call_soon_threadsafe`:

```
agent thread               main event loop            browser
────────────               ───────────────            ───────
on_event("tool_start")  →  queue.put_nowait()
                        ←  queue.get()         →  ws.send_json()  →  handle(msg)
```

**WebSocket event protocol (server → client):**

| Event | Payload | When |
|-------|---------|------|
| `thinking_start` | — | Before each Claude API call |
| `thinking_end` | — | After each Claude API call |
| `tool_start` | `name`, `label` | Before tool executes |
| `tool_end` | `name`, `label` | After tool returns |
| `response` | `text` | Final agent reply |
| `error` | `text` | Unhandled exception |
| `done` | — | Always sent last |

**REST endpoints:**

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/auth/register` | None | Create account; returns JWT |
| `POST` | `/auth/login` | None | Sign in; returns JWT |
| `GET` | `/api/schedule` | JWT | Tasks + preferences — drives calendar rendering |
| `POST` | `/api/preferences` | JWT | Save work hours, timezone, location, etc. |
| `GET` | `/api/location` | None | IP-based location detection |

**UI layout:** The app is hidden behind a login/register screen until the user has a valid JWT in `localStorage`. After auth, a two-panel layout is shown: chat left, calendar sidebar right.

*Calendar:*
- Vertical day-view timeline spanning `work_start` → `work_end`
- Each scheduled task is a colour-coded block; colour encodes priority (1 = grey → 5 = red)
- Tasks with `chunks` render as multiple separate blocks ("Write design doc · 1/3") with break gaps visible between them
- Navigation arrows step through the planning horizon by `planning_days` days at a time
- A red dot + horizontal line marks the current time when viewing today's plan
- Fetched via `GET /api/schedule` on WebSocket connect and after every agent response

*Settings:*
- **Timezone** — `<select>` with 60+ IANA options grouped by region
- **Location** — city name with auto-detect (browser GPS or IP fallback) and geocoding
- **Work start / end** — `<input type="time">`
- **Break between tasks** — number input (0–60 min)
- **Planning horizon** — 1–7 days
- **Max focus block** — `<select>`: No limit / 25 min (Pomodoro) / 50 / 90 / 120 / 150 min
- Save button → `POST /api/preferences` → toast confirmation + calendar refresh

The CLI uses the same `on_event` callback with a `ThinkingDisplay` class that overwrites a single terminal line using `\r`, avoiding multi-line ANSI cursor tracking issues.

---

## 8. Production Considerations (100k+ Users)

### What breaks first

**Token budget per session.** At 100k users, long sessions (>30 turns) will start hitting limits or getting expensive. Fix: implement rolling window compression — summarise the first N turns into a compact state snapshot and drop the raw history.

**Cold starts.** Each new session re-sends the full system prompt + conversation history. At scale, prompt caching (`cache_control` in the Anthropic API) dramatically reduces cost for the stable system prompt portions.

**Scheduler correctness under concurrent edits.** `JSONSessionManager` is not thread-safe. If the same session is modified by two concurrent requests (unlikely in single-user scenario, but possible via the REST API), race conditions can corrupt state. Fix: optimistic locking on session writes, or a single-writer queue per session.

**User store concurrency.** `data/users.json` is read and written with no locking. Concurrent registrations could corrupt the file. Fix: replace with SQLite (with WAL mode) or Postgres.

**Focus block fragmentation.** At high task density, many small free slots cause a long task's blocks to be scattered across the day. The greedy algorithm handles this correctly but the resulting schedule may feel disjointed. Fix: add a "minimum block size" threshold — don't start a chunk unless the remaining slot is at least N minutes.

### What to monitor

| Metric | Alert threshold | Why |
|--------|----------------|-----|
| Tool call error rate | > 5% | Tool failure = broken schedule |
| Unschedulable task rate | > 20% | Users setting impossible plans |
| Session token count | > 150k | Approaching context limit |
| `schedule_tasks` latency | > 200ms | Scheduler regression |
| Chunk count per task | avg > 4 | Users over-loading single day |
| Auth failure rate | > 10% | Brute-force or misconfiguration |

### What to fix first at scale

1. **Prompt caching** — biggest cost reduction, zero behaviour change
2. **Session compression** — enables long-running sessions
3. **Persistent storage** — swap JSON files for a real DB (SQLite → Postgres)
4. **Token revocation** — add a `jti` claim and a blocklist for sign-out
5. **Structured logging** — per-turn tool call traces for debugging

---

## 9. What Was Deliberately Not Built

| Feature | Why Not |
|---------|---------|
| Calendar API (Google Calendar, Outlook) | Requires OAuth per user, adds infra complexity, not needed to demonstrate agent quality |
| Long-term memory across days | Out of scope — single-day planner |
| Vector memory / semantic search | Overkill for <50 tasks per session |
| Natural language date parsing ("next Monday") | Dates referenced as YYYY-MM-DD; agent interprets relative dates correctly in most cases |
| Constraint solver (ILP) | EDF is optimal for the single-machine case; ILP adds exponential complexity with no practical benefit here |
| DST-aware time arithmetic | All stored times are `HH:MM` wall-clock strings; DST transitions within a workday are an edge case that doesn't affect correctness for typical use |
| Longer break after N focus blocks | `break_minutes` is uniform; a "long break every 3 chunks" rule would require a more complex slot model |
| Per-task focus block override | All tasks share `Preferences.max_chunk_minutes`; per-task chunk sizes would add model complexity |
| Token revocation on sign-out | JWTs are stateless; sign-out is client-side only (token removed from `localStorage`). Add a server-side blocklist for production |
| Password reset / email verification | No email service integrated; add for production use |

---

## 10. Where the Agent Breaks

### Ambiguous task duration
*"I need to prepare for the meeting"* — the agent estimates a duration. No feedback loop to refine it. Fix: system prompt instructs asking when duration is absent; agent sometimes proceeds with an estimate anyway.

### Focus blocks scattered by pinned tasks
If the user pins several tasks, the remaining free slots may be small and non-contiguous. A long task's focus blocks get placed in whatever fragments are available — which may mean 3 blocks spread across the whole day rather than a coherent morning session. Fix: add a "minimum viable block" threshold; if the only available slot is < 20 min don't start a chunk there.

### Greedy chunk placement ignores other pending tasks
`_place_chunked()` consumes free slots greedily. If a high-priority long task is processed first (as EDF dictates), it may consume slots that a later short task with an earlier deadline could have used. In practice EDF sorts by deadline first so this is rare, but it can occur when tasks share a deadline. Fix: consider partial placement and backtracking.

### Repeatedly shifting constraints
"Move X to 2 PM" → "move X to 3 PM" → "move X back to 2 PM" runs the full scheduler three times. Each result is globally consistent, but the experience is choppy. Fix: batch constraint changes before rescheduling.

### Priority ties with no deadline
Two tasks with the same priority and no deadline are scheduled in insertion order (Python sort is stable). Fix: expose a secondary sort key like "effort" or allow manual rank within a priority level.

### Context window exhaustion
After ~50 turns, conversation history approaches token limits. No truncation logic exists. Fix: rolling-window summariser that compresses the first 20 turns into a structured summary whenever total token count exceeds a threshold.

---

## 11. What It Would Take to Fix the Major Gaps

| Gap | Fix | Effort |
|-----|-----|--------|
| Context exhaustion | Rolling-window summariser with structured state snapshot | Medium (2–3 days) |
| Focus block fragmentation | Minimum viable block threshold + skip-ahead logic | Low (half a day) |
| Greedy chunk suboptimality | Backtracking or two-pass placement | Medium |
| Ambiguous durations | Clarification turn before tool call | Low (half-done by system prompt) |
| User store concurrency | SQLite WAL or Postgres | Low (mostly infra) |
| Token revocation | JWT blocklist in Redis | Low |
| Production persistence | Redis session store + Postgres task store | Low-Medium (mostly infra) |
| Calendar API integration | OAuth flow + CalDAV/Graph API adapter | High (auth infra is the hard part) |
| Password reset | SMTP integration + reset-token store | Medium |