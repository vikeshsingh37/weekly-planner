# Weekly Planner

AI agent that schedules your week from natural language. Tell it what you need to get done — it fits tasks into your work hours, splits long ones into focus blocks, handles deadlines, and adjusts when plans change. Each user has their own isolated calendar.

---

## Get started

**1. Install**
```bash
uv sync
```

**2. Start Langfuse + Postgres (observability + session storage)**
```bash
docker compose up -d
```
This starts Langfuse at **http://localhost:3000** and Postgres on port 5432. Open Langfuse, create a local account, create a project, and copy the two keys — you'll need them in the next step.

**3. Configure**
```bash
cp .env.example .env
```
Edit `.env` and fill in the values:
```env
ANTHROPIC_API_KEY=sk-ant-...        # from console.anthropic.com

SECRET_KEY=<random string>          # signs auth tokens — generate one:
                                    # python -c "import secrets; print(secrets.token_hex(32))"

DATABASE_URL=postgresql://langfuse:langfuse@localhost:5432/langfuse  # session storage

LANGFUSE_PUBLIC_KEY=pk-lf-...       # from http://localhost:3000
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

**4. Start the server**
```bash
uv run python server.py
```

**5. Open the app**

Go to **http://localhost:8000**, register with your email and password, then start typing your tasks. Every conversation turn will appear as a trace in Langfuse at http://localhost:3000.

---

## What you can do

```
You: I have a design doc (3h, deadline 2 PM), PR reviews (1h), and standup at 9:30 AM today.

Agent: Added 3 tasks. Here's today's schedule:

  • 9:30 – 10:00 AM  Standup (pinned)

  Write design doc — 2 focus blocks (deadline 2 PM):
    • 10:05 – 11:35 AM  Block 1/2
    • 11:40 AM – 1:10 PM  Block 2/2

  • 1:15 – 2:15 PM  PR reviews

You: Move standup to 10 AM. And add a 2h deep work block for tomorrow.

Agent: Done — standup pinned at 10 AM, schedule adjusted around it.
       Added deep work (2h) for tomorrow at 9:00–11:00 AM.
```

Things you can ask:
- Add tasks for any day this week — *"add a 90-min planning session for Wednesday"*
- Pin tasks to exact times — *"schedule standup on Tuesday and Thursday at 9:30 AM"*
- Move or reschedule tasks — *"move the design doc to tomorrow afternoon"*
- Remove tasks — agent asks for confirmation before deleting
- Change your work hours or timezone — *"my day starts at 8 AM"*
- Plan around weather — *"schedule a 30-min run today"* (fetches forecast automatically)
- See the full week — *"show me my schedule for the next 5 days"*

---

## Architecture

```
server.py               FastAPI server — auth endpoints, REST API, WebSocket
cli.py                  Interactive CLI (same agent, no browser needed)
static/index.html       Single-file web UI — login screen + chat + 24-hour calendar + debug trace
api/                    Abstract interfaces (Scheduler, SessionManager, ToolRunner) + Pydantic models
impl/
  auth.py               JWT auth, bcrypt passwords, email → user-ID mapping
  memory.py             _BaseSessionManager (in-memory + lock) + JSONSessionManager (file fallback)
  postgres_memory.py    PostgresSessionManager — JSONB session storage in Postgres
  scheduler.py          Earliest-Deadline-First scheduler with focus-block splitting
  tools.py              Tool implementations (add, schedule, move, remove, prefs, weather)
  weather.py            Open-Meteo forecast fetcher
agent/
  agent.py              Agentic loop — calls Claude, executes tools in parallel, streams events
  system_prompt.txt     Agent instructions (edit freely)
data/users.json         User registry — auto-created on first register
```

**Key choices:**
- Plain Python agentic loop (~50 lines) — no LangGraph, easy to trace and debug
- `api/` vs `impl/` split — swap Postgres for Redis or EDF for ILP without touching the agent
- Deterministic scheduling — the LLM picks which tool to call; Python decides the time slots
- EDF (Earliest-Deadline-First) — optimal for single-machine scheduling with deadlines
- Parallel tool execution — multiple tool calls per LLM turn run concurrently via `ThreadPoolExecutor`; session-level locking ensures state safety
- Postgres session storage — same Postgres instance as Langfuse; falls back to JSON files if `DATABASE_URL` is unset
- Debug trace button — after each agent reply, a collapsible panel shows every tool step with its execution time (ms)

See [`docs/architecture.md`](docs/architecture.md) for full design rationale.

---

## CLI

```bash
uv run python cli.py                 # uses system username as session key
uv run python cli.py --user alice    # explicit user
uv run python cli.py --verbose       # show tool calls
```

---

## Observability (Langfuse)

Every Claude API call and tool execution is traced as a Langfuse span — the local Docker stack is set up in step 2 of Get started above.

**Using Langfuse Cloud instead of Docker:** swap the host and use your cloud project keys:
```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Evals

```bash
uv run python run_evals.py                           # all 15 cases
uv run python run_evals.py --category task_completion
uv run python run_evals.py --verbose                 # show tool calls + responses
uv run python run_evals.py --output results/out.json
```

| Category | Cases | Tests |
|----------|-------|-------|
| `task_completion` | 4 | All tasks scheduled, correct partial scheduling |
| `hallucination` | 3 | Agent never invents time slots or tasks |
| `graceful_failure` | 4 | Impossible schedules, conflicts, empty sessions |
| `memory` | 4 | Tasks, moves, and removals persist across turns |

---

## Requirements

- Python 3.14+
- Docker (for Langfuse observability + Postgres session storage)
- Anthropic API key (`claude-sonnet-4-6`)
