# Weekly Planner

AI agent that schedules your week from natural language. Tell it what you need to get done — it fits tasks into your work hours, splits long ones into focus blocks, handles deadlines, and adjusts when plans change. Each user has their own isolated calendar.

---

## Get started

**1. Install**
```bash
uv sync
```

**2. Start Langfuse (observability)**
```bash
docker compose up -d
```
This starts Langfuse + Postgres at **http://localhost:3000**. Open it, create a local account, create a project, and copy the two keys — you'll need them in the next step.

**3. Configure**
```bash
cp .env.example .env
```
Edit `.env` and fill in all four values:
```env
ANTHROPIC_API_KEY=sk-ant-...        # from console.anthropic.com

SECRET_KEY=<random string>          # signs auth tokens — generate one:
                                    # python -c "import secrets; print(secrets.token_hex(32))"

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
You: I have a design doc (3h, deadline 2 PM), PR reviews (1h), and a standup (30 min) today.

Agent: Added 3 tasks. Here's today's schedule:

  Write design doc — 2 focus blocks (deadline 2 PM):
    • 9:00 – 10:30 AM  Block 1/2
    • 10:35 – 12:05 PM  Block 2/2

  • 12:10 – 1:10 PM  PR reviews
  • 1:15 – 1:45 PM   Standup

You: Move standup to 10 AM. And add a 2h deep work block for tomorrow.

Agent: Done — standup pinned at 10 AM, schedule adjusted around it.
       Added deep work (2h) for tomorrow at 9:00–11:00 AM.
```

Things you can ask:
- Add tasks for any day this week — *"add a 90-min planning session for Wednesday"*
- Move or reschedule tasks — *"move the design doc to tomorrow afternoon"*
- Change your work hours or timezone — *"my day starts at 8 AM"*
- Plan around weather — *"schedule a 30-min run today"* (fetches forecast automatically)
- See the full week — *"show me my schedule for the next 5 days"*

---

## Architecture

```
server.py          FastAPI server — auth endpoints, REST API, WebSocket
cli.py             Interactive CLI (same agent, no browser needed)
static/index.html  Single-file web UI — login screen + chat + calendar
api/               Abstract interfaces (Scheduler, SessionManager, ToolRunner) + Pydantic models
impl/
  auth.py          JWT auth, bcrypt passwords, email → user-ID mapping
  memory.py        JSON session files (one per user: sessions/{user_id}/state.json)
  scheduler.py     Earliest-Deadline-First scheduler with focus-block splitting
  tools.py         Tool implementations (add, schedule, move, remove, prefs, weather)
  weather.py       Open-Meteo forecast fetcher
agent/
  agent.py         Agentic loop — calls Claude, executes tools, streams events
  system_prompt.txt  Agent instructions (edit freely)
data/users.json    User registry — auto-created on first register
```

**Key choices:**
- Plain Python agentic loop (~50 lines) — no LangGraph, easy to trace and debug
- `api/` vs `impl/` split — swap the JSON store for Redis or the EDF scheduler for ILP without touching the agent
- Deterministic scheduling — the LLM picks which tool to call; Python decides the time slots
- EDF (Earliest-Deadline-First) — optimal for single-machine scheduling with deadlines
- User isolation at the filesystem level — each email maps to its own session directory

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
- Docker (for Langfuse observability)
- Anthropic API key (`claude-sonnet-4-6`)