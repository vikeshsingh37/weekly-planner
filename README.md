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

OPENAI_API_KEY=sk-...               # LLM-as-judge for evals (model configured in evals/config.py)
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
run_local_evals.py      Offline eval runner — no Langfuse / OpenAI needed, CI-friendly
run_langfuse_eval.py    Full eval runner — scores all metrics and pushes results to Langfuse
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
  config.py             Agent config — model, token limits, temperature, thinking settings
  system_prompt.txt     Agent instructions (edit freely)
evals/
  config.py             Eval config — judge model, pass thresholds, dataset name, shared paths
  eval_data.py          EvalCase definitions used by the local runner
  eval_runner.py        Local runner logic — isolated sessions, check assertions
  llm_judge.py          LLM-as-judge prompts for faithfulness / helpfulness / failure_explanation
  push_to_langfuse.py   Upserts the dataset to Langfuse (run once, or after dataset changes)
eval_data/
  dataset.py            25 EvalDatapoint definitions with tool-call and session-state expectations
  schemas.py            EvalDatapoint, ExpectedToolCall, AnswerCheck dataclasses
  metrics.py            TSA / TPA / SSA / AKR / AF / GFR scoring logic (deterministic)
  visualize.py          CLI table for comparing eval runs
tests/
  test_scheduler.py     EDF scheduler — helpers, scheduling, deadlines, pinned tasks, chunking, conflicts
  test_metrics.py       All 6 metric functions (TSA, TPA, FTA, AKR, AF, SSA, GFR)
  test_schemas.py       ParamCheck operators and dot-path traversal
  test_models.py        Task and Preferences Pydantic validators
  test_memory.py        JSONSessionManager add / find / remove / preferences / history
  test_eval_config.py   evals/config.py sanity checks
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

## Tests

124 unit tests covering the scheduler, all metric functions, Pydantic model validators, session manager, and eval config. No API keys or external services required.

```bash
uv run pytest           # run all tests
uv run pytest -v        # verbose output
uv run pytest tests/test_scheduler.py   # single file
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

25 hand-crafted cases across 5 categories. One runner — `run_langfuse_eval.py` — scores all metrics and pushes results to Langfuse.

```bash
# Push dataset to Langfuse once (or after dataset changes)
uv run python evals/push_to_langfuse.py

# Run evals — one LLM call per case, all metrics scored and pushed to Langfuse
uv run python run_langfuse_eval.py
uv run python run_langfuse_eval.py --run-name sprint-12
uv run python run_langfuse_eval.py --dataset weekly-planner-v2

# Quick local sanity check (no Langfuse / OpenAI needed)
uv run python run_local_evals.py
```

| Category | Cases | What it tests |
|----------|-------|---------------|
| `tool_selection` | 7 | Correct tool chosen for each intent |
| `tool_params` | 6 | Correct arguments extracted from natural language |
| `multi_turn` | 5 | State coherence across 2–5 turns |
| `final_answer` | 4 | Response text faithfully reflects tool outputs |
| `edge_case` | 3 | Graceful handling of impossible / ambiguous inputs |

**Metrics scored per case:**

| Type | Metric | What it measures |
|------|--------|-----------------|
| Deterministic | TSA | Tool called on the correct turn |
| Deterministic | TPA | Correct arguments extracted |
| Deterministic | SSA | Session state correct after all turns |
| LLM-as-judge | faithfulness | Response matches actual tool outputs |
| LLM-as-judge | helpfulness | Response is clear and actionable |
| LLM-as-judge | failure_explanation | For edge cases: explains *why*, not just "sorry" |

The LLM-as-judge model and pass thresholds are configured in `evals/config.py`.

See [`eval_data/README.md`](eval_data/README.md) for full dataset documentation.

---

## Requirements

- Python 3.14+
- Docker (for Langfuse observability + Postgres session storage)
- Anthropic API key (`claude-sonnet-4-6`)