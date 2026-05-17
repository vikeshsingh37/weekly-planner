# Weekly Planner — Eval Dataset

25 hand-crafted evaluation datapoints covering the agent's 7 tools across
single-turn and multi-turn scenarios, common use cases, and edge cases.

## File layout

```
eval_data/
├── schemas.py      EvalDatapoint + sub-types (ParamCheck, AnswerCheck, …)
├── dataset.py      25 EvalDatapoint instances (ALL_DATAPOINTS list)
├── metrics.py      Metric functions + METRIC_REGISTRY + descriptions
├── visualize.py    Terminal + HTML report renderer for saved results
└── README.md       This file

eval_results/           auto-created on first run
└── YYYYMMDD_HHMMSS.json  one timestamped file per run_evals.py invocation
```

## Dataset overview

| Category | Count | What it tests |
|---|---|---|
| `tool_selection` | 9 | Correct tool chosen for each intent — including multi-turn sequences where the tool choice is the key signal (e.g. `move_task` vs `schedule_tasks`, remove then reschedule) |
| `tool_params` | 7 | Correct arguments extracted from natural language (durations, priorities, deadlines, timezones, cross-turn preference propagation) |
| `final_answer` | 5 | Response text is faithful (12h format, names unscheduled tasks, explains deadline conflicts, end-to-end coherence) |
| `edge_case` | 4 | Graceful handling of impossible inputs (non-existent task, impossible deadline, task splitting, error without session corruption) |

**Turn depth**: 12 cases are single-turn; 13 cases have 2–5 turns (max 5). Category and conversation length are independent — use `EvalDatapoint.is_multi_turn` to slice by length across all categories.

> The former `multi_turn` category has been dissolved. Those 5 cases now live in the category that best describes what they test (`tool_selection` × 2, `tool_params` × 1, `edge_case` × 1, `final_answer` × 1).

## Evaluation dimensions

All six metrics are scored for every datapoint — there is no per-case `metrics` list.

### A. Tool-level

| Metric | Key | Target |
|---|---|---|
| Tool Selection Accuracy | `TSA` | ≥ 0.95 |
| Tool Parameter Accuracy | `TPA` | ≥ 0.90 |
| First-Turn Tool Accuracy | `FTA` | 1.0 |

**TSA** — Fraction of required tool calls made on the correct turn.  
**TPA** — Fraction of `ParamCheck` assertions that pass against actual tool inputs.  
**FTA** — Binary: was the first tool called the expected one? (high-signal intent proxy)

#### Capturing tool calls

No agent instrumentation needed — tool calls are already in Langfuse traces.
`WeeklyPlannerAgent._run_tool()` is `@observe`-decorated and explicitly calls
`langfuse_context.update_current_observation(input=inputs)` and
`update_current_observation(output=result)`, so every tool span shows the exact
parameters passed (e.g. the full tasks array for `parse_and_add_tasks`) and the
tool result. Tools with no input params (e.g. `schedule_tasks`) correctly show
`{}` as input and the scheduled slots in the output.

After a run, fetch the `eval_user` traces from the Langfuse API and reconstruct
the tool call log:

```python
from langfuse import Langfuse
from eval_data.metrics import tool_selection_accuracy, tool_param_accuracy

lf = Langfuse()
traces = lf.get_traces(user_id="eval_user", limit=50).data  # adjust limit to # cases

for trace in traces:
    tool_call_log = [
        {
            "tool": obs.name,               # span name = tool name
            "turn_index": obs.metadata.get("turn_index", 0),
            "params": obs.input or {},      # span input = tool params dict
        }
        for obs in lf.get_observations(trace_id=trace.id).data
        if obs.type == "SPAN" and obs.name in TOOL_NAMES
    ]
    tsa = tool_selection_accuracy(expected_tool_calls, tool_call_log)
    tpa = tool_param_accuracy(expected_tool_calls, tool_call_log)
```

`TOOL_NAMES` is the set of the 7 tool names defined in `api/tools.py`.
If `turn_index` is not yet in span metadata, match by trace timestamp ordering
instead — the spans within a trace are ordered chronologically.

### B. State-level

| Metric | Key | Target |
|---|---|---|
| Session State Accuracy | `SSA` | ≥ 0.90 |

**SSA** — Fraction of `SessionCheck` predicates that pass after all turns complete.
This is the ground-truth check: did the right things actually persist?

### C. LLM-as-judge

These three metrics run in `run_evals.py` (project root) via the OpenAI API (`evals/llm_judge.py`).
They catch semantic issues that keyword matching cannot reliably detect.

The judge model is set in `evals/config.py` (`JUDGE_MODEL`). Change it there to switch models without touching any other file.

| Metric | Target | What it checks |
|---|---|---|
| `faithfulness` | ≥ 0.8 | Response matches actual tool outputs — correct times, names, durations |
| `helpfulness` | ≥ 0.8 | Response is clear, complete, and actionable for the user |
| `failure_explanation` | ≥ 0.8 | For `edge_case` only: agent explains *why* (e.g. "only 30 min before deadline") not just "sorry" |

Each judge call returns a float 0–1 and a one-sentence reason, which is stored as the Langfuse score `comment`.

---

## Evaluation platform — in-repo vs. Langfuse

**Short answer: use both. They serve different purposes.**

### `run_evals.py` (project root) — the single eval runner

Runs all 6 metrics in one pass: TSA, TPA, SSA (deterministic) + faithfulness, helpfulness, failure_explanation (GPT-4.5 LLM-as-judge). Pushes all scores to Langfuse with reasons attached as score comments. Also reports a cross-cut breakdown by conversation length (single-turn vs multi-turn) independent of category.

Before any turns run, each session receives two layers of preferences:

1. **`EVAL_DEFAULT_PREFERENCES`** (from `evals/config.py`) — applied to every case:
   - `current_time = "08:00"` — simulated time so "1 PM" tasks are always in the future
   - `work_start = "00:00"`, `work_end = "23:59"` — full-day window; no task fails due to work-window boundaries unless the case explicitly narrows them
   - `timezone = "Asia/Kolkata"`, `location_name = "Bengaluru, Karnataka, India"` with pre-set coordinates — weather tests never hit "location not set"

2. **Per-case `preferences`** and **`current_time`** — applied on top, overriding the defaults. Cases like FA_02 (narrow work window to force unschedulable tasks) and FA_03 still work correctly because their explicit `preferences` override the 24-hour default.

The judge also receives the effective `current_time` and work window so it can verify scheduling math independently of when the suite runs.

Requires: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `OPENAI_API_KEY` in `.env`.

### Langfuse — recommended for production quality monitoring

You already have Langfuse integrated (`agent.py` uses `@observe` decorators and
`_run_tool` is already traced as a child span). Langfuse adds:

1. **Dataset API** — push these 25 cases as `DatasetItem` objects, run the agent
   against them as a `DatasetRun`, and score each item. Scores appear in the
   Langfuse dashboard alongside p50/p95 latency, cost, and model metadata.

2. **LLM-as-judge** — Langfuse supports "online evaluations" that score a
   configurable % of live traces using a prompt template + model of your choice.
   No code changes needed once configured in the Langfuse UI.

3. **Trace drill-down** — for failed cases, inspect the full LLM message log,
   tool spans, and input/output at each step.

4. **A/B model comparison** — run the same dataset against two model versions;
   compare scores in the dashboard.

#### Pushing this dataset to Langfuse

```python
import langfuse
from eval_data.dataset import ALL_DATAPOINTS

lf = langfuse.Langfuse()
dataset = lf.create_dataset(name="weekly-planner-v1")

for dp in ALL_DATAPOINTS:
    lf.create_dataset_item(
        dataset_name="weekly-planner-v1",
        input={"turns": dp.turns, "preferences": dp.preferences},
        expected_output={
            "expected_tools": [e.tool for e in dp.expected_tool_calls],
            "answer_contains_any": [
                ac.contains_any for ac in dp.answer_checks
            ],
        },
        metadata={"id": dp.id, "category": dp.category, "notes": dp.notes,
                  "num_turns": dp.num_turns, "is_multi_turn": dp.is_multi_turn},
    )
```

---

## Real-time evaluation recommendations

These run on live production traffic rather than a static dataset.

### 1. Inline deterministic checks (zero latency cost)

After every `schedule_tasks` call, check for conflicts and log the result as a
Langfuse score. This is cheap (~1ms) and catches scheduler bugs immediately:

```python
# in impl/tools.py — ToolRunner.schedule_tasks(), after scheduling:
from langfuse import Langfuse
conflicts = scheduler.check_conflicts(session.state.tasks)
lf = Langfuse()
lf.score(
    trace_id=current_trace_id(),
    name="no_scheduling_conflicts",
    value=1.0 if not conflicts else 0.0,
)
```

### 2. LLM-as-judge on sampled live traces (Langfuse online eval)

Configure in the Langfuse UI — no code changes:
- **Evaluator model**: any OpenAI-compatible model (see `evals/config.py` `JUDGE_MODEL`)
- **Sample rate**: 10%–20% of traces
- **Prompt template**:

```
You are evaluating a scheduling assistant response.

Tool outputs: {tool_outputs}
Agent response: {output}

Score 0–1 on faithfulness (response matches tool output) and helpfulness (clear, actionable).
Return JSON: {"faithfulness": float, "helpfulness": float}
```

For offline evals, `evals/llm_judge.py` implements the same three judges (faithfulness, helpfulness, failure_explanation) and is called automatically by `run_evals.py` (project root). The model is read from `evals/config.py`.

### 3. User feedback signal

Add a 👍 / 👎 button to the chat UI. On click, POST to `/api/feedback` and
record the score on the Langfuse trace:

```python
# server.py — new endpoint
@app.post("/api/feedback")
async def feedback(trace_id: str, score: int):  # score: 1 or -1
    langfuse.score(trace_id=trace_id, name="user_feedback", value=(score + 1) / 2)
```

### 4. Tool error rate monitoring (Langfuse dashboard)

The existing `_run_tool` span already propagates exceptions to Langfuse.
In the Langfuse dashboard, create a metric: `error_rate` grouped by `span.name`
(= tool name). Alert if any tool's error rate exceeds 5% over a 1h window.

### 5. Latency alerting

Langfuse tracks p50/p95 LLM latency per trace. Set an alert:
- p95 > 10s → agent is doing extra tool calls or the model is cold
- Median turns per session > 3 for simple scheduling tasks → agent is asking
  too many clarifying questions

### 6. Cost monitoring

Langfuse tracks input/output tokens per trace. A runaway agentic loop (>10
iterations) will spike cost. Alert if `input_tokens + output_tokens` per trace
exceeds 8,000 tokens.

---

## Eval workflow — one run, not two

Running predictions twice (once locally, once for Langfuse) wastes LLM API
calls. The scripts are designed so a **single agent invocation per case** feeds
both the terminal report and the Langfuse dataset run.

```
evals/push_to_langfuse.py    ← run once to upload the 25 dataset items
        │
        ▼
run_evals.py         ← run to evaluate (one LLM call per case, at project root)
        │
        ├── agent.chat() calls attach as child spans via item.observe()
        │   so all turns of a case share ONE root trace, linked to the item
        │
        ├── Deterministic scores (TSA, TPA, SSA) pushed to the trace
        ├── LLM-as-judge scores (faithfulness, helpfulness, failure_explanation)
        │   pushed with one-sentence reason as score comment
        │
        └── JSON saved to eval_results/ → visualize.py for terminal / HTML
```

### How item.observe() links traces

`item.observe(run_name=...)` calls `langfuse_context._set_root_trace_id()` before
yielding. This makes the agent's existing `@observe`-decorated `chat()` method
attach as a child span of that root trace instead of starting a new one. On
context exit, the trace is automatically linked to the Langfuse dataset item run.
No changes to the agent are needed.

### What you see in Langfuse after a run

- **Datasets → weekly-planner-v1 → Runs** — one run per `run_evals.py` invocation
- **Per-item**: TSA, TPA, SSA, faithfulness, helpfulness, failure_explanation scores; full multi-turn trace with tool call spans
- **Compare runs**: select two runs to diff scores side-by-side (useful after a model or prompt change)

---

## Quick start

```bash
# Install deps (uv manages the venv; project is installed in editable mode automatically)
uv sync

# Step 1 — push dataset items to Langfuse (only needed once, or after dataset changes)
uv run python evals/push_to_langfuse.py

# Step 2 — run evals (one agent call per case, all 6 metrics scored, traces linked to dataset items)
uv run python run_evals.py
uv run python run_evals.py --run-name sprint-12
uv run python run_evals.py --dataset weekly-planner-v2

# Visualize the latest saved result (terminal)
uv run python eval_data/visualize.py

# Visualize a specific file and also export an HTML report
uv run python eval_data/visualize.py eval_results/20260516_143022.json --html

# Compare two runs side-by-side (e.g. before vs. after a model change)
uv run python eval_data/visualize.py --compare eval_results/before.json eval_results/after.json

# Print all metric descriptions
uv run python -c "
from eval_data.metrics import METRIC_DESCRIPTIONS
for k, v in METRIC_DESCRIPTIONS.items():
    print(f'\n{k}\n{v}')
"
```

### Session isolation

Every `run_evals.py` invocation:
1. Deletes `sessions/eval_user/` (removes any file-backed state from previous runs)
2. Creates each case in a fresh in-memory `JSONSessionManager` (no cross-case contamination)
3. Tags all Langfuse traces with `user_id="eval_user"` and `session_id="<run_name>/<case_id>"` so eval
   traffic is visually separated from real users in the Langfuse dashboard and each case has its own trace group

### Result files

Results accumulate in `eval_results/` — one JSON file per run. The visualizer always
defaults to the latest file when no path is given. Add `eval_results/` to `.gitignore`
if you don't want to commit run artifacts, or commit selected files as regression baselines.