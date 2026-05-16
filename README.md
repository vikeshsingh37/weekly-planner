# Daily Planner Agent

A conversational AI agent that helps you plan your workday. Tell it what tasks you have, and it will schedule them. Ask it to move things around. It remembers everything you said.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Run the agent
python main.py
```

To keep a session across restarts:
```bash
python main.py --session my_day
```

To see every tool call the agent makes:
```bash
python main.py --verbose
```

---

## Example Conversation

```
You: I need to study ML for 2 hours, gym for 1 hour, and finish a report by 6 PM.

Agent: I've added your 3 tasks and scheduled them:
  • 09:00 – 11:00  Study ML (2h, priority 3)
  • 11:05 – 12:05  Gym (1h, priority 3)
  • 12:10 – 13:10  Finish report (1h, priority 3, deadline 18:00)

You have 4h50m of free time remaining today.

You: Move gym to evening, like 17:00.

Agent: Done. Gym is now pinned at 17:00 – 18:00. I've rescheduled the rest:
  • 09:00 – 11:00  Study ML
  • 11:05 – 12:05  Finish report
  • 17:00 – 18:00  Gym (pinned)
```

---

## Architecture

```
main.py                     ← CLI entry point
agent/
  agent.py                  ← Conversational loop (Claude API)
  tools.py                  ← 5 deterministic tool functions
  scheduler.py              ← EDF scheduling algorithm
  memory.py                 ← Session state + JSON persistence
evals/
  test_cases.py             ← 15 eval scenarios across 4 categories
  eval_runner.py            ← Runs cases, reports pass/fail
run_evals.py                ← Eval CLI entry point
docs/
  architecture.md           ← Design decisions, trade-offs, failure modes
```

**Key design choices:**
- Plain Python + Claude API — no LangGraph overhead, easier to debug
- All scheduling logic is deterministic Python — the LLM only decides which tool to call
- EDF (Earliest-Deadline-First) scheduler — provably optimal for single-day planning
- Full conversation history stored in session state — multi-turn memory is free

See [`docs/architecture.md`](docs/architecture.md) for the full design rationale.

---

## Running Evals

```bash
# Run all 15 eval cases
python run_evals.py

# Single category
python run_evals.py --category task_completion
python run_evals.py --category hallucination
python run_evals.py --category graceful_failure
python run_evals.py --category memory

# Save JSON report
python run_evals.py --output results/report.json

# Verbose (show tool calls + responses)
python run_evals.py --verbose
```

### Eval categories

| Category | Cases | What it measures |
|----------|-------|-----------------|
| `task_completion` | 4 | Did all requested tasks get scheduled? Correct partial scheduling? |
| `hallucination` | 3 | Does the agent invent time slots or tasks not in state? |
| `graceful_failure` | 4 | Impossible schedules, deadline conflicts, empty sessions |
| `memory` | 4 | Do tasks, moves, and removals persist across turns? |

Each case is a series of user turns followed by assertions against the final session state and agent responses. No manual review needed — all checks are automated.

---

## Tools

| Tool | When called | Logic |
|------|-------------|-------|
| `parse_and_add_tasks` | User mentions new tasks | Claude extracts structure; Python writes to state |
| `schedule_tasks` | After adding/removing tasks | EDF algorithm assigns time slots |
| `move_task` | User specifies a time for a task | Pins task, reschedules others around it |
| `remove_task` | User removes a task | Deletes from state |
| `get_schedule` | Before reporting to user | Returns current state snapshot |
| `update_preferences` | User changes work hours | Updates work_start/end/break_minutes |

---

## Known Limitations

- **Context window:** sessions > ~50 turns will approach token limits. No truncation logic yet.
- **Single day only:** no cross-day planning.
- **Greedy scheduler:** EDF is optimal for fitting tasks before deadlines but doesn't optimize for user preferences (e.g., "I prefer deep work in the morning").
- **Ambiguous durations:** if the user doesn't provide a duration, Claude estimates — which can be wrong.
- **No calendar integration:** schedules exist only in the agent's session state.

See [`docs/architecture.md`](docs/architecture.md) for detailed analysis and proposed fixes.

---

## Requirements

- Python 3.11+
- `anthropic>=0.40.0`
- `python-dotenv>=1.0.0`
- An Anthropic API key (claude-sonnet-4-6)
