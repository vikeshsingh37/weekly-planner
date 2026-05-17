#!/usr/bin/env python3
"""
Primary eval runner — scores all metrics in a single pass against Langfuse.

For each dataset item the script:
  1. Fetches the item from Langfuse and matches it to its local EvalDatapoint.
  2. Runs all conversation turns under item.observe() so agent spans attach to
     the same root trace (no duplicate LLM calls).
  3. Scores all eight metrics and pushes them to the trace.
  4. Saves a local JSON report to eval_results/.

Metrics
───────
  Deterministic (objective facts):
    TSA — Tool Selection Accuracy   (tool called on the right turn?)
    TPA — Tool Parameter Accuracy   (right args extracted?)
    SSA — Session State Accuracy    (session state correct after all turns?)

  LLM-as-judge / GPT-4.5 (semantic understanding):
    faithfulness          — response matches actual tool outputs, not hallucinated
    helpfulness           — response is clear, complete, and actionable
    failure_explanation   — for edge/failure cases: agent explains *why*, not just "sorry"

Usage:
    uv run python run_langfuse_eval.py
    uv run python run_langfuse_eval.py --dataset weekly-planner-v2
    uv run python run_langfuse_eval.py --run-name sprint-12

Requires in .env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST, OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pathlib
import shutil
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from evals.llm_judge import judge_faithfulness, judge_helpfulness, judge_failure_explanation  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATASET_NAME  = "weekly-planner-v1"
RESULTS_DIR   = "eval_results"
EVAL_USER_ID  = "eval_user"
_SESSION_DIR  = os.path.join("sessions", EVAL_USER_ID)


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _score_ssa(dp, session) -> float:
    if not dp.session_checks:
        return 1.0
    passed = sum(1 for c in dp.session_checks if c.fn(session))
    return round(passed / len(dp.session_checks), 4)


def _score_tsa(dp, tool_call_log: list[dict]) -> float:
    from eval_data.metrics import tool_selection_accuracy
    return round(tool_selection_accuracy(dp.expected_tool_calls, tool_call_log).score, 4)


def _score_tpa(dp, tool_call_log: list[dict]) -> float:
    from eval_data.metrics import tool_param_accuracy
    return round(tool_param_accuracy(dp.expected_tool_calls, tool_call_log).score, 4)


# ── Per-item runner ───────────────────────────────────────────────────────────

def run_item(item, dp, lf, run_name: str) -> dict:
    from impl.memory import JSONSessionManager
    from impl.tools import ToolRunner
    from agent.agent import WeeklyPlannerAgent

    session = JSONSessionManager(session_file=None)
    if dp.preferences:
        session.update_preferences(dp.preferences)

    # Capture tool calls via the existing on_event hook.
    # tool_start fires with (name, inputs); tool_end fires with (name, result).
    # We match them by position so each entry gets both params and result.
    tool_call_log: list[dict] = []
    tool_output_log: list[dict] = []
    current_turn = [0]

    def on_event(event: str, *args):
        if event == "tool_start":
            tool_call_log.append({
                "tool": args[0],
                "turn_index": current_turn[0],
                "params": args[1],
            })
            tool_output_log.append({
                "tool": args[0],
                "turn_index": current_turn[0],
                "params": args[1],
                "result": None,  # filled in by tool_end
            })
        elif event == "tool_end":
            # Find the most recent matching entry without a result yet
            for entry in reversed(tool_output_log):
                if entry["tool"] == args[0] and entry["result"] is None:
                    entry["result"] = args[1]
                    break

    agent = WeeklyPlannerAgent(
        session=session, tools=ToolRunner(),
        on_event=on_event, user_id=EVAL_USER_ID,
    )
    responses: list[str] = []
    error: str | None = None
    t0 = time.time()

    # item.observe() sets the root trace ID so all agent.chat() calls (which
    # are @observe-decorated) attach as child spans of the SAME trace.
    # On context exit the trace is automatically linked to the dataset item run.
    faithfulness_score = helpfulness_score = failure_explanation_score = None
    faithfulness_reason = helpfulness_reason = failure_explanation_reason = ""

    try:
        with item.observe(run_name=run_name) as trace_id:
            for i, turn in enumerate(dp.turns):
                current_turn[0] = i
                responses.append(agent.chat(turn))

            ssa = _score_ssa(dp, session)
            tsa = _score_tsa(dp, tool_call_log)
            tpa = _score_tpa(dp, tool_call_log)

            # ── LLM-as-judge metrics (GPT-4.5) ───────────────────────────────
            faithfulness_score, faithfulness_reason = judge_faithfulness(
                dp.turns, tool_output_log, responses
            )
            helpfulness_score, helpfulness_reason = judge_helpfulness(
                dp.turns, responses
            )
            if dp.category in ("edge_case", "graceful_failure"):
                failure_explanation_score, failure_explanation_reason = judge_failure_explanation(
                    dp.turns, responses
                )

            passed = ssa >= 0.9 and tsa >= 0.95 and tpa >= 0.9

            # Push deterministic scores
            lf.score(trace_id=trace_id, name="tool_selection_accuracy", value=tsa)
            lf.score(trace_id=trace_id, name="tool_param_accuracy",     value=tpa)
            lf.score(trace_id=trace_id, name="session_state_accuracy",  value=ssa)

            # Push LLM-as-judge scores
            lf.score(trace_id=trace_id, name="faithfulness",  value=faithfulness_score,
                     comment=faithfulness_reason)
            lf.score(trace_id=trace_id, name="helpfulness",   value=helpfulness_score,
                     comment=helpfulness_reason)
            if failure_explanation_score is not None:
                lf.score(trace_id=trace_id, name="failure_explanation",
                         value=failure_explanation_score, comment=failure_explanation_reason)

            lf.score(trace_id=trace_id, name="case_passed", value=1.0 if passed else 0.0)

    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        ssa = tsa = tpa = 0.0
        faithfulness_score = helpfulness_score = failure_explanation_score = 0.0
        passed = False

    duration = round(time.time() - t0, 2)
    status = "PASS" if passed else "FAIL"
    llm_judge_str = (
        f" faith={faithfulness_score:.2f} help={helpfulness_score:.2f}"
        if faithfulness_score is not None else ""
    )
    if failure_explanation_score is not None:
        llm_judge_str += f" fail_exp={failure_explanation_score:.2f}"
    logger.info(
        "  [%s] %-12s %s  tsa=%.2f tpa=%.2f ssa=%.2f%s  (%ss)",
        dp.category, dp.id, status, tsa, tpa, ssa,
        llm_judge_str,
        duration,
    )
    if error:
        logger.debug("    ERROR: %s", error[:200])

    return {
        "name": dp.id,
        "category": dp.category,
        "passed": passed,
        "duration_seconds": duration,
        "scores": {
            "tsa": tsa,
            "tpa": tpa,
            "ssa": ssa,
            "faithfulness": faithfulness_score,
            "faithfulness_reason": faithfulness_reason,
            "helpfulness": helpfulness_score,
            "helpfulness_reason": helpfulness_reason,
            "failure_explanation": failure_explanation_score,
            "failure_explanation_reason": failure_explanation_reason,
        },
        "tool_call_log": tool_call_log,
        "error": error,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Langfuse eval dataset — one LLM call per case")
    parser.add_argument("--dataset",  default=DATASET_NAME, help="Langfuse dataset name")
    parser.add_argument("--run-name", default=None,         help="Dataset run name (default: timestamp)")
    parser.add_argument("--output",   default=None,         help="JSON output path (default: eval_results/)")
    args = parser.parse_args()

    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "OPENAI_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"Missing env var: {var} — add it to .env")

    from langfuse import Langfuse
    from eval_data.dataset import ALL_DATAPOINTS

    lf = Langfuse()
    run_name = args.run_name or datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")

    # Fetch dataset from Langfuse
    logger.info("Fetching dataset %r …", args.dataset)
    try:
        dataset = lf.get_dataset(args.dataset)
    except Exception as e:
        sys.exit(
            f"Could not fetch dataset {args.dataset!r}: {e}\n"
            f"Run 'uv run python evals/push_to_langfuse.py' first."  # push_to_langfuse stays in evals/
        )

    # Index local datapoints by id for session checks / answer checks
    dp_by_id = {dp.id: dp for dp in ALL_DATAPOINTS}

    # Match Langfuse items to local datapoints
    pairs = []
    for item in dataset.items:
        dp_id = (item.metadata or {}).get("id")
        dp = dp_by_id.get(dp_id)
        if dp is None:
            logger.warning("  skipping item %s — no matching local EvalDatapoint", item.id)
            continue
        pairs.append((item, dp))

    if not pairs:
        sys.exit("No matching datapoints found. Re-push the dataset with push_to_langfuse.py.")

    # Wipe any file-backed eval_user session from previous runs
    if os.path.isdir(_SESSION_DIR):
        shutil.rmtree(_SESSION_DIR)

    logger.info("Run: %r  |  %d items  |  dataset: %r\n", run_name, len(pairs), args.dataset)

    results = []
    by_category: dict[str, list] = {}

    for item, dp in pairs:
        result = run_item(item, dp, lf, run_name)
        results.append(result)
        by_category.setdefault(dp.category, []).append(result)

    # Flush all scores / links to Langfuse before exiting
    lf.flush()

    # ── Summary ────────────────────────────────────────────────────────────────
    total   = len(results)
    n_pass  = sum(1 for r in results if r["passed"])
    pct     = n_pass / total * 100 if total else 0

    logger.info("\n%s", "=" * 56)
    logger.info("Run: %s", run_name)
    logger.info("%-24s  %d/%d  (%.0f%%)", "OVERALL", n_pass, total, pct)

    category_scores = {}
    for cat, cat_results in sorted(by_category.items()):
        cp = sum(1 for r in cat_results if r["passed"])
        ct = len(cat_results)
        category_scores[cat] = {"passed": cp, "total": ct, "pct": round(cp / ct * 100, 1)}
        logger.info("  %-22s  %d/%d  (%.0f%%)", cat, cp, ct, cp / ct * 100)

    host = os.environ["LANGFUSE_HOST"].rstrip("/")
    logger.info("\nLangfuse → %s/datasets/%s/runs/%s",
                host, args.dataset.replace(" ", "%20"), run_name.replace(" ", "%20"))

    # ── Save JSON ──────────────────────────────────────────────────────────────
    report = {
        "run_name": run_name,
        "dataset":  args.dataset,
        "summary": {
            "total_cases": total,
            "passed_cases": n_pass,
            "overall_pct": round(pct, 1),
        },
        "by_category": category_scores,
        "cases": results,
    }

    if args.output:
        out_path = args.output
    else:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(RESULTS_DIR, f"{stamp}_langfuse_{run_name}.json")

    pathlib.Path(out_path).write_text(json.dumps(report, indent=2))
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()