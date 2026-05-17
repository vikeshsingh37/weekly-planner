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
    uv run python run_evals.py
    uv run python run_evals.py --dataset weekly-planner-v2
    uv run python run_evals.py --run-name sprint-12

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
from evals.config import (  # noqa: E402
    DATASET_NAME, RESULTS_DIR, EVAL_USER_ID,
    EVAL_SESSION_DIR as _SESSION_DIR,
    PASS_THRESHOLD_SSA, PASS_THRESHOLD_TSA, PASS_THRESHOLD_TPA,
    EVAL_DEFAULT_PREFERENCES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


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
    # Layer 1: eval-wide defaults (time-independent, Bengaluru location, 24h work window)
    session.update_preferences(EVAL_DEFAULT_PREFERENCES)
    # Layer 2: per-case overrides (e.g. FA_02 narrows work_start/work_end to test overflow)
    prefs_to_apply = dict(dp.preferences or {})
    if dp.current_time:
        prefs_to_apply["current_time"] = dp.current_time
    if prefs_to_apply:
        session.update_preferences(prefs_to_apply)

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
        session_id=f"{run_name}/{dp.id}",
    )
    responses: list[str] = []
    error: str | None = None
    t0 = time.time()

    # Pre-initialise all scores so exception handling never clobbers values that
    # were correctly computed before the failure point.  Deterministic scores
    # (tsa/tpa/ssa) must survive a judge failure; judge scores are independent.
    ssa = tsa = tpa = 0.0
    passed = False
    faithfulness_score = helpfulness_score = failure_explanation_score = None
    faithfulness_reason = helpfulness_reason = failure_explanation_reason = ""
    judge_error: str | None = None

    # item.observe() sets the root trace ID so all agent.chat() calls (which
    # are @observe-decorated) attach as child spans of the SAME trace.
    # On context exit the trace is automatically linked to the dataset item run.
    try:
        with item.observe(run_name=run_name) as trace_id:
            for i, turn in enumerate(dp.turns):
                current_turn[0] = i
                responses.append(agent.chat(turn))

            # Record the full multi-turn conversation as the trace output so
            # Langfuse shows every turn (not just the first @observe span's output).
            lf.trace(
                id=trace_id,
                output={
                    "turns_completed": len(responses),
                    "final_response": responses[-1] if responses else "",
                    "conversation": [
                        {"turn": i + 1, "user": dp.turns[i], "assistant": responses[i]}
                        for i in range(len(responses))
                    ],
                },
            )

            # Deterministic scores — computed from local state, cannot fail due to
            # external services, so they're safe here before the judge calls.
            ssa = _score_ssa(dp, session)
            tsa = _score_tsa(dp, tool_call_log)
            tpa = _score_tpa(dp, tool_call_log)

            # ── LLM-as-judge metrics (GPT-4.5) ───────────────────────────────
            # Wrapped in their own try so a judge failure (missing key, rate
            # limit, OpenAI outage) doesn't clobber the deterministic scores.
            judge_context = {
                "current_time": dp.current_time or None,
                "work_start": (dp.preferences or {}).get("work_start", "09:00"),
                "work_end": (dp.preferences or {}).get("work_end", "18:00"),
            }
            if "max_chunk_minutes" in (dp.preferences or {}):
                judge_context["max_chunk_minutes"] = dp.preferences["max_chunk_minutes"]

            try:
                faithfulness_score, faithfulness_reason = judge_faithfulness(
                    dp.turns, tool_output_log, responses, context=judge_context
                )
                helpfulness_score, helpfulness_reason = judge_helpfulness(
                    dp.turns, responses, context=judge_context
                )
                if dp.category in ("edge_case", "graceful_failure"):
                    failure_explanation_score, failure_explanation_reason = judge_failure_explanation(
                        dp.turns, responses, context=judge_context
                    )
            except Exception as judge_err:
                judge_error = f"{type(judge_err).__name__}: {judge_err}"
                logger.warning("  !! judge failed for %s: %s", dp.id, judge_error)

            passed = ssa >= PASS_THRESHOLD_SSA and tsa >= PASS_THRESHOLD_TSA and tpa >= PASS_THRESHOLD_TPA

            # Push deterministic scores
            lf.score(trace_id=trace_id, name="tool_selection_accuracy", value=tsa)
            lf.score(trace_id=trace_id, name="tool_param_accuracy",     value=tpa)
            lf.score(trace_id=trace_id, name="session_state_accuracy",  value=ssa)

            # Push LLM-as-judge scores (only if the judge succeeded)
            if faithfulness_score is not None:
                lf.score(trace_id=trace_id, name="faithfulness",
                         value=faithfulness_score, comment=faithfulness_reason)
            if helpfulness_score is not None:
                lf.score(trace_id=trace_id, name="helpfulness",
                         value=helpfulness_score, comment=helpfulness_reason)
            if failure_explanation_score is not None:
                lf.score(trace_id=trace_id, name="failure_explanation",
                         value=failure_explanation_score, comment=failure_explanation_reason)

            lf.score(trace_id=trace_id, name="case_passed", value=1.0 if passed else 0.0)

    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        # ssa/tsa/tpa/passed keep whatever values they held when the exception
        # fired: 0.0/False if turns failed, correctly computed if only the
        # Langfuse push failed.

    duration = round(time.time() - t0, 2)
    status = "PASS" if passed else "FAIL"
    if judge_error:
        llm_judge_str = " faith=N/A help=N/A [judge failed]"
    elif faithfulness_score is not None:
        llm_judge_str = f" faith={faithfulness_score:.2f} help={helpfulness_score:.2f}"
        if failure_explanation_score is not None:
            llm_judge_str += f" fail_exp={failure_explanation_score:.2f}"
    else:
        llm_judge_str = ""
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
        "judge_error": judge_error,
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
    # Track multi-turn flag alongside results for cross-cut reporting.
    # multi_turn here means len(turns) > 1, NOT the "multi_turn" category —
    # cases in tool_selection, tool_params, final_answer, and edge_case can
    # also be multi-turn.
    by_multi_turn: dict[bool, list] = {True: [], False: []}

    for item, dp in pairs:
        result = run_item(item, dp, lf, run_name)
        result["num_turns"] = dp.num_turns
        result["is_multi_turn"] = dp.is_multi_turn
        results.append(result)
        by_category.setdefault(dp.category, []).append(result)
        by_multi_turn[dp.is_multi_turn].append(result)

    # Flush all scores / links to Langfuse before exiting
    lf.flush()

    # ── Summary ────────────────────────────────────────────────────────────────
    total   = len(results)
    n_pass  = sum(1 for r in results if r["passed"])
    pct     = n_pass / total * 100 if total else 0

    n_judge_fail = sum(1 for r in results if r.get("judge_error"))

    logger.info("\n%s", "=" * 56)
    logger.info("Run: %s", run_name)
    logger.info("%-24s  %d/%d  (%.0f%%)", "OVERALL", n_pass, total, pct)

    logger.info("\nBy category (primary test focus):")
    category_scores = {}
    for cat, cat_results in sorted(by_category.items()):
        cp = sum(1 for r in cat_results if r["passed"])
        ct = len(cat_results)
        category_scores[cat] = {"passed": cp, "total": ct, "pct": round(cp / ct * 100, 1)}
        logger.info("  %-22s  %d/%d  (%.0f%%)", cat, cp, ct, cp / ct * 100)

    # Cross-cut: single-turn vs multi-turn (independent of category).
    # 13 of 25 cases have > 1 turn; they span all 5 categories.
    logger.info("\nBy conversation length (cross-cutting all categories):")
    multi_turn_scores = {}
    for is_mt, mt_results in [(True, by_multi_turn[True]), (False, by_multi_turn[False])]:
        if not mt_results:
            continue
        label = "multi-turn (turns > 1)" if is_mt else "single-turn"
        cp = sum(1 for r in mt_results if r["passed"])
        ct = len(mt_results)
        multi_turn_scores[label] = {"passed": cp, "total": ct, "pct": round(cp / ct * 100, 1)}
        logger.info("  %-24s  %d/%d  (%.0f%%)", label, cp, ct, cp / ct * 100)

    if n_judge_fail:
        logger.warning("\n  !! LLM judge failed on %d/%d cases — check OPENAI_API_KEY and quota", n_judge_fail, total)
        for r in results:
            if r.get("judge_error"):
                logger.warning("     %s: %s", r["name"], r["judge_error"])

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
            "judge_failures": n_judge_fail,
        },
        "by_category": category_scores,
        "by_multi_turn": multi_turn_scores,
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