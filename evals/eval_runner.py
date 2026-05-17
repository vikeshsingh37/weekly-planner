"""
Eval runner — executes all test cases and produces a structured report.

Each case runs in an isolated agent instance (fresh session, shared preferences).
Results are printed to stdout and optionally written to a JSON file.
"""

import json
import logging
import os
import shutil
import time
import traceback
from dataclasses import dataclass, asdict
from typing import List, Optional

from agent.agent import WeeklyPlannerAgent
from impl.memory import JSONSessionManager
from impl.tools import ToolRunner
from evals.eval_data import EvalCase, ALL_CASES

logger = logging.getLogger(__name__)

EVAL_USER_ID = "eval_user"
_EVAL_SESSION_DIR = os.path.join("sessions", EVAL_USER_ID)


@dataclass
class CheckResult:
    description: str
    passed: bool
    error: Optional[str] = None


@dataclass
class CaseResult:
    name: str
    category: str
    passed: bool
    checks: List[CheckResult]
    agent_responses: List[str]
    duration_seconds: float
    error: Optional[str] = None


def run_case(case: EvalCase) -> CaseResult:
    """Run one eval case in an isolated in-memory session tagged as eval_user."""
    session = JSONSessionManager(session_file=None)
    agent = WeeklyPlannerAgent(session=session, tools=ToolRunner(), user_id=EVAL_USER_ID)

    # Use a 24-hour window by default so evals are not time-of-day sensitive.
    # Case-specific preferences override these defaults.
    session.update_preferences({"work_start": "00:00", "work_end": "23:59"})
    if case.preferences:
        session.update_preferences(case.preferences)

    responses: List[str] = []
    error = None
    t_start = time.time()

    try:
        for turn in case.turns:
            logger.debug("    User: %s", turn[:80])
            response = agent.chat(turn)
            responses.append(response)
            logger.debug("    Agent: %s", response[:120])
    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    duration = time.time() - t_start

    check_results = []
    all_passed = error is None

    for check in case.checks:
        if error:
            check_results.append(CheckResult(check.description, False, "Case errored"))
            continue
        try:
            passed = check.fn(session, responses)
            check_results.append(CheckResult(check.description, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            check_results.append(CheckResult(check.description, False, str(e)))
            all_passed = False

    return CaseResult(
        name=case.name,
        category=case.category,
        passed=all_passed,
        checks=check_results,
        agent_responses=responses,
        duration_seconds=round(duration, 2),
        error=error,
    )


def run_all(
    cases: List[EvalCase] = None,
    output_file: Optional[str] = None,
    categories: Optional[List[str]] = None,
) -> dict:
    cases = cases or ALL_CASES
    if categories:
        cases = [c for c in cases if c.category in categories]

    # Wipe any persisted file-backed session for eval_user so history never leaks between runs.
    if os.path.isdir(_EVAL_SESSION_DIR):
        shutil.rmtree(_EVAL_SESSION_DIR)
        logger.debug("Cleared eval session dir: %s", _EVAL_SESSION_DIR)

    logger.info("\n%s", "=" * 60)
    logger.info("Weekly Planner Agent — Eval Suite")
    logger.info("Running %d cases", len(cases))
    logger.info("%s\n", "=" * 60)

    results: List[CaseResult] = []
    by_category: dict = {}

    for case in cases:
        result = run_case(case)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        logger.info("  [%s] %s ... %s (%ss)", case.category, case.name, status, result.duration_seconds)

        if not result.passed:
            for cr in result.checks:
                if not cr.passed:
                    err_str = f" [{cr.error}]" if cr.error else ""
                    logger.info("    ✗ %s%s", cr.description, err_str)
        if result.error:
            logger.debug("    ERROR: %s", result.error[:300])

        by_category.setdefault(result.category, []).append(result)

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("RESULTS BY CATEGORY")
    logger.info("%s", "=" * 60)

    total_pass = 0
    total_cases = len(results)
    category_scores = {}

    for cat, cat_results in sorted(by_category.items()):
        n_pass = sum(1 for r in cat_results if r.passed)
        n_total = len(cat_results)
        pct = n_pass / n_total * 100
        category_scores[cat] = {"passed": n_pass, "total": n_total, "pct": round(pct, 1)}
        total_pass += n_pass
        logger.info("  %-25s %d/%d (%.0f%%)", cat, n_pass, n_total, pct)

    overall_pct = total_pass / total_cases * 100 if total_cases else 0
    logger.info("\n  %-25s %d/%d (%.0f%%)", "OVERALL", total_pass, total_cases, overall_pct)

    # Per-check breakdown
    total_checks = sum(len(r.checks) for r in results)
    passed_checks = sum(sum(1 for c in r.checks if c.passed) for r in results)
    logger.info(
        "  %-25s %d/%d (%.0f%%)",
        "Check-level",
        passed_checks,
        total_checks,
        passed_checks / total_checks * 100 if total_checks else 0,
    )
    logger.info("%s\n", "=" * 60)

    report = {
        "summary": {
            "total_cases": total_cases,
            "passed_cases": total_pass,
            "overall_pct": round(overall_pct, 1),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "check_pct": round(passed_checks / total_checks * 100, 1) if total_checks else 0,
        },
        "by_category": category_scores,
        "cases": [
            {
                "name": r.name,
                "category": r.category,
                "passed": r.passed,
                "duration_seconds": r.duration_seconds,
                "checks": [asdict(c) for c in r.checks],
                "error": r.error,
            }
            for r in results
        ],
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Report written to: %s", output_file)

    return report
