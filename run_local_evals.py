#!/usr/bin/env python3
"""
Eval entry point.

Usage:
    python run_evals.py                          # run all cases
    python run_evals.py --category memory        # single category
    python run_evals.py --verbose                # show tool calls + responses
    python run_evals.py --output results.json    # save report
"""

import argparse
import datetime
import logging
import os
import sys

from evals.eval_runner import run_all
from evals.eval_data import ALL_CASES
from evals.config import RESULTS_DIR

CATEGORIES = sorted({c.category for c in ALL_CASES})


def main():
    parser = argparse.ArgumentParser(description="Run weekly planner eval suite")
    parser.add_argument("--category", choices=CATEGORIES, help="Run only one category")
    parser.add_argument("--verbose", action="store_true", help="Show agent tool calls")
    parser.add_argument("--output", default=None, help="Path to write JSON report (auto-generated if omitted)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.output:
        output_file = args.output
    else:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_DIR, f"{stamp}.json")

    categories = [args.category] if args.category else None
    report = run_all(output_file=output_file, categories=categories)

    logging.getLogger().info("Results saved → %s", output_file)

    # Exit non-zero if any case failed (useful for CI)
    sys.exit(0 if report["summary"]["passed_cases"] == report["summary"]["total_cases"] else 1)


if __name__ == "__main__":
    main()
