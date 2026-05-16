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
import logging
import sys

from evals.eval_runner import run_all
from evals.test_cases import ALL_CASES

CATEGORIES = sorted({c.category for c in ALL_CASES})


def main():
    parser = argparse.ArgumentParser(description="Run weekly planner eval suite")
    parser.add_argument("--category", choices=CATEGORIES, help="Run only one category")
    parser.add_argument("--verbose", action="store_true", help="Show agent tool calls")
    parser.add_argument("--output", default=None, help="Path to write JSON report")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    categories = [args.category] if args.category else None
    report = run_all(
        output_file=args.output,
        categories=categories,
    )
    # Exit non-zero if any case failed (useful for CI)
    sys.exit(0 if report["summary"]["passed_cases"] == report["summary"]["total_cases"] else 1)


if __name__ == "__main__":
    main()
