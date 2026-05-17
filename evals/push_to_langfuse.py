#!/usr/bin/env python3
"""
Push the eval_data dataset to Langfuse.

Creates (or updates) one Langfuse dataset item per EvalDatapoint.
Uses the datapoint id as the stable item id, so re-running is safe — existing
items are upserted, not duplicated.

Usage:
    uv run python evals/push_to_langfuse.py
    uv run python evals/push_to_langfuse.py --dataset weekly-planner-v2
    uv run python evals/push_to_langfuse.py --dry-run

Requires: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

# Ensure project root is on sys.path when the script is run directly from evals/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATASET_NAME = "weekly-planner-v1"
DATASET_DESCRIPTION = (
    "25 hand-crafted eval cases for the Weekly Planner agent covering "
    "tool selection, parameter extraction, multi-turn coherence, answer "
    "faithfulness, and edge cases."
)


# ── Serialisation ─────────────────────────────────────────────────────────────

def _serialise_param_checks(checks: list) -> list[dict]:
    return [
        {
            "field": c.field,
            "op": c.op,
            "value": c.value,
            "description": c.description,
        }
        for c in checks
    ]


def _serialise_tool_calls(tool_calls: list) -> list[dict]:
    return [
        {
            "tool": tc.tool,
            "turn_index": tc.turn_index,
            "required": tc.required,
            "param_checks": _serialise_param_checks(tc.param_checks),
        }
        for tc in tool_calls
    ]


def _serialise_answer_checks(answer_checks: list) -> list[dict]:
    return [
        {
            "turn_index": ac.turn_index,
            "contains_all": ac.contains_all,
            "contains_any": ac.contains_any,
            "excludes": ac.excludes,
            "description": ac.description,
        }
        for ac in answer_checks
    ]


def _serialise_session_checks(session_checks: list) -> list[str]:
    # Lambda functions can't be serialised — store descriptions only.
    # The actual assertions run in the eval runner, not in Langfuse.
    return [c.description for c in session_checks]


def datapoint_to_item(dp) -> tuple[dict, dict, dict]:
    """Return (input, expected_output, metadata) dicts for one EvalDatapoint."""
    input_ = {
        "turns": dp.turns,
        "preferences": dp.preferences,
    }

    expected_output = {
        "expected_tool_calls": _serialise_tool_calls(dp.expected_tool_calls),
        "answer_checks": _serialise_answer_checks(dp.answer_checks),
        "session_checks": _serialise_session_checks(dp.session_checks),
    }

    metadata = {
        "id": dp.id,
        "category": dp.category,
        "description": dp.description,
        "metrics": dp.metrics,
        "notes": dp.notes,
        "turn_count": len(dp.turns),
    }

    return input_, expected_output, metadata


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Push eval dataset to Langfuse")
    parser.add_argument("--dataset", default=DATASET_NAME, help="Dataset name in Langfuse")
    parser.add_argument("--dry-run", action="store_true", help="Print items without uploading")
    args = parser.parse_args()

    # Validate Langfuse credentials before importing the heavy SDK
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        if not os.environ.get(var):
            sys.exit(
                f"Missing env var: {var}\n"
                "Copy .env.example to .env and fill in your Langfuse keys."
            )

    from eval_data.dataset import ALL_DATAPOINTS

    if args.dry_run:
        logger.info("DRY RUN — would push %d items to dataset %r", len(ALL_DATAPOINTS), args.dataset)
        for dp in ALL_DATAPOINTS:
            inp, exp, meta = datapoint_to_item(dp)
            logger.info("  [%s] %s  turns=%d  tools=%d  checks=%d",
                        dp.category, dp.id, len(dp.turns),
                        len(dp.expected_tool_calls), len(dp.session_checks))
        return

    from langfuse import Langfuse

    lf = Langfuse()

    # Create or retrieve the dataset
    logger.info("Creating dataset %r …", args.dataset)
    lf.create_dataset(
        name=args.dataset,
        description=DATASET_DESCRIPTION,
        metadata={"source": "eval_data/dataset.py", "version": args.dataset},
    )

    # Upsert all datapoints
    logger.info("Upserting %d items …\n", len(ALL_DATAPOINTS))
    by_category: dict[str, int] = {}

    for dp in ALL_DATAPOINTS:
        inp, exp, meta = datapoint_to_item(dp)

        # Stable global id: dataset-name + "__" + datapoint-id
        item_id = f"{args.dataset}__{dp.id}"

        lf.create_dataset_item(
            dataset_name=args.dataset,
            id=item_id,
            input=inp,
            expected_output=exp,
            metadata=meta,
        )

        by_category[dp.category] = by_category.get(dp.category, 0) + 1
        logger.info("  ✓  %-12s  %s", dp.category, dp.id)

    # Flush so all items are sent before the script exits
    lf.flush()

    logger.info("\nDone — %d items pushed to dataset %r", len(ALL_DATAPOINTS), args.dataset)
    logger.info("\nBy category:")
    for cat, count in sorted(by_category.items()):
        logger.info("  %-20s %d", cat, count)

    host = os.environ["LANGFUSE_HOST"].rstrip("/")
    logger.info("\nOpen in Langfuse → %s/datasets/%s", host, args.dataset.replace(" ", "%20"))


if __name__ == "__main__":
    main()