"""Eval suite configuration — judge model, thresholds, and shared paths."""

import os

# LLM-as-judge (OpenAI)
JUDGE_MODEL = "gpt-5.1"
JUDGE_MAX_TOKENS = 256

# Shared eval identity
EVAL_USER_ID = "eval_user"
EVAL_SESSION_DIR = os.path.join("sessions", EVAL_USER_ID)
RESULTS_DIR = "eval_results"

# Langfuse dataset
DATASET_NAME = "weekly-planner-v1"
DATASET_DESCRIPTION = (
    "25 hand-crafted eval cases for the Weekly Planner agent covering "
    "tool selection, parameter extraction, multi-turn coherence, answer "
    "faithfulness, and edge cases."
)

# Per-metric pass thresholds (used by run_langfuse_eval.py)
PASS_THRESHOLD_SSA = 0.9
PASS_THRESHOLD_TSA = 0.95
PASS_THRESHOLD_TPA = 0.9