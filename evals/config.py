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

# Per-metric pass thresholds (used by run_evals.py)
PASS_THRESHOLD_SSA = 0.9
PASS_THRESHOLD_TSA = 0.95
PASS_THRESHOLD_TPA = 0.9

# Baseline preferences applied to every eval session before per-case overrides.
# Goals:
#   - current_time="08:00" → any task time (e.g. "1 PM") is always in the future
#   - work_start/end = full day → no task falls outside the work window by accident
#   - timezone/location → weather tests work without asking the user for their city
# Per-case `preferences` and `current_time` fields are applied on top and override these.
EVAL_DEFAULT_PREFERENCES: dict = {
    "current_time": "08:00",
    "work_start": "00:00",
    "work_end": "23:59",
    "timezone": "Asia/Kolkata",
    "location_name": "Bengaluru, Karnataka, India",
    "latitude": 12.9716,   # pre-set to skip geocoding API calls during evals
    "longitude": 77.5946,
}