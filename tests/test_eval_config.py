"""Sanity checks for evals/config.py."""
from evals.config import (
    DATASET_DESCRIPTION,
    DATASET_NAME,
    EVAL_SESSION_DIR,
    EVAL_USER_ID,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    PASS_THRESHOLD_SSA,
    PASS_THRESHOLD_TPA,
    PASS_THRESHOLD_TSA,
    RESULTS_DIR,
)


class TestEvalConfig:
    def test_judge_model_is_nonempty_string(self):
        assert isinstance(JUDGE_MODEL, str) and JUDGE_MODEL

    def test_judge_max_tokens_positive(self):
        assert isinstance(JUDGE_MAX_TOKENS, int) and JUDGE_MAX_TOKENS > 0

    def test_eval_user_id(self):
        assert EVAL_USER_ID == "eval_user"

    def test_session_dir_contains_user_id(self):
        assert EVAL_USER_ID in EVAL_SESSION_DIR

    def test_results_dir_nonempty(self):
        assert isinstance(RESULTS_DIR, str) and RESULTS_DIR

    def test_dataset_name_nonempty(self):
        assert isinstance(DATASET_NAME, str) and DATASET_NAME

    def test_dataset_description_nonempty(self):
        assert isinstance(DATASET_DESCRIPTION, str) and DATASET_DESCRIPTION

    def test_thresholds_in_valid_range(self):
        for threshold in (PASS_THRESHOLD_SSA, PASS_THRESHOLD_TSA, PASS_THRESHOLD_TPA):
            assert 0.0 < threshold <= 1.0

    def test_tsa_threshold_stricter_than_ssa(self):
        assert PASS_THRESHOLD_TSA >= PASS_THRESHOLD_SSA