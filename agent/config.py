"""Agent configuration — model, LLM parameters, and system prompt."""

from pathlib import Path

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
MAX_LOOP_ITERATIONS = 10   # max LLM calls per chat() turn before giving up
TEMPERATURE = 1.0
THINKING_TYPE = "adaptive"
THINKING_EFFORT = "medium"

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text().strip()