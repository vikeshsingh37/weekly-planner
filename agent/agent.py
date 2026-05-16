"""
Weekly Planner Agent — conversational loop with Langfuse observability.

Depends only on api abstractions (AbstractSessionManager, AbstractToolRunner).
Concrete implementations are injected at construction time.

Tracing (Langfuse v2, REST-based — compatible with self-hosted Langfuse Server v2):
  chat()      → root span tagged with user_id / session_id
  _call_llm() → generation span with model name + token counts
  _run_tool() → tool span with dynamic tool name
"""

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anthropic
from dotenv import load_dotenv

from agent.config import MAX_LOOP_ITERATIONS, MAX_TOKENS, MODEL, SYSTEM_PROMPT
from api.memory import AbstractSessionManager
from api.tools import AbstractToolRunner, TOOL_DEFINITIONS

load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# Langfuse v2 decorators — silently disabled when keys are not set.
try:
    from langfuse.decorators import langfuse_context, observe
    _LANGFUSE = True
except ImportError:
    _LANGFUSE = False

    def observe(_fn=None, **_kw):  # type: ignore[misc]
        def _wrap(fn): return fn
        return _wrap(_fn) if callable(_fn) else _wrap

    class _DummyCtx:  # type: ignore[misc]
        def update_current_trace(self, **_): pass
        def update_current_observation(self, **_): pass
    langfuse_context = _DummyCtx()  # type: ignore[assignment]


class WeeklyPlannerAgent:
    def __init__(
        self,
        session: AbstractSessionManager,
        tools: AbstractToolRunner,
        on_event: Callable[..., None] | None = None,
        user_id: str | None = None,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._session = session
        self._tools = tools
        self._on_event: Callable[..., None] = on_event or (lambda *_: None)
        self._user_id = user_id

    @observe()
    def chat(self, user_message: str) -> str:
        """Process one user turn and return the agent's text response."""
        if self._user_id:
            langfuse_context.update_current_trace(
                user_id=self._user_id,
                session_id=self._user_id,
            )
        self._session.reload()
        self._session.add_message("user", user_message)
        response_text = self._run_agentic_loop()
        self._session.add_message("assistant", response_text)
        self._session.save()
        return response_text

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        prefs = self._session.state.preferences
        try:
            tz = ZoneInfo(prefs.timezone)
            now = datetime.now(tz)
            time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
            tz_label = prefs.timezone
        except (ZoneInfoNotFoundError, KeyError, Exception):
            now = datetime.now()
            time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
            tz_label = "local"
        return (
            SYSTEM_PROMPT
            + f"\n\nCurrent date/time ({tz_label}): {time_str}. "
            "Never schedule tasks at or before the current time."
        )

    def _run_agentic_loop(self) -> str:
        """Send messages → execute tools → repeat until end_turn or iteration cap."""
        messages = list(self._session.get_history())
        system = self._build_system_prompt()

        for iteration in range(MAX_LOOP_ITERATIONS):
            self._on_event("thinking_start")
            response = self._call_llm(messages, system)
            self._on_event("thinking_end")

            logger.debug(
                "  [loop %d/%d] stop=%s blocks=%d",
                iteration + 1, MAX_LOOP_ITERATIONS,
                response.stop_reason, len(response.content),
            )

            if response.stop_reason == "end_turn":
                return "\n".join(
                    b.text for b in response.content if hasattr(b, "text")
                )

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    logger.debug("  [tool] %s(%s)", block.name, json.dumps(block.input)[:100])
                    result = self._run_tool(block.name, block.input)
                    logger.debug("  [result] %s", json.dumps(result)[:200])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            return "\n".join(b.text for b in response.content if hasattr(b, "text"))

        logger.warning("  [loop] hit MAX_LOOP_ITERATIONS=%d, stopping", MAX_LOOP_ITERATIONS)
        return (
            "I've reached the maximum number of reasoning steps for this turn. "
            "Please try breaking your request into smaller parts."
        )

    @observe(as_type="generation")
    def _call_llm(self, messages: list, system: str):
        """Single Anthropic API call — traced as a Langfuse generation with token usage."""
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        langfuse_context.update_current_observation(
            model=MODEL,
            usage_details={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        )
        return response

    @observe(as_type="tool")
    def _run_tool(self, name: str, inputs: dict) -> dict:
        """Execute one tool call — traced as a Langfuse tool span."""
        langfuse_context.update_current_observation(name=f"tool:{name}")
        self._on_event("tool_start", name, inputs)
        result = self._tools.run(name, inputs, self._session)
        self._on_event("tool_end", name, result)
        return result