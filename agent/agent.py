"""
Daily Planner Agent — conversational loop.

Depends only on api abstractions (AbstractSessionManager, AbstractToolRunner).
Concrete implementations are injected at construction time, making this
trivially testable and swappable (e.g., swap JSONSessionManager for Redis).
"""

import json
import os

import anthropic
from dotenv import load_dotenv

from api.memory import AbstractSessionManager
from api.tools import AbstractToolRunner, TOOL_DEFINITIONS

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a focused daily planning assistant. You help users:
- Collect tasks with durations, priorities, and deadlines
- Schedule tasks efficiently into their workday
- Adjust plans when constraints change
- Handle impossible schedules honestly

Guidelines:
1. Always call parse_and_add_tasks when the user mentions new tasks.
2. Always call schedule_tasks after adding/removing tasks to assign time slots.
3. Call get_schedule before reporting the schedule — never invent slot times.
4. When a task cannot be scheduled, clearly explain what could not fit and why.
5. Ask for duration if the user mentions a task without one.
6. Be concise. Use bullet points for the schedule. Show times in HH:MM format.

You have full session memory — previous tasks and the schedule persist across this conversation."""


class DailyPlannerAgent:
    def __init__(
        self,
        session: AbstractSessionManager,
        tools: AbstractToolRunner,
        verbose: bool = False,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._session = session
        self._tools = tools
        self._verbose = verbose

    def chat(self, user_message: str) -> str:
        """Process one user turn and return the agent's text response."""
        self._session.add_message("user", user_message)
        response_text = self._run_agentic_loop()
        self._session.add_message("assistant", response_text)
        self._session.save()
        return response_text

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_agentic_loop(self) -> str:
        """Send messages → execute tools → repeat until end_turn."""
        messages = list(self._session.get_history())

        while True:
            response = self._client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            if self._verbose:
                print(f"  [loop] stop={response.stop_reason} blocks={len(response.content)}")

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
                    if self._verbose:
                        print(f"  [tool] {block.name}({json.dumps(block.input)[:100]})")
                    result = self._tools.run(block.name, block.input, self._session)
                    if self._verbose:
                        print(f"  [result] {json.dumps(result)[:200]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — return whatever text we have
            return "\n".join(b.text for b in response.content if hasattr(b, "text"))
