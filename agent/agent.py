"""
Daily Planner Agent — conversational loop.

Architecture:
  - One-shot agentic loop per user turn (Claude may call multiple tools per turn).
  - Session state lives in SessionManager; conversation history drives multi-turn memory.
  - All tool logic is deterministic Python — the LLM only decides *which* tool to call.
"""

import json
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

from .memory import SessionManager
from .tools import TOOL_DEFINITIONS, run_tool

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
3. Call get_schedule before reporting the schedule to the user — never invent slot times.
4. When a task cannot be scheduled, clearly explain what could not fit and why.
5. Ask for duration if the user mentions a task without one.
6. Confirm significant changes (removing tasks, overriding the whole schedule) before executing.
7. Be concise. Use bullet points for the schedule. Show times in HH:MM format.

You have full session memory — previous tasks and the schedule persist across this conversation."""


class DailyPlannerAgent:
    def __init__(self, session_file: Optional[str] = None, verbose: bool = False):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.session = SessionManager(session_file)
        self.verbose = verbose

    def chat(self, user_message: str) -> str:
        """Process one user turn and return the agent's response."""
        self.session.add_message("user", user_message)

        response_text = self._run_agentic_loop()

        self.session.add_message("assistant", response_text)
        self.session.save()
        return response_text

    def _run_agentic_loop(self) -> str:
        """
        Inner loop: send messages → execute tools → repeat until end_turn.
        Returns the final text response.
        """
        messages = list(self.session.get_history())

        while True:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            if self.verbose:
                print(f"  [loop] stop_reason={response.stop_reason}, blocks={len(response.content)}")

            if response.stop_reason == "end_turn":
                # Extract final text
                return "\n".join(
                    block.text for block in response.content if hasattr(block, "text")
                )

            if response.stop_reason == "tool_use":
                # Append assistant message (may contain both text and tool_use blocks)
                messages.append({"role": "assistant", "content": response.content})

                # Execute all requested tools and collect results
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    if self.verbose:
                        print(f"  [tool] {block.name}({json.dumps(block.input)[:120]})")

                    result = run_tool(block.name, block.input, self.session)

                    if self.verbose:
                        print(f"  [result] {json.dumps(result)[:200]}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — treat as done
            return "\n".join(
                block.text for block in response.content if hasattr(block, "text")
            )

    def reset(self):
        """Clear all session state (useful for testing)."""
        from .memory import SessionState
        self.session.state = SessionState()
        self.session.save()
