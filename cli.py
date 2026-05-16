#!/usr/bin/env python3
"""
Weekly Planner Agent — interactive CLI entry point.

Usage:
    uv run python cli.py                   # uses your system username
    uv run python cli.py --user alice      # explicit user (calendar persists across sessions)
    uv run python cli.py --verbose         # show tool calls
"""

import argparse
import getpass
import logging
import os
import sys
import threading
import time

from agent.agent import WeeklyPlannerAgent
from impl.memory import JSONSessionManager
from impl.tools import ToolRunner

logger = logging.getLogger(__name__)

# ── Animated thinking display ──────────────────────────────────────────────────

_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_TOOL_LABELS: dict[str, str] = {
    "parse_and_add_tasks": "Adding tasks",
    "schedule_tasks":      "Scheduling",
    "move_task":           "Moving task",
    "remove_task":         "Removing task",
    "get_schedule":        "Reading schedule",
    "update_preferences":  "Updating preferences",
}


class ThinkingDisplay:
    """
    Single-line animated display that updates in-place with \\r.

    While working:
        ⠙ Thinking...
        ⠙ Scheduling...       ← label changes as tools run

    On finish, collapses to a dim summary on its own line:
        ▸ Adding tasks  ·  Scheduling  ·  Reading schedule
    """

    def __init__(self) -> None:
        self._label = "Thinking..."
        self._done_steps: list[str] = []
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._frame = 0
        self._last_width = 0
        self._thread: threading.Thread | None = None
        self._is_tty = sys.stdout.isatty()

    def start(self) -> None:
        if not self._is_tty:
            return
        self._label = "Thinking..."
        self._done_steps = []
        self._last_width = 0
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def on_event(self, event: str, *args) -> None:
        if not self._is_tty:
            return
        with self._lock:
            if event == "tool_start":
                self._label = _TOOL_LABELS.get(args[0], args[0]) + "..."
            elif event == "tool_end":
                self._done_steps.append(_TOOL_LABELS.get(args[0], args[0]))
                self._label = "Thinking..."

    def finish(self) -> None:
        if not self._is_tty:
            return
        self._stop_evt.set()
        if self._thread:
            self._thread.join()
            self._thread = None
        # Erase the spinner line
        sys.stdout.write(f"\r{' ' * (self._last_width + 4)}\r")
        sys.stdout.flush()
        # Collapsed summary
        if self._done_steps:
            summary = "  ·  ".join(self._done_steps)
            sys.stdout.write(f"\033[2m▸ {summary}\033[0m\n")
            sys.stdout.flush()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            with self._lock:
                self._draw()
            time.sleep(0.08)
            self._frame += 1

    def _draw(self) -> None:
        f = _FRAMES[self._frame % len(_FRAMES)]
        text = f"{f} {self._label}"
        # Pad to overwrite any longer previous label
        padding = max(0, self._last_width - len(text))
        sys.stdout.write(f"\r\033[2m{text}\033[0m{' ' * padding}")
        sys.stdout.flush()
        self._last_width = len(text)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weekly Planner conversational agent")
    parser.add_argument(
        "--user",
        default=None,
        help="User ID — calendar and tasks persist across sessions (default: system username)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show tool calls")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stdout,
    )

    user_id = args.user or getpass.getuser()
    session_file = f"sessions/{user_id}/state.json"
    os.makedirs(f"sessions/{user_id}", exist_ok=True)

    display = ThinkingDisplay()
    session = JSONSessionManager(session_file=session_file)
    agent = WeeklyPlannerAgent(
        session=session, tools=ToolRunner(), on_event=display.on_event, user_id=user_id
    )

    print("Weekly Planner Agent  (type 'quit' or Ctrl-C to exit)")
    print("─" * 50)
    if os.path.exists(session_file):
        print(f"Resuming calendar for user: {user_id}")
    else:
        print(f"New calendar for user: {user_id}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        display.start()
        response = agent.chat(user_input)
        display.finish()

        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()