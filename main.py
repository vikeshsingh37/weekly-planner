#!/usr/bin/env python3
"""
Daily Planner Agent — interactive CLI entry point.

Usage:
    python main.py                     # fresh session
    python main.py --session my_day    # named persistent session
    python main.py --verbose           # show tool calls
"""

import argparse
import os
import sys

from agent.agent import DailyPlannerAgent
from impl.memory import JSONSessionManager
from impl.tools import ToolRunner


def main():
    parser = argparse.ArgumentParser(description="Daily Planner conversational agent")
    parser.add_argument("--session", default=None, help="Session name for persistence")
    parser.add_argument("--verbose", action="store_true", help="Show tool calls")
    args = parser.parse_args()

    session_file = None
    if args.session:
        os.makedirs("sessions", exist_ok=True)
        session_file = f"sessions/{args.session}.json"

    session = JSONSessionManager(session_file=session_file)
    agent = DailyPlannerAgent(session=session, tools=ToolRunner(), verbose=args.verbose)

    print("Daily Planner Agent  (type 'quit' or Ctrl-C to exit)")
    print("─" * 50)
    if session_file and os.path.exists(session_file):
        print(f"Resuming session: {args.session}")
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

        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
