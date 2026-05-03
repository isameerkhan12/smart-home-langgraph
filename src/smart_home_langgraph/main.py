# ---------------------------------------------------------------------------
# main.py
# Purpose: Entry point for running the smart-home agent as an interactive chat.
#
# Usage (from project root, with venv activated):
#   $env:PYTHONPATH = "src"
#   python -m smart_home_langgraph.main
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver

from smart_home_langgraph.graph.workflow import build_workflow, initial_state


class ChatSession:
    """Reusable multi-turn chat wrapper around the compiled workflow."""

    def __init__(self, app=None, max_repairs: int = 2, thread_id: str = "default") -> None:
        self._conn = sqlite3.connect(database="smart_home_agent.db", check_same_thread=False)
        self._app = app or build_workflow(max_repairs=max_repairs, checkpointer=SqliteSaver(conn=self._conn))
        self._max_repairs = max_repairs
        self._thread_id = thread_id  # stable ID = same conversation resumes on restart

    def ask(self, query: str) -> tuple[str, int]:
        result = self._app.invoke(
            initial_state(
                query,
                max_repairs=self._max_repairs,
            ),
            config={"configurable": {"thread_id": self._thread_id}},
        )
        return result["response"], result["memory_written_count"]

    def reset(self) -> None:
        # Start a new thread so old conversation is preserved but not continued.
        self._thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def interactive_chat(max_repairs: int = 2, thread_id: str = "default") -> None:
    """Start a terminal chat loop that preserves conversation history."""
    session = ChatSession(max_repairs=max_repairs, thread_id=thread_id)
    print(f"Smart-home assistant chat [thread: {session._thread_id}]. Type 'exit' to quit or 'reset' to start a new conversation.")

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return

        if not query:
            continue
        lowered = query.lower()
        if lowered in {"exit", "quit"}:
            print("Session ended.")
            return
        if lowered == "reset":
            session.reset()
            print(f"New conversation started [thread: {session._thread_id}].")
            continue

        response, _ = session.ask(query)
        print(f"\nAssistant: {response}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the smart-home LangGraph agent.")
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=2,
        help="Maximum number of critique/repair loops per turn.",
    )
    parser.add_argument(
        "--thread",
        default="default",
        help="Thread ID to resume a previous conversation. Defaults to 'default'.",
    )
    args = parser.parse_args()
    interactive_chat(max_repairs=args.max_repairs, thread_id=args.thread)


if __name__ == "__main__":
    main()
