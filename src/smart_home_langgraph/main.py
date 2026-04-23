# ---------------------------------------------------------------------------
# main.py
# Purpose: Entry point for running the smart-home agent from the command line.
#
# Usage (from project root, with venv activated):
#   $env:PYTHONPATH = "src"
#   python -m smart_home_langgraph.main --query "Why is my heating bill so high?"
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse

from smart_home_langgraph.graph.workflow import build_workflow, initial_state


def run(query: str) -> tuple[str, int]:
    """
    Run the agent for a single user query.

    Returns:
      (response_text, memory_written_count)
    """
    app = build_workflow()
    result = app.invoke(initial_state(query))
    return result["response"], result["memory_written_count"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the smart-home LangGraph agent.")
    parser.add_argument(
        "--query",
        default="How can I reduce my evening power usage?",
        help="User question for the agent.",
    )
    args = parser.parse_args()

    response, written = run(args.query)

    print(response)


if __name__ == "__main__":
    main()
