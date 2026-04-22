# ---------------------------------------------------------------------------
# main.py
# Purpose: Entry point for running the agent from the command line.
#
# Usage (with venv activated, from project root):
#   $env:PYTHONPATH = "src"
#   python -m smart_home_langgraph.main --query "Why is my heating bill so high?"
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse  # standard library module for parsing command-line arguments

from smart_home_langgraph.graph.workflow import build_workflow


def run(query: str) -> str:
    # Build the compiled LangGraph app (all nodes + edges wired together).
    app = build_workflow()

    # invoke() runs the graph synchronously from the entry point to END.
    # We pass the initial state: user_query is set, intent and response start empty
    # because those will be filled in by the nodes as execution flows through the graph.
    result = app.invoke({"user_query": query, "intent": "", "response": ""})

    # result is the final AgentState dict after all nodes have run.
    # We only return the "response" field — the finished answer for the user.
    return result["response"]


def main() -> None:
    # argparse lets us pass --query "..." from the terminal without editing code.
    parser = argparse.ArgumentParser(description="Run the Phase 0 smart-home hello graph.")
    parser.add_argument(
        "--query",
        default="How can I reduce my evening power usage?",  # used if no --query is given
        help="User question for the graph.",
    )
    args = parser.parse_args()  # reads sys.argv and fills in args.query
    print(run(args.query))


# This block runs only when you execute `python main.py` directly.
# It does NOT run when another file imports main.py (e.g. the test file).
if __name__ == "__main__":
    main()
