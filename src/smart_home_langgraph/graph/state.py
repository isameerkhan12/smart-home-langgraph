# ---------------------------------------------------------------------------
# graph/state.py
# Purpose: Define the shared state that flows through the entire LangGraph.
#
# What is a LangGraph state?
#   Every node in the graph receives the current state as input and returns
#   an updated copy of that state as output. The graph merges the update
#   automatically and passes it to the next node.
#   Think of it as a shared notebook that every node can read and write to.
#
# Why TypedDict?
#   TypedDict lets us use a plain Python dict but with type hints on every
#   key, so editors and type checkers warn us if we use the wrong field name.
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    # The raw natural-language question or task from the user.
    user_query: str

    # The category we assign after reading the query, e.g. "energy_optimization".
    # Starts as an empty string; the detect_intent node fills it in.
    intent: str

    # The final answer we will show the user.
    # Starts as an empty string; the build_response node fills it in.
    response: str
