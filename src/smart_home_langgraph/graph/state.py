# ---------------------------------------------------------------------------
# graph/state.py
# Purpose: Unified typed state schema for the smart-home agent.
#
# AgentState is the shared "notebook" passed between every node in the graph.
# Each node receives this dict, reads the fields it needs, adds/updates
# its own fields, and returns the updated dict.
#
# Fields by concern:
#   - Query + intent:    what the user asked and how we classified it
#   - Context:           sensor summary + long-term memory retrieved for this run
#   - Response:          the final answer and whether it came from a live LLM call
#   - Tools:             tool execution state and results
#   - Critique + repair: structured quality evaluation and self-repair tracking
#   - Learning metrics:  episode outcome recorded for trend analysis
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from smart_home_langgraph.evaluation.metrics import EpisodeRecord


class ToolExecutionResult(TypedDict):
    """Result from a tool execution."""
    
    # Name of the tool that was executed.
    tool_name: str
    
    # Input that was passed to the tool.
    tool_input: str
    
    # Output returned by the tool.
    tool_output: str
    
    # Whether the execution succeeded.
    success: bool
    
    # Error message if execution failed.
    error: str


class CritiqueResult(TypedDict):
    """Structured output from the critique node."""

    # True if response passes quality check, False if repairs needed.
    passed: bool

    # List of identified issues (empty if passed).
    issues: list[str]

    # Review outcome: "success", "minor_revision", "major_revision", or "fail".
    severity: str

    # Hints/suggestions for repairing the response.
    repair_hints: str

    # Why the response was accepted (empty when not passed).
    pass_reasons: list[str]

    # Critique execution status for observability/debugging.
    # Values: "not_run", "completed", "skipped_config", "fallback_error".
    critique_status: str


class AgentState(TypedDict):
    """Complete state schema shared by every node in the workflow."""

    # ---- User input --------------------------------------------------------
    # The raw natural-language question from the user.
    user_query: str

    # Conversation messages merged with LangGraph's message-aware reducer.
    # Named "messages" for compatibility with LangGraph's tools_condition.
    messages: Annotated[list[BaseMessage], add_messages]

    # Summary of earlier messages kept separate to avoid losing context.
    summary: str

    # Intent label inferred from the query (e.g. "energy_optimization").
    intent: str

    # ---- Context (populated by retrieve_context node) ----------------------
    # Long-term memory context assembled from mistakes, recipes, preferences.
    memory_context: str

    # Error-focused memory context retrieved for repair attempts.
    error_memory_context: str

    # Normalized error signature used for deduping and retrieval.
    error_signature: str

    # ---- Planner decision (populated by memory_evaluator node) --------------
    # True when memory evaluator decides memory is sufficient; tools will not be bound.
    use_memory_only: bool

    # Explanation from memory evaluator for the decision.
    planner_reason: str

    # ---- Response (populated by generate_response / repair_response nodes) --
    # The final answer to return to the user. May be regenerated on repair.
    response: str

    # True when a live LLM was called successfully; False when fallback was used.
    used_live_llm: bool

    # ---- Tool execution state ----------------------------------------------
    # The last tool execution result (if any).
    tool_result: ToolExecutionResult | None

    # Generated code from the LLM (for code execution tools).
    generated_code: str

    # Number of tool execution attempts for this query.
    tool_execution_count: int

    # ---- Critique + repair loop --------------------------------------------
    # Structured quality feedback from the critique node.
    critique_result: CritiqueResult

    # Current repair attempt count (starts at 0, incremented on each repair).
    repair_count: int

    # Maximum number of repair attempts allowed before exiting the loop.
    max_repairs: int

    # ---- Learning metrics --------------------------------------------------
    # Per-episode outcome metrics populated by the memory_writer node.
    episode_record: EpisodeRecord

    # Number of records written to memory stores in this episode.
    memory_written_count: int

    # Number of error-memory records written in this episode.
    error_memory_written_count: int
