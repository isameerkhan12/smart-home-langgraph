# ---------------------------------------------------------------------------
# services/planner_client.py
#
# Purpose:
#   Evaluate whether retrieved memory is sufficient to answer the user's
#   question, or if tools (e.g., python_repl) are needed for computation.
#   This decision gates tool access in the response generation step.
# ---------------------------------------------------------------------------
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.graph.state import AgentState
from smart_home_langgraph.services.json_utils import result_to_model
from smart_home_langgraph.services.llm_schema import PlannerDecision
from smart_home_langgraph.services.model_factory import build_model


PLANNER_PROMPT = """\
You are a planning assistant. Your job is to decide whether the user's question \
can be fully answered using ONLY the provided memory context, or if computation/tools are needed.

Guidelines:
- Select memory_only when memory contains the specific values, facts, or conclusions needed.
- Select memory_only when memory contains a semantically equivalent answer (same question, same data).
- Select partial when memory contains some required values but not all values needed to finish the answer.
- Select none when memory is empty, irrelevant, or missing key facts required for the answer.
- If the user explicitly asks to recalculate, verify, or recompute, avoid memory_only.
- Be conservative: if uncertain whether memory is sufficient, choose partial or none.

Mode mapping:
- memory_only -> tools_usage=false
- partial -> tools_usage=true
- none -> tools_usage=true

Also decide is_multi_step:
- true when answering requires several dependent calculation steps, where a later
  step needs the result of an earlier step (e.g. "find X, then using X compute Y").
- false for single-step lookups or single calculations.

Respond with a JSON object:
{
    "tools_usage": true/false,
    "memory_usage_mode": "memory_only" | "partial" | "none",
    "is_multi_step": true/false,
  "reason": "brief explanation (1 sentence)"
}

Respond ONLY with the JSON object."""


def _build_planner_prompt(state: AgentState) -> list:
    """Build the planner prompt messages."""
    user_content = (
        f"Question: {state['user_query']}\n\n"
        f"Memory Context:\n{state.get('memory_context', 'No memory available.')}"
    )
    return [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=user_content),
    ]


def _requires_provider_api_key(settings) -> bool:
    """Return True only when current provider requires an API key."""
    return settings.llm_provider.lower() == "openrouter"


def evaluate_memory_sufficiency(state: AgentState) -> PlannerDecision:
    """
    Decide if memory is sufficient to answer the question.

    Args:
        state: AgentState with user_query and memory_context populated.

    Returns:
        PlannerDecision with tools_usage (bool), memory_usage_mode, and reason.
    """
    settings = get_settings()

    # Fallback: if no memory context, tools are needed
    memory_context = state.get("memory_context", "")
    if not memory_context or "No relevant long-term memories found" in memory_context:
        return PlannerDecision(
            tools_usage=True,
            memory_usage_mode="none",
            is_multi_step=False,
            reason="No relevant memory available; tools needed for computation.",
        )

    # Fallback: if API key missing, default to using memory to avoid tool errors
    if _requires_provider_api_key(settings) and not settings.openrouter_api_key:
        return PlannerDecision(
            tools_usage=False,
            memory_usage_mode="memory_only",
            is_multi_step=False,
            reason="API key missing; defaulting to memory-based response.",
        )

    prompt_messages = _build_planner_prompt(state)

    try:
        model = build_model(settings, temperature=0.0)
        result = model.invoke(prompt_messages)
        return result_to_model(result, PlannerDecision)
    except Exception as exc:  # noqa: BLE001 - robust fallback on any error
        # On error, default to tools (safer to compute than give wrong answer)
        return PlannerDecision(
            tools_usage=True,
            memory_usage_mode="none",
            is_multi_step=False,
            reason=f"Planner error ({exc}); defaulting to tools.",
        )
