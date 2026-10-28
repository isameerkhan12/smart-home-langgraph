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


MEMORY_EVALUATOR_PROMPT = """\
You are a planning assistant. Given a question and retrieved memories, produce
three kinds of output: a task-level annotation, an entry-level assessment of
each memory, and a set-level decision derived from those assessments.

Memory types: [recipe] = a working procedure or code approach. Other types hold
facts, values, preferences, or past mistakes.

PART 1: Task annotation (look only at the question, ignore the memories)
- is_multi_step: true if answering requires several dependent steps, where a
    later step needs the result of an earlier one; false for a single lookup or
    single calculation.

PART 2: Entry assessment (judge each memory in isolation, ignoring the others)
Assign exactly one contribution value per memory:
- "complete": this memory alone contains the final answer, with entities,
    parameters, dates, and units matching the question exactly.
- "partial": this memory supplies a required value or intermediate result, but
    not the final answer.
- "guidance": this memory supplies no values, but a recipe, preference, or past
    mistake that should shape how the answer is computed.
- "unrelated": using this memory would not change how the question is answered.
A memory whose conditions differ from the question (other parameters, entity,
or date) cannot be "complete".

PART 3: Set decision (derive from Part 2, looking at all memories together)
- "irrelevant": every entry is "unrelated".
- "sufficient": the entries together contain the final answer and nothing
    remains to be computed or looked up. This holds if at least one entry is
    "complete", or if "partial" entries cover every required value and need no
    further calculation to combine. "guidance" entries alone never make the set
    sufficient.
- "supportive": at least one entry is not "unrelated", but the set is not
    sufficient.
If the user explicitly asks to recalculate, verify, or recompute, never choose
"sufficient".
- tools_usage: false for "sufficient", true otherwise.

Output a single JSON object, no code fences, no text before or after, in this
shape (2-space indentation):
{
    "is_multi_step": true/false,
    "memory_evaluations": [
        {
            "id": "<memory id>",
            "memory": "<memory text, copied exactly, JSON-escaped>",
            "contribution": "complete" | "partial" | "guidance" | "unrelated",
            "reason": "<one short sentence>"
        }
    ],
    "memory_set_verdict": "sufficient" | "supportive" | "irrelevant",
    "tools_usage": true/false,
    "overall_reason": "<one sentence>"
}

Rules: one entry per memory, in the original order, without the type label. If
the Memory Context is empty, use "memory_evaluations": [] and verdict
"irrelevant".

Respond ONLY with the JSON object."""


def _build_memory_evaluator_prompt(state: AgentState) -> list:
    """Build the memory evaluator prompt messages."""
    user_content = (
        f"Question: {state['user_query']}\n\n"
        f"Memory Context:\n{state.get('memory_context', 'No memory available.')}"
    )
    return [
        SystemMessage(content=MEMORY_EVALUATOR_PROMPT),
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
        PlannerDecision with entry assessments and a set-level verdict.
    """
    settings = get_settings()

    # Fallback: if no memory context, tools are needed
    memory_context = state.get("memory_context", "")
    if not memory_context or "No relevant long-term memories found" in memory_context:
        return PlannerDecision(
            tools_usage=True,
            memory_set_verdict="irrelevant",
            is_multi_step=False,
            overall_reason="No relevant memory available; tools needed for computation.",
        )

    # Fallback: if API key missing, default to using memory to avoid tool errors
    if _requires_provider_api_key(settings) and not settings.openrouter_api_key:
        return PlannerDecision(
            tools_usage=False,
            memory_set_verdict="sufficient",
            is_multi_step=False,
            overall_reason="API key missing; defaulting to memory-based response.",
        )

    prompt_messages = _build_memory_evaluator_prompt(state)

    try:
        model = build_model(settings, temperature=0.0)
        result = model.invoke(prompt_messages)
        return result_to_model(result, PlannerDecision)
    except Exception as exc:  # noqa: BLE001 - robust fallback on any error
        # On error, default to tools (safer to compute than give wrong answer)
        return PlannerDecision(
            tools_usage=True,
            memory_set_verdict="irrelevant",
            is_multi_step=False,
            overall_reason=f"Planner error ({exc}); defaulting to tools.",
        )
