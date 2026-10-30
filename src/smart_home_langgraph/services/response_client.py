# ---------------------------------------------------------------------------
# services/response_client.py
#
# Purpose:
#   Build the final generation prompt and call the configured LLM provider to
#   produce a response. If the live provider is unavailable (missing key,
#   network issue, API error), return a deterministic fallback so the workflow
#   still completes safely.
#
#   Also supports tool-enabled generation using bind_tools for data analysis.
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool

from smart_home_langgraph.config.settings import get_settings


def _filter_tool_messages(messages: list) -> list:
    """
    Filter out AIMessages with tool_calls and their corresponding ToolMessages.
    
    Some providers are strict about preserving function-call turn ordering.
    When trimming history, this sequence can break and cause invalid request
    errors. We filter tool-related messages from history before generation.
    """
    filtered = []
    for msg in messages:
        # Skip AIMessages that have tool_calls (function call turns)
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            continue
        # Skip ToolMessages (function response turns)
        if isinstance(msg, ToolMessage):
            continue
        filtered.append(msg)
    return filtered
from smart_home_langgraph.graph.state import AgentState
from smart_home_langgraph.services.json_utils import result_to_text
from smart_home_langgraph.services.model_factory import build_model


def _requires_provider_api_key(settings) -> bool:
    """Return True only when current provider requires an API key."""
    return settings.llm_provider.lower() == "openrouter"


def _provider_api_key_name(settings) -> str:
    """Return required API key env var name for current provider."""
    if settings.llm_provider.lower() == "openrouter":
        return "OPENROUTER_API_KEY"
    return "API_KEY"


DATA_CONTEXT = """\
Dataset (House 1 energy data), preloaded as the pandas DataFrame `df`:
- Time: datetime string; Unix: epoch seconds; timestamp: datetime (use it to filter by date or time).
- Aggregate, Fridge, Chest_Freezer, Upright_Freezer, Tumble_Dryer, Washing_Machine, Dishwasher, Computer_Site, Television_Site, Electric_Heater: power in Watts.
- Range: 2013-10-09 to 2015-07-10.
- Energy in kWh is not stored; derive the sampling interval from `df['Unix'].diff()` instead of assuming it.
- To convert power readings to energy, multiply each reading by the interval in seconds and divide by 3,600,000."""

ANSWER_FORMAT = (
    "Answer format: short bullet points, one fact per bullet with its value, unit, and period. "
    "No introduction and no advice."
)


def format_memory_assessment(state: AgentState) -> str:
    """Render the memory evaluator's verdict and non-unrelated memories for prompts."""
    useful = [m for m in state.get("memory_evaluations", []) if m["contribution"] != "unrelated"]
    lines = [f"Memory set verdict: {state.get('memory_set_verdict', 'irrelevant')}"]
    lines += [f"- [{m['contribution']}] {m['memory']} ({m['reason']})" for m in useful]
    if not useful:
        lines.append("(no useful memory)")
    return "\n".join(lines)


def _build_prompt_messages(state: AgentState, max_tokens: int = 2000) -> list:
    """Build the memory-only answer prompt (no tools are bound on this path)."""
    system_content = (
        "You are a smart-home data analysis assistant. The retrieved memory already contains the final answer.\n"
        "Write the answer from memory only.\n\n"
        f"Memory (assessed by the memory evaluator):\n{format_memory_assessment(state)}\n\n"
        "Rules:\n"
        "- Use only values that appear in the memory above; copy numbers, units, and periods exactly.\n"
        "- Do not compute, estimate, or add facts that are not in the memory.\n"
        "- If the memory does not fully answer the question, state which part is missing.\n\n"
        f"{ANSWER_FORMAT}"
    )
    return [SystemMessage(content=system_content), HumanMessage(content=state["user_query"])]


def _fallback_response(state: AgentState, reason: str) -> str:
    """Return a deterministic fallback response when live LLM call is unavailable."""
    return (
        "[Phase 3 fallback: live LLM call not used] "
        f"Reason: {reason}. "
        f"Intent detected: {state['intent']}. "
        "Use memory context and available tools to produce a practical next step."
    )


def _generate_text(state: AgentState, prompt_messages: list) -> tuple[str, bool]:
    """Invoke the configured provider on prompt_messages, with a deterministic fallback on failure."""
    settings = get_settings()
    if _requires_provider_api_key(settings) and not settings.openrouter_api_key:
        return _fallback_response(state, f"missing {_provider_api_key_name(settings)}"), False

    try:
        model = build_model(settings, temperature=0.2)
        result = model.invoke(prompt_messages)
        return result_to_text(result).strip(), True
    except Exception as exc:  # noqa: BLE001 - we want robust fallback for all runtime failures
        return _fallback_response(state, f"{settings.llm_provider} error: {exc}"), False


def generate_response(
    state: AgentState,
) -> tuple[str, bool]:
    """
    Generate a response with the configured provider.

    Args:
        state: AgentState used for generation.

    Returns:
      (response_text, used_live_llm)
    """
    prompt_messages = _build_prompt_messages(state)
    _ = count_tokens_approximately(prompt_messages)
    return _generate_text(state, prompt_messages)


def _build_plan_prompt_messages(state: AgentState) -> list:
    """Build the prompt that asks for a step-by-step plan before any code runs."""
    system_content = (
        "You are a planning assistant for smart-home energy data analysis. "
        "Propose a numbered plan for the question. Do not write code or call tools.\n\n"
        f"{DATA_CONTEXT}\n\n"
        f"Memory (assessed by the memory evaluator):\n{format_memory_assessment(state)}\n\n"
        "Planning rules:\n"
        "- Each step is one computation that runs as a single Python call. Merge trivial steps "
        "(for example filtering and summing the same column) into one step.\n"
        "- Use as few steps as the question needs; each step must use the result of an earlier step or the dataset.\n"
        "- Take values from [complete] and [partial] memories as given. Do not recompute or verify them unless the user asks.\n"
        "- Follow [guidance] memories for how to compute.\n"
        "- Do not add data-quality or verification steps unless the user asks.\n"
        "- The last step produces the final answer.\n\n"
        "Output format, one line per step and nothing else:\n"
        "Step <n>: <what is computed> | uses: <memory value, earlier step, or dataset>"
    )

    plan_critique = state.get("plan_critique_result")
    if state.get("plan") and plan_critique and not plan_critique.get("passed", True):
        system_content += (
            f"\n\nPrevious plan (needs revision):\n{state['plan']}\n\n"
            f"Revision hints:\n{plan_critique.get('repair_hints', '')}"
        )

    return [SystemMessage(content=system_content), HumanMessage(content=state["user_query"])]


def propose_plan(state: AgentState) -> str:
    """Generate a step-by-step plan text for a multi-step question."""
    prompt_messages = _build_plan_prompt_messages(state)
    text, _ = _generate_text(state, prompt_messages)
    return text


def _build_tool_system_content(state: AgentState) -> str:
    """Build the context and rules shared by single-step and multi-step tool prompts."""
    return (
        "You are a smart-home data analysis assistant with two tools: a Python tool and submit_final_answer.\n\n"
        f"{DATA_CONTEXT}\n\n"
        f"Memory (assessed by the memory evaluator):\n{format_memory_assessment(state)}\n\n"
        "Rules:\n"
        "- Take values from [complete] memories as final and from [partial] memories as supplied components. "
        "Assign them to Python variables and use them; compute only missing components. Do not recompute or verify them unless the user asks.\n"
        "- Follow [guidance] memories for how to compute.\n"
        "- Query `df` only for what memory does not supply.\n"
        "- Print every value you compute.\n"
        "- Submit the answer with submit_final_answer.\n\n"
        f"{ANSWER_FORMAT}\n\n"
    )


def _build_single_step_tool_prompt_messages(state: AgentState) -> list:
    """Build the tool prompt for a question answerable in one execution flow."""
    system_content = _build_tool_system_content(state)
    system_content += (
        "This is a single-step question. Compute it in as few Python calls as possible, "
        "then call submit_final_answer.\n"
    )

    system_message = SystemMessage(content=system_content)
    return [system_message, *state.get("messages", []), HumanMessage(content=state["user_query"])]


def _build_multi_step_tool_prompt_messages(state: AgentState) -> list:
    """Build the tool prompt for executing one step of an approved plan."""
    system_content = _build_tool_system_content(state)

    if state.get("plan"):
        system_content += (
            f"Approved plan:\n{state['plan']}\n\n"
            "Memory retrieved for the current step (not evaluated; use it only if it applies):\n"
            f"{state['step_specific_memory']}\n\n"
            "Execution rules:\n"
            f"- You are executing plan step {state['current_step']}. Run only that step, in one Python call, and reuse the results of earlier steps.\n"
            "- Do not solve future steps or repeat completed steps.\n"
            f"- The first line of the code must be a comment: `# Plan step {state['current_step']}: <what this step computes>`.\n"
            "- Call submit_final_answer only after every plan step is done, using the results of the steps.\n\n"
        )

    step_feedback = state.get("step_critique_result")
    if step_feedback and not step_feedback.get("passed", True) and step_feedback.get("repair_hints"):
        system_content += (
            f"Feedback on the previous step:\n{step_feedback['repair_hints']}\n"
            "Correct this step before continuing with the plan.\n\n"
        )

    system_message = SystemMessage(content=system_content)
    return [system_message, *state.get("messages", []), HumanMessage(content=state["user_query"])]


def _build_tool_prompt_messages(state: AgentState, max_tokens: int = 2000) -> list:
    """Select the focused tool prompt for single-step or multi-step execution."""
    if state.get("is_multi_step", False):
        return _build_multi_step_tool_prompt_messages(state)
    return _build_single_step_tool_prompt_messages(state)


def generate_response_with_tools(
    state: AgentState,
    tools: Sequence[BaseTool],
) -> AIMessage:
    """
    Generate a response using an LLM with tools bound.
    
    Uses LangChain's bind_tools to attach tool definitions to the LLM.
    The LLM can then decide whether to call tools or respond directly.
    
    Args:
        state: AgentState used for generation.
        tools: Sequence of tools to bind to the LLM.
        
    Returns:
        AIMessage with potential tool_calls or direct content.
    """
    settings = get_settings()
    
    # Handle missing API key
    if _requires_provider_api_key(settings) and not settings.openrouter_api_key:
        return AIMessage(
            content=_fallback_response(state, f"missing {_provider_api_key_name(settings)}")
        )
    
    prompt_messages = _build_tool_prompt_messages(state)
    
    try:
        # Build the model and bind tools
        model = build_model(settings, temperature=0.2)
        model_with_tools = model.bind_tools(tools)
        
        # Invoke the model
        result = model_with_tools.invoke(prompt_messages)
        
        # Ensure we return an AIMessage
        if isinstance(result, AIMessage):
            return result
        else:
            # Wrap the result in an AIMessage if needed
            return AIMessage(content=result_to_text(result))
            
    except Exception as exc:  # noqa: BLE001 - robust fallback on any error
        return AIMessage(
            content=_fallback_response(state, f"{settings.llm_provider} error: {exc}")
        )


def generate_tool_enabled_response(
    state: AgentState,
    tools: Sequence[BaseTool],
) -> AIMessage:
    """Backward-compatible alias used by workflow nodes."""
    return generate_response_with_tools(state, tools)

