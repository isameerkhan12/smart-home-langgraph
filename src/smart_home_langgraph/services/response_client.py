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


def _build_prompt_messages(state: AgentState, max_tokens: int = 2000) -> list:
    """
    Build a chat-style prompt with both global and step-specific memory contexts.
    
    For multi-step queries:
      - Includes global memory (overall context)
      - Includes step-specific memory (current step focus, top 1-2 results)
    
    For single-step queries:
      - Uses only global memory
    """
    system_content = (
        "You are a smart-home assistant.\n"
        "Use the provided memory context and tools for data analysis.\n"
        "Carry context forward across the conversation when the user asks follow-up questions.\n"
        "Prioritize: safety, avoid known mistakes, reuse proven recipe, and obey preferences.\n\n"
        f"Detected Intent:\n{state['intent']}\n\n"
    )
    
    # Include global memory for all queries
    system_content += f"Global Memory Context:\n{state['memory_context']}\n\n"
    
    # For multi-step queries, include step-specific memory (strict, focused context)
    if state.get("step_specific_memory"):
        system_content += f"For This Step:\n{state['step_specific_memory']}\n\n"
    
    system_content += "Return a concise actionable answer in 4-6 bullet points."

    system_message = SystemMessage(content=system_content)
    # filtered_messages = _filter_tool_messages(state["messages"])
    # trimmed_history = trim_messages(
    #     filtered_messages,
    #     strategy="last",
    #     token_counter=count_tokens_approximately,
    #     max_tokens=max_tokens,
    # )
    # return [system_message, *trim_messages, HumanMessage(content=state["user_query"])]
    return [system_message, HumanMessage(content=state["user_query"])]


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
    """Build a prompt asking the LLM to propose a step-by-step strategy before executing."""
    system_content = (
        "You are a smart-home data analysis assistant. Before writing any code, "
        "propose a short numbered plan of the dependent steps needed to answer the question.\n"
        "Each step should state what will be computed and why it is needed for the next step.\n"
        "Do not write Python code or call any tool. Only describe the plan.\n\n"
        f"Detected Intent: {state['intent']}\n\n"
    )
    if state.get("memory_context"):
        system_content += (
            f"Memory Context:\n{state['memory_context']}\n\n"
            "Decide whether this memory is relevant to each plan step.\n"
            "At the top of the plan, include a 'Memory usage:' line stating whether the plan uses memory overall or is built from scratch.\n"
            "For every step, include exactly one line in this format: 'Memory used: yes|partial|no - <brief description>'.\n"
            "Use 'yes' when the step is based entirely on memory, 'partial' when it combines memory with fresh computation, and 'no - none' when it is computed entirely from fresh data.\n\n"
        )

    plan_critique = state.get("plan_critique_result")
    if state.get("plan") and plan_critique and not plan_critique.get("passed", True):
        system_content += (
            f"Previous Plan (needs revision):\n{state['plan']}\n\n"
            f"Revision Hints:\n{plan_critique.get('repair_hints', '')}\n\n"
        )

    system_message = SystemMessage(content=system_content)
    return [system_message, HumanMessage(content=state["user_query"])]


def propose_plan(state: AgentState) -> str:
    """Generate a step-by-step plan text for a multi-step question."""
    prompt_messages = _build_plan_prompt_messages(state)
    text, _ = _generate_text(state, prompt_messages)
    return text


def _build_tool_system_content(state: AgentState) -> str:
    """Build the shared context and tool instructions for data analysis."""
    system_content = (
        "You are a smart-home data analysis assistant with access to tools.\n\n"
        "You have access to a Python execution tool that can analyze the smart home energy consumption data (House_1 dataset).\n"
        "The DataFrame 'df' is pre-loaded with the following columns:\n"
        "- Time: Datetime string (YYYY-MM-DD HH:MM:SS)\n"
        "- Unix: Unix timestamp (seconds since epoch)\n"
        "- Aggregate: Total power consumption in Watts\n"
        "- Fridge, Chest_Freezer, Upright_Freezer, Tumble_Dryer, Washing_Machine, Dishwasher, Computer_Site, Television_Site, Electric_Heater: Individual appliance power in Watts\n"
        "- timestamp: Datetime version of Unix timestamp (for convenient filtering by date/time)\n"
        "Dataset date range: October 9, 2013 to July 10, 2015 with detailed per-appliance power readings.\n\n"
        "When the user asks a data analysis question:\n"
        "1. FIRST read the Memory Context below. For any numeric fact already present\n"
        "   (e.g. 'Washing Machine March 2014 = 7.36 kWh'), assign it as a Python variable\n"
        "   at the top of your code (e.g. washing_machine_kwh = 7.36) and use that\n"
        "   variable in your calculation — do NOT recompute it from the DataFrame.\n"
        "2. Query the DataFrame only for missing values or an explicitly required verification.\n"
        "3. Print the result of each calculation.\n"
        "4. Use proper pandas methods (mean(), sum(), value_counts(), etc.).\n"
        "5. Handle potential errors gracefully.\n\n"
        f"Detected Intent: {state['intent']}\n\n"
        f"Memory Usage Mode: {state.get('memory_usage_mode', 'none')}\n\n"
        f"step results : {state.get('tool_result', 'none')}\n\n"
    )

    if state.get("memory_context"):
        system_content += f"Memory Context:\n{state['memory_context']}\n\n"

    return system_content


def _build_single_step_tool_prompt_messages(state: AgentState) -> list:
    """Build the tool prompt for a question answerable in one execution flow."""
    system_content = _build_tool_system_content(state)
    system_content += (
        "This is a single-step question. Complete the requested analysis in the current tool flow.\n"
        "When the answer is complete, call submit_final_answer.\n\n"
    )

    system_message = SystemMessage(content=system_content)
    return [system_message, *state.get("messages", []), HumanMessage(content=state["user_query"])]


def _build_multi_step_tool_prompt_messages(state: AgentState) -> list:
    """Build the tool prompt for executing one step of an approved plan."""
    system_content = _build_tool_system_content(state)

    if state.get("plan"):
        system_content += (
            f"Approved Plan:\n{state['plan']}\n\n"
            "Identify the next unfinished plan step and execute only this step.\n"
            "Reuse results from completed steps.\n"
            "The first line of every Python tool call must be a comment declaring long-term memory usage.\n"
            "Use exactly one of these formats:\n"
            "# Memory used: none\n"
            "# Memory used: yes - <specific memory value, recipe, or fact>\n"
            "# Memory used: partial - <specific memory value, recipe, or fact reused; explain what will be computed from fresh data>\n"
            "Report only long-term memory used directly in this step.\n"
            "Values produced by earlier plan steps are intermediate results, not long-term memory.\n"
            "If this step uses only those intermediate results and does not read from Memory Context, write:\n"
            "# Memory used: none\n"
            "Use memory values directly and do not recompute them unless verification is explicitly required.\n"
            "Do not call submit_final_answer until every plan step is complete.\n\n"
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

