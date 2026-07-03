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
    """Build a chat-style prompt trimmed to token limit using LangChain's trim_messages."""
    system_content = (
        "You are a smart-home assistant.\n"
        "Use the provided sensor and memory context.\n"
        "Carry context forward across the conversation when the user asks follow-up questions.\n"
        "Prioritize: safety, avoid known mistakes, reuse proven recipe, and obey preferences.\n\n"
        f"Detected Intent:\n{state['intent']}\n\n"
        f"Sensor Context:\n{state['sensor_context']}\n\n"
        f"Memory Context:\n{state['memory_context']}\n\n"
    )
    # Include summary if it exists (from previous message summarization)
    if state.get("summary"):
        system_content += f"[CONVERSATION SUMMARY]\n{state['summary']}\n\n"

    system_content += "Return a concise actionable answer in 4-6 bullet points."

    system_message = SystemMessage(content=system_content)
    # Filter out tool-related messages to avoid provider sequencing errors,
    # then trim conversation history to fit within token budget.
    filtered_messages = _filter_tool_messages(state["messages"])
    trimmed_history = trim_messages(
        filtered_messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
    )
    print(
        f"Prompt token count (approx): "
        f"{count_tokens_approximately([system_message] + trimmed_history) + count_tokens_approximately([HumanMessage(content=state['user_query'])])}"
    )
    return [system_message, *trimmed_history, HumanMessage(content=state["user_query"])]


def _fallback_response(state: AgentState, reason: str) -> str:
    """Return a deterministic fallback response when live LLM call is unavailable."""
    return (
        "[Phase 3 fallback: live LLM call not used] "
        f"Reason: {reason}. "
        f"Intent detected: {state['intent']}. "
        "Use sensor and memory context to produce a practical next step."
    )


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
    settings = get_settings()
    if _requires_provider_api_key(settings) and not settings.openrouter_api_key:
        return _fallback_response(state, f"missing {_provider_api_key_name(settings)}"), False

    prompt_messages = _build_prompt_messages(state)

    _ = count_tokens_approximately(prompt_messages)

    try:
        model = build_model(settings, temperature=0.2)
        result = model.invoke(prompt_messages)
        text = result_to_text(result)
        return text.strip(), True
    except Exception as exc:  # noqa: BLE001 - we want robust fallback for all runtime failures
        return _fallback_response(state, f"{settings.llm_provider} error: {exc}"), False


def _build_tool_prompt_messages(state: AgentState, max_tokens: int = 2000) -> list:
    """
    Build a prompt for tool-enabled generation.
    
    This prompt instructs the LLM to use the available tools (primarily python_repl)
    to analyze the smart home telemetry data and answer calculation questions.
    """
    system_content = (
        "You are a smart-home data analysis assistant with access to tools.\n\n"
        "You have access to a Python execution tool that can analyze the smart home telemetry data.\n"
        "The DataFrame 'df' is pre-loaded with the following columns:\n"
        "- Unix_Timestamp: Event timestamp\n"
        "- Transaction_ID: Unique event ID\n"
        "- Television, Dryer, Oven, Refrigerator, Microwave: Appliance states (0=off, 1=on)\n"
        "- Line_Voltage, Voltage: Power readings in volts\n"
        "- Apparent_Power: Power in VA\n"
        "- Energy_Consumption_kWh: Energy consumed in kWh\n"
        "- Offloading_Decision: Smart grid decision (0/1)\n"
        "- Bandwidth: Network bandwidth\n"
        "- timestamp: Datetime version of Unix_Timestamp\n\n"
        "When the user asks a data analysis question:\n"
        "1. Use the python_repl tool to write and execute pandas code\n"
        "2. Always print() the final result\n"
        "3. Use proper pandas methods (mean(), sum(), value_counts(), etc.)\n"
        "4. Handle potential errors gracefully\n\n"
        f"Detected Intent: {state['intent']}\n\n"
    )
    
    # Add context if available
    if state.get("sensor_context"):
        system_content += f"Recent Sensor Summary:\n{state['sensor_context']}\n\n"
    
    if state.get("memory_context"):
        system_content += f"Memory Context:\n{state['memory_context']}\n\n"
    
    if state.get("summary"):
        system_content += f"[CONVERSATION SUMMARY]\n{state['summary']}\n\n"
    
    system_message = SystemMessage(content=system_content)
    
    # Filter out tool-related messages to avoid provider sequencing errors,
    # then trim conversation history.
    filtered_messages = _filter_tool_messages(state["messages"])
    trimmed_history = trim_messages(
        filtered_messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
    )
    
    return [system_message, *trimmed_history, HumanMessage(content=state["user_query"])]


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

