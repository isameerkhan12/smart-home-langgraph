# ---------------------------------------------------------------------------
# services/gemini_client.py
#
# Purpose:
#   Build the final generation prompt and call Gemini to produce a response.
#   If Gemini is unavailable (missing key, network issue, API error), return
#   a deterministic fallback so the workflow still completes safely.
# ---------------------------------------------------------------------------
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.graph.state import AgentState


@traceable(name="build_generation_prompt", run_type="prompt")
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
    # Trim conversation history to fit within token budget, keeping the most recent messages.
    trimmed_history = trim_messages(
        state["conversation_history"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
    )
    print(
        f"Prompt token count (approx): "
        f"{count_tokens_approximately([system_message] + trimmed_history) + count_tokens_approximately([HumanMessage(content=state['user_query'])])}"
    )
    return [system_message, *trimmed_history, HumanMessage(content=state["user_query"])]


@traceable(name="build_summary_prompt", run_type="prompt")
def _build_summary_prompt(messages_to_summarize: list) -> list:
    """Build a simple prompt for summarizing old conversation messages (no smart-home context)."""
    system_message = SystemMessage(
        content=(
            "You are a conversation summarizer. "
            "Concisely summarize the following messages, preserving key decisions and context. "
            "Be factual and avoid opinions."
        )
    )
    # Build a simple text representation of messages to summarize
    messages_text = "\n".join(
        [f"{getattr(m, 'type', 'unknown')}: {m.content}" for m in messages_to_summarize]
    )
    return [system_message, HumanMessage(content=messages_text)]


def _fallback_response(state: AgentState, reason: str) -> str:
    """Return a deterministic fallback response when live LLM call is unavailable."""
    return (
        "[Phase 3 fallback: live Gemini call not used] "
        f"Reason: {reason}. "
        f"Intent detected: {state['intent']}. "
        "Use sensor and memory context to produce a practical next step."
    )


@traceable(name="generate_with_gemini", run_type="llm")
def generate_with_gemini(state: AgentState = None, is_summary: bool = False, messages_to_summarize: list = None) -> tuple[str, bool]:
    """
    Generate a response with Gemini or summarize messages.
    
    Args:
        state: AgentState for normal generation (ignored if is_summary=True)
        is_summary: If True, summarize messages_to_summarize instead of generating response
        messages_to_summarize: List of messages to summarize (only used if is_summary=True)

    Returns:
      (response_text, used_live_llm)
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        if is_summary:
            return "[Summarization skipped: missing GEMINI_API_KEY]", False
        return _fallback_response(state, "missing GEMINI_API_KEY"), False

    # Choose prompt builder based on task
    if is_summary:
        prompt_messages = _build_summary_prompt(messages_to_summarize)
    else:
        prompt_messages = _build_prompt_messages(state)

    _ = count_tokens_approximately(prompt_messages)

    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        )
        result = model.invoke(prompt_messages)

        # LangChain model outputs can be either plain text or structured content.
        if isinstance(result.content, str):
            text = result.content
        else:
            text = str(result.content)

        return text.strip(), True
    except Exception as exc:  # noqa: BLE001 - we want robust fallback for all runtime failures
        if is_summary:
            return f"[Summarization error: {exc}]", False
        return _fallback_response(state, f"Gemini error: {exc}"), False
