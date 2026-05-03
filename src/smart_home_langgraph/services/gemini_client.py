# ---------------------------------------------------------------------------
# services/gemini_client.py
#
# Purpose:
#   Build the final generation prompt and call Gemini to produce a response.
#   If Gemini is unavailable (missing key, network issue, API error), return
#   a deterministic fallback so the workflow still completes safely.
# ---------------------------------------------------------------------------
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.graph.state import AgentState


def _build_prompt_messages(state: AgentState, max_turns: int = 6) -> list:
    """Build a simple chat-style prompt using LangChain message objects."""
    system_message = SystemMessage(
        content=(
            "You are a smart-home assistant.\n"
            "Use the provided sensor and memory context.\n"
            "Carry context forward across the conversation when the user asks follow-up questions.\n"
            "Prioritize: safety, avoid known mistakes, reuse proven recipe, and obey preferences.\n\n"
            f"Detected Intent:\n{state['intent']}\n\n"
            f"Sensor Context:\n{state['sensor_context']}\n\n"
            f"Memory Context:\n{state['memory_context']}\n\n"
            "Return a concise actionable answer in 4-6 bullet points."
        )
    )
    history = state["conversation_history"][-max_turns:]
    return [system_message, *history, HumanMessage(content=state["user_query"])]


def _fallback_response(state: AgentState, reason: str) -> str:
    """Return a deterministic fallback response when live LLM call is unavailable."""
    return (
        "[Phase 3 fallback: live Gemini call not used] "
        f"Reason: {reason}. "
        f"Intent detected: {state['intent']}. "
        "Use sensor and memory context to produce a practical next step."
    )


def generate_with_gemini(state: AgentState) -> tuple[str, bool]:
    """
    Generate a response with Gemini.

    Returns:
      (response_text, used_live_llm)
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        return _fallback_response(state, "missing GEMINI_API_KEY"), False

    prompt_messages = _build_prompt_messages(state)

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
        return _fallback_response(state, f"Gemini error: {exc}"), False
