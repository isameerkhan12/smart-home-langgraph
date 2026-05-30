# ---------------------------------------------------------------------------
# services/summary_client.py
#
# Purpose:
#   Build summary prompt messages and call the configured LLM provider to
#   summarize older conversation history.
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langsmith import traceable

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.services.json_utils import result_to_text
from smart_home_langgraph.services.model_factory import build_model


def _requires_provider_api_key(settings) -> bool:
    """Return True only when current provider requires an API key."""
    return settings.llm_provider.lower() == "gemini"


@traceable(name="build_summary_prompt", run_type="prompt")
def _build_summary_prompt(messages_to_summarize: list[Any]) -> list:
    """Build a simple prompt for summarizing old conversation messages."""
    system_message = SystemMessage(
        content=(
            "You are a conversation summarizer. "
            "Concisely summarize the following messages, preserving key decisions and context. "
            "Be factual and avoid opinions."
        )
    )
    messages_text = "\n".join(
        [f"{getattr(message, 'type', 'unknown')}: {message.content}" for message in messages_to_summarize]
    )
    return [system_message, HumanMessage(content=messages_text)]


@traceable(name="summarize_messages", run_type="llm")
def summarize_messages(messages_to_summarize: list[Any]) -> tuple[str, bool]:
    """Summarize a list of conversation messages using the configured provider."""
    settings = get_settings()
    if _requires_provider_api_key(settings) and not settings.gemini_api_key:
        return "[Summarization skipped: missing GEMINI_API_KEY]", False

    prompt_messages = _build_summary_prompt(messages_to_summarize)
    _ = count_tokens_approximately(prompt_messages)

    try:
        model = build_model(settings, temperature=0.2)
        result = model.invoke(prompt_messages)
        text = result_to_text(result)
        return text.strip(), True
    except Exception as exc:  # noqa: BLE001 - robust fallback for runtime failures
        return f"[Summarization error: {exc}]", False
