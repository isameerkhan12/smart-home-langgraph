# ---------------------------------------------------------------------------
# services/memory_extractor.py
# Purpose: Native LTM extraction using Gemini structured output.
# ---------------------------------------------------------------------------
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.memory.ltm_schema import MemoryDecision


MEMORY_EXTRACTION_PROMPT = """You extract long-term memories from a smart-home assistant turn.

Rules:
- Return should_write=false if there is no durable memory worth storing.
- Output short atomic memories only.
- Use memory_type from: preference, recipe, mistake, general.
- Set is_new=false if a memory already exists with the same meaning.
- Avoid duplicates and avoid speculative facts.
"""


def extract_structured_memories(
    *,
    user_query: str,
    assistant_response: str,
    intent: str,
    critique_passed: bool,
    critique_issues: list[str],
    existing_memories_text: str,
) -> MemoryDecision:
    """Extract typed memory candidates from the current interaction."""
    settings = get_settings()
    if not settings.gemini_api_key:
        return MemoryDecision(should_write=False, memories=[])

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0,
    ).with_structured_output(MemoryDecision)

    user_payload = (
        f"Intent: {intent}\n"
        f"Critique passed: {critique_passed}\n"
        f"Critique issues: {', '.join(critique_issues) if critique_issues else '(none)'}\n\n"
        f"User query:\n{user_query}\n\n"
        f"Assistant response:\n{assistant_response}\n\n"
        f"Existing memories:\n{existing_memories_text}\n"
    )

    try:
        return model.invoke(
            [
                SystemMessage(content=MEMORY_EXTRACTION_PROMPT),
                HumanMessage(content=user_payload),
            ]
        )
    except Exception:
        return MemoryDecision(should_write=False, memories=[])
