# ---------------------------------------------------------------------------
# services/memory_extractor.py
# Purpose: Native LTM extraction using the configured LLM provider.
# ---------------------------------------------------------------------------
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.memory.ltm_schema import MemoryDecision
from smart_home_langgraph.services.json_utils import result_to_model
from smart_home_langgraph.services.model_factory import build_model


# MEMORY_EXTRACTION_PROMPT = """You extract long-term memories from a smart-home assistant turn.

# Rules:
# - Return should_write=false if there is no durable memory worth storing.
# - Output short atomic memories only.
# - Use memory_type from: preference, recipe, mistake, general.
# - Set is_new=false if a memory already exists with the same meaning.
# - Avoid duplicates and avoid speculative facts.
# """

MEMORY_EXTRACTION_PROMPT = """You extract long-term memories from a smart-home assistant turn.

Memory Types:
- "recipe": Code patterns, calculation methods, or reusable approaches that worked successfully.
  ALWAYS store as recipe if the response contains working Python code or a successful calculation method.
  Example: "To calculate average energy: use df['Energy_Consumption_kWh'].mean()"
- "mistake": Errors, failed approaches, or wrong column names to avoid in future.
  Store if the response mentions an error, correction, or failed attempt.
  Example: "Avoid using 'Voltage_Reading' column - correct name is 'Line_Voltage'"
- "preference": User preferences about format, units, or behavior.
- "general": Other useful facts about the smart home data or domain.

Rules:
- Return should_write=true if ANY of these are present:
  1. Working Python code that produced a result
  2. A calculation method or formula
  3. An error that was corrected
  4. A user preference
- Output short atomic memories that can be reused for similar tasks.
- Set is_new=false ONLY if existing_memories already contains the exact same approach.
- Be generous about storing recipes - they help answer similar future questions faster.
"""

OLLAMA_FALLBACK_JSON_INSTRUCTION = (
    "\nReturn valid JSON matching this schema: "
    '{"should_write": true/false, "memories": [{"text": "...", "memory_type": "preference|recipe|mistake|general", "is_new": true/false, "metadata": {}}]}'
)


def _build_system_prompt(provider: str) -> str:
    """Build extraction prompt and include Ollama JSON hint only when needed."""
    if provider == "ollama":
        return MEMORY_EXTRACTION_PROMPT + OLLAMA_FALLBACK_JSON_INSTRUCTION
    return MEMORY_EXTRACTION_PROMPT


def _invoke_ollama_fallback(settings, user_payload: str) -> MemoryDecision | None:
    """Fallback for Ollama models that ignore structured output."""
    try:
        from langchain_ollama import ChatOllama

        fallback_model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        fallback_result = fallback_model.invoke(
            [
                SystemMessage(content=MEMORY_EXTRACTION_PROMPT + OLLAMA_FALLBACK_JSON_INSTRUCTION),
                HumanMessage(content=user_payload),
            ]
        )
        return result_to_model(fallback_result, MemoryDecision)
    except Exception:
        return None


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
    provider = settings.llm_provider.lower()
    if provider == "openrouter" and not settings.openrouter_api_key:
        return MemoryDecision(should_write=False, memories=[])

    model = build_model(
        settings,
        temperature=0,
        structured_output_schema=MemoryDecision,
    )

    user_payload = (
        f"Intent: {intent}\n"
        f"Critique passed: {critique_passed}\n"
        f"Critique issues: {', '.join(critique_issues) if critique_issues else '(none)'}\n\n"
        f"User query:\n{user_query}\n\n"
        f"Assistant response:\n{assistant_response}\n\n"
        f"Existing memories:\n{existing_memories_text}\n"
    )

    system_prompt = _build_system_prompt(provider)

    try:
        result = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_payload),
            ]
        )
        return result_to_model(result, MemoryDecision)
    except Exception:
        pass

    if provider != "ollama":
        return MemoryDecision(should_write=False, memories=[])

    fallback_decision = _invoke_ollama_fallback(settings, user_payload)
    if fallback_decision is not None:
        return fallback_decision

    return MemoryDecision(should_write=False, memories=[])
