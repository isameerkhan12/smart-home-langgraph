# ---------------------------------------------------------------------------
# services/critique_client.py
#
# Purpose:
#   Evaluate response quality using a structured critique prompt.
#   Determines if the response needs repair or is acceptable to return.
#   Handles Gemini errors gracefully with fallback "accept" decision.
# ---------------------------------------------------------------------------
from __future__ import annotations

from langsmith import traceable

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.graph.state import AgentState, CritiqueResult
from smart_home_langgraph.services.json_utils import result_to_model
from smart_home_langgraph.services.llm_schema import CritiqueDecision
from smart_home_langgraph.services.model_factory import build_model


@traceable(name="critique_response", run_type="llm")
def critique_response(state: AgentState) -> CritiqueResult:
    """
    Evaluate response quality.

    Asks Gemini: "Does this response meet our quality criteria?"
    Returns structured CritiqueResult with pass/fail and repair hints.

    Returns:
      CritiqueResult dict with passed, issues, severity, repair_hints.
    """
    settings = get_settings()
    if settings.llm_provider.lower() == "gemini" and not settings.gemini_api_key:
        # If no key, assume response is acceptable (safe fallback).
        return {
            "passed": True,
            "issues": [],
            "severity": "low",
            "repair_hints": "",
        }

    prompt = (
        "You are a quality evaluator for smart-home assistant responses.\n"
        "Evaluate the following response against these criteria:\n"
        "  1. Is it actionable and concrete?\n"
        "  2. Does it prioritize safety?\n"
        "  3. Does it avoid contradictions with sensor data?\n"
        "  4. Is it concise (max 4-6 points)?\n\n"
        f"User Query:\n{state['user_query']}\n\n"
        f"Detected Intent:\n{state['intent']}\n\n"
        f"Response to Evaluate:\n{state['response']}\n\n"
        "Return a JSON object with:\n"
        '  "passed": true/false\n'
        '  "issues": list of identified problems (empty if passed)\n'
        '  "severity": "critical", "medium", or "low"\n'
        '  "repair_hints": suggestions for improvement (empty if passed)\n\n'
        "Respond ONLY with the JSON object, wrapped in ```json ```."
    )

    try:
        model = build_model(
            settings,
            temperature=0.3,
            structured_output_schema=CritiqueDecision,
        )
        result = model.invoke(prompt)
        return result_to_model(result, CritiqueDecision).model_dump()
    except Exception as exc:  # noqa: BLE001 - robust fallback on any error
        # If critique fails, assume response is acceptable (safe fallback).
        return {
            "passed": True,
            "issues": [],
            "severity": "low",
            "repair_hints": f"Critique unavailable: {exc}",
        }
