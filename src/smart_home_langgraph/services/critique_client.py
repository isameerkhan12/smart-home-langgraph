# ---------------------------------------------------------------------------
# services/critique_client.py
#
# Purpose:
#   Evaluate response quality using a structured critique prompt.
#   Determines if the response needs repair or is acceptable to return.
#   Handles Gemini errors gracefully with fallback "accept" decision.
#
#   Also evaluates tool/code execution results for data analysis queries.
# ---------------------------------------------------------------------------
from __future__ import annotations

from langsmith import traceable

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.graph.state import AgentState, CritiqueResult
from smart_home_langgraph.services.json_utils import result_to_model
from smart_home_langgraph.services.llm_schema import CritiqueDecision
from smart_home_langgraph.services.model_factory import build_model


def _build_code_critique_prompt(state: AgentState) -> str:
    """Build a critique prompt for code/tool execution results."""
    tool_result = state.get("tool_result")
    
    prompt = (
        "You are a quality evaluator for smart-home data analysis responses.\n"
        "Evaluate the following response that includes code execution results.\n\n"
        "Evaluation criteria:\n"
        "  1. Did the code execute successfully without errors?\n"
        "  2. Does the result answer the user's question?\n"
        "  3. Is the result formatted clearly and understandably?\n"
        "  4. Are the calculations/analysis logically correct?\n"
        "  5. Does the response provide context for the numbers?\n\n"
        f"User Query:\n{state['user_query']}\n\n"
        f"Detected Intent:\n{state['intent']}\n\n"
        f"Response to Evaluate:\n{state['response']}\n\n"
    )
    
    if tool_result:
        prompt += (
            f"Code Executed:\n```python\n{tool_result.get('tool_input', 'N/A')}\n```\n\n"
            f"Execution Output:\n{tool_result.get('tool_output', 'N/A')}\n\n"
            f"Execution Success: {tool_result.get('success', True)}\n\n"
        )
        
        if not tool_result.get("success"):
            prompt += f"Error Message:\n{tool_result.get('error', 'N/A')}\n\n"
    
    prompt += (
        "Return a JSON object with:\n"
        '  "passed": true/false\n'
        '  "issues": list of identified problems (empty if passed)\n'
        '  "severity": "critical", "medium", or "low"\n'
        '  "repair_hints": suggestions for fixing code or improving response (empty if passed)\n\n'
        "Mark as FAILED (passed=false) if:\n"
        "- Code execution had errors\n"
        "- Result doesn't answer the question\n"
        "- Calculations appear incorrect\n"
        "- Response is unclear or missing context\n\n"
        "Respond ONLY with the JSON object, wrapped in ```json ```."
    )
    
    return prompt


def _build_standard_critique_prompt(state: AgentState) -> str:
    """Build a critique prompt for standard (non-tool) responses."""
    return (
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


@traceable(name="critique_response", run_type="llm")
def critique_response(state: AgentState) -> CritiqueResult:
    """
    Evaluate response quality.

    Uses different evaluation criteria for:
    - Standard responses: actionable, safe, consistent with data
    - Tool/code execution: successful execution, correct answer, clear formatting

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

    # Check if this was a tool-based response
    tool_result = state.get("tool_result")
    has_tool_execution = tool_result is not None
    
    # Use appropriate prompt based on response type
    if has_tool_execution:
        prompt = _build_code_critique_prompt(state)
        
        # Quick fail check: if tool execution failed, mark as failed immediately
        if not tool_result.get("success", True):
            return {
                "passed": False,
                "issues": ["Code execution failed", tool_result.get("error", "Unknown error")],
                "severity": "critical",
                "repair_hints": (
                    f"The previous code raised an error. "
                    f"Fix the following issue: {tool_result.get('error', 'Unknown error')}. "
                    f"Common fixes: check column names, handle NaN values, use correct pandas methods."
                ),
            }
    else:
        prompt = _build_standard_critique_prompt(state)

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
