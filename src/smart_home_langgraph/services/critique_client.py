# ---------------------------------------------------------------------------
# services/critique_client.py
#
# Purpose:
#   Evaluate response quality using a structured critique prompt.
#   Determines if the response needs repair or is acceptable to return.
#   Handles provider/runtime errors gracefully with fallback "accept" decision.
#
#   Also evaluates tool/code execution results for data analysis queries.
# ---------------------------------------------------------------------------
from __future__ import annotations

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.graph.state import AgentState, CritiqueResult
from smart_home_langgraph.services.json_utils import result_to_model
from smart_home_langgraph.services.llm_schema import CritiqueDecision
from smart_home_langgraph.services.model_factory import build_model


def _build_shared_critique_prompt(state: AgentState) -> str:
    """Build the shared critique context used by both prompt variants."""
    return (
        f"User Query:\n{state['user_query']}\n\n"
        f"Detected Intent:\n{state['intent']}\n\n"
        f"Response to Evaluate:\n{state['response']}\n\n"
    )


def _build_code_critique_prompt(state: AgentState) -> str:
    """Build a critique prompt for code/tool execution results."""
    tool_result = state.get("tool_result")
    
    prompt = (
        "You are a strict evaluator for smart-home data analysis responses.\n"
        "Use only evidence from the task, code, and output.\n\n"
        "Checks (all required):\n"
        "  1. Code plausibility: Is the executed code plausible given the task?\n"
        "  2. Execution correctness: Did execution succeed without errors?\n"
        "  3. Numerical plausibility: Are values realistic, internally consistent, and appropriate for the domain?\n"
        "     - Example checks (not exhaustive):\n"
        "       percentages are typically in [0, 100] unless the task defines a different scale.\n"
        "       quantities such as energy, counts, durations, and rates are usually non-negative unless justified.\n"
        "       units, magnitudes, and trends in the response should match the computed output.\n"
        "  4. Logical correctness: Are the calculations and analysis steps logically correct?\n"
        "  5. Response plausibility and consistency: Is the final response plausible and consistent with the task ?\n"
        "  6. Clarity: The execution result may be raw output from a Python function. Ignore and do not complain about formatting issues; evaluate correctness only.\n\n"
        f"{_build_shared_critique_prompt(state)}"
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
        '  "severity": "success", "minor_revision", "major_revision", or "fail"\n'
        '  "repair_hints": suggestions for fixing code or improving response (MUST be "" when passed=true)\n'
        '  "pass_reasons": concise list of why the response passed (2-4 items when passed=true, empty when passed=false)\n'
        '  "critique_status": "completed"\n\n'
        "Set passed=true only for severity=success. Set passed=false for all revision or fail outcomes.\n\n"
        "Severity guide:\n"
        "- success: correct and no repair needed\n"
        "- minor_revision: mostly correct, but needs a small fix or clarification\n"
        "- major_revision: significant issues in reasoning, plausibility, or answer consistency\n"
        "- fail: execution failed or the code/output is fundamentally unusable\n\n"
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
        "  4. Judge the final answer, not raw tool formatting artifacts.\n"
        "  5. Is it concise (max 4-6 points)?\n\n"
        
        f"{_build_shared_critique_prompt(state)}"
        "Return a JSON object with:\n"
        '  "passed": true/false\n'
        '  "issues": list of identified problems (empty if passed)\n'
        '  "severity": "success", "minor_revision", "major_revision", or "fail"\n'
        '  "repair_hints": suggestions for improvement (MUST be "" when passed=true)\n'
        '  "pass_reasons": concise list of why the response passed (2-4 items when passed=true, empty when passed=false)\n'
        '  "critique_status": "completed"\n\n'
        "Set passed=true only for severity=success. Set passed=false for all revision or fail outcomes.\n\n"
        "Respond ONLY with the JSON object, wrapped in ```json ```."
    )


def _to_critique_result(decision: CritiqueDecision) -> CritiqueResult:
    """Convert validated schema object to the workflow state type."""
    return decision.model_dump()


def critique_response(state: AgentState) -> CritiqueResult:
    """
    Evaluate response quality.

    Uses different evaluation criteria for:
    - Standard responses: actionable, safe, consistent with data
    - Tool/code execution: successful execution, correct answer, clear formatting

    Returns:
    CritiqueResult dict with passed, issues, severity, repair_hints,
    pass_reasons, and critique_status.
    """
    settings = get_settings()
    if settings.llm_provider.lower() == "openrouter" and not settings.openrouter_api_key:
        # If no key, accept with explicit skipped status.
        return _to_critique_result(
            CritiqueDecision.skipped_config(
                "Critique step was skipped because OpenRouter API key is missing."
            )
        )

    # Check if this was a tool-based response
    tool_result = state.get("tool_result")
    has_tool_execution = tool_result is not None
    
    # Use appropriate prompt based on response type
    if has_tool_execution:
        prompt = _build_code_critique_prompt(state)
        
        # Quick fail check: if tool execution failed, mark as failed immediately
        if not tool_result.get("success", True):
            return _to_critique_result(
                CritiqueDecision(
                    passed=False,
                    issues=["Code execution failed", tool_result.get("error", "Unknown error")],
                    severity="fail",
                    repair_hints=(
                        f"The previous code raised an error. "
                        f"Fix the following issue: {tool_result.get('error', 'Unknown error')}. "
                        f"Common fixes: check column names, handle NaN values, use correct pandas methods."
                    ),
                    critique_status="completed",
                )
            )
    else:
        prompt = _build_standard_critique_prompt(state)

    try:
        model = build_model(
            settings,
            temperature=0.3,
            structured_output_schema=CritiqueDecision,
        )
        result = model.invoke(prompt)
        return _to_critique_result(result_to_model(result, CritiqueDecision))
    except Exception as exc:  # noqa: BLE001 - robust fallback on any error
        # If critique fails, accept with explicit fallback status.
        return _to_critique_result(CritiqueDecision.fallback_error(str(exc)))
