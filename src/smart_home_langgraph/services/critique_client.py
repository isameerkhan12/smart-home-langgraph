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

from langchain_core.messages import AIMessage, ToolMessage

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.graph.state import AgentState, CritiqueResult
from smart_home_langgraph.services.json_utils import result_to_model
from smart_home_langgraph.services.llm_schema import CritiqueDecision
from smart_home_langgraph.services.model_factory import build_model


def _build_query_context(state: AgentState) -> str:
    """Build the user query + detected intent header shared by every critique prompt."""
    return (
        f"User Query:\n{state['user_query']}\n\n"
        f"Detected Intent:\n{state['intent']}\n\n"
    )


def _build_shared_critique_prompt(state: AgentState) -> str:
    """Build the shared critique context used by response-evaluating prompt variants."""
    memory_usage_mode = state.get("memory_usage_mode", "none")
    memory_context = state.get("memory_context", "")

    shared = _build_query_context(state) + f"Memory Usage Mode: {memory_usage_mode}\n\n"
    if memory_context and "No relevant" not in memory_context and "No long-term" not in memory_context:
        shared += f"Retrieved Memory Context:\n{memory_context}\n\n"
    shared += f"Response to Evaluate:\n{state['response']}\n\n"
    return shared


def _format_tool_execution_context(state: AgentState) -> str:
    """Format all tool calls and results for critique prompts."""
    messages = state.get("messages", [])
    context: list[str] = []

    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage):
            continue

        for tool_call in getattr(message, "tool_calls", []):
            tool_call_id = tool_call.get("id")
            tool_name = tool_call.get("name", "unknown_tool")
            tool_args = tool_call.get("args", {})
            output = "N/A"
            for result in messages[index + 1:]:
                if isinstance(result, ToolMessage) and result.tool_call_id == tool_call_id:
                    output = result.content if isinstance(result.content, str) else str(result.content)
                    break

            if tool_name == "python_repl":
                code = tool_args.get("query") or tool_args.get("code", "")
                context.append(
                    f"Tool: {tool_name}\nCode Executed:\n```python\n{code}\n```\n\n"
                    f"Execution Output:\n{output}\n\n"
                )
            else:
                context.append(f"Tool: {tool_name}\nTool Output:\n{output}\n\n")

    return "".join(context)


def _build_code_critique_prompt(state: AgentState) -> str:
    """Build a critique prompt for code/tool execution results."""
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
    
    prompt += _format_tool_execution_context(state)
    
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


def _build_partial_critique_prompt(state: AgentState) -> str:
    """Build a critique prompt for responses using memory and tool output."""
    prompt = (
        "You are a strict evaluator for smart-home data analysis responses.\n"
        "Context: The retrieved memory was produced by a prior agent run and has already been checked by a critique node.\n"
        "This response may combine checked memory with fresh tool execution.\n"
        "Use memory to evaluate memory-backed claims, and code/output to evaluate tool-backed claims.\n\n"
        "Checks (all required):\n"
        "  1. Memory grounding: Memory-backed values or claims must appear explicitly in the retrieved memory.\n"
        "  2. Memory numerical accuracy: Memory-backed numbers must match memory exactly.\n"
        "  3. Code plausibility: Executed code must be plausible for the newly computed part of the task.\n"
        "  4. Execution correctness: Tool execution must succeed without errors.\n"
        "  5. Tool numerical plausibility: Computed values must be realistic, internally consistent, and match the output.\n"
        "  6. Source consistency: If memory and tool output describe the same fact, they must not contradict.\n"
        "  7. Logical correctness: Calculations, units, reasoning, and combined conclusions must be coherent.\n"
        "  8. Task alignment and clarity: The response must answer the original question; ignore raw tool formatting artifacts.\n\n"
        "If the retrieved memory is irrelevant, insufficient for the memory-backed claim, misused, or contradicted by tool output, "
        "set force_full_recompute=true and instruct repair to ignore memory and recompute the full answer from tools.\n\n"
        f"{_build_shared_critique_prompt(state)}"
    )

    prompt += _format_tool_execution_context(state)

    prompt += (
        "Return a JSON object with:\n"
        '  "passed": true/false\n'
        '  "issues": list of identified problems (empty if passed)\n'
        '  "severity": "success", "minor_revision", "major_revision", or "fail"\n'
        '  "repair_hints": suggestions for fixing code or improving response (MUST be "" when passed=true)\n'
        '  "pass_reasons": concise list of why the response passed (2-4 items when passed=true, empty when passed=false)\n'
        '  "force_full_recompute": true only when repair must ignore memory and recompute everything from tools; otherwise false\n'
        '  "critique_status": "completed"\n\n'
        "Set passed=true only for severity=success. Set passed=false for all revision or fail outcomes.\n\n"
        "Severity guide:\n"
        "- success: memory-backed and computed values are correct and no repair is needed\n"
        "- minor_revision: mostly correct, but needs a small fix or clarification\n"
        "- major_revision: significant issues in memory use, computation, reasoning, or answer consistency\n"
        "- fail: execution failed or the response is fundamentally unsupported\n\n"
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


def _build_memory_critique_prompt(state: AgentState) -> str:
    """Build a critique prompt for memory-only responses."""
    return (
        "You are a strict evaluator for smart-home energy data analysis responses.\n\n"
        "Context: The retrieved memory was produced by a prior agent run and has already been checked by a critique node.\n"
        "Checks (all required):\n"
        "  1. Grounding — every value or claim in the response must appear explicitly in the memory.\n"
        "  2. Numerical accuracy — numbers must match memory exactly.\n"
        "  3. Logical integrity — reasoning, units, and conclusions must be coherent with the memory facts.\n"
        "  4. Task alignment — the response must address the original question.\n\n"
        f"{_build_shared_critique_prompt(state)}"
        "Return a JSON object:\n"
        '  "passed": true | false\n'
        '  "issues": list of problems (empty if passed)\n'
        '  "severity": "success" | "minor_revision" | "major_revision" | "fail"\n'
        '  "repair_hints": fix instructions ("" when passed=true)\n'
        '  "pass_reasons": 2–4 verification statements (empty when passed=false).\n'
        '    State what was confirmed and that it matched — e.g. "reported value 0.0036 kWh matches memory exactly".\n'
        '    Do NOT write "as mentioned in memory" or "according to the context".\n'
        '  "force_full_recompute": false\n'
        '  "critique_status": "completed"\n\n'
        "Rules:\n"
        "- passed=true only when severity=success.\n"
        "- passed=false for minor_revision, major_revision, and fail.\n\n"
        "Severity guide:\n"
        "  success        — all values grounded and numerically accurate\n"
        "  minor_revision — mostly correct but one detail is missing or imprecise\n"
        "  major_revision — a key value is absent or inconsistent with memory\n"
        "  fail           — response fabricates content or directly contradicts memory\n\n"
        "Respond ONLY with the JSON object wrapped in ```json ```.\n"
    )


def _build_plan_critique_prompt(state: AgentState) -> str:
    """Build a critique prompt for a proposed step-by-step plan, before execution."""
    return (
        "You are a strict reviewer for a proposed data-analysis plan.\n"
        "No code has executed yet. Judge only whether the plan's approach is sound.\n\n"
        "Checks (all required):\n"
        "  1. Does the plan address every part of the user's question?\n"
        "  2. Are the steps logically ordered, with later steps correctly depending on earlier results?\n"
        "  3. Does the plan avoid unnecessary or irrelevant steps?\n"
        "  4. Is the plan specific enough to be executed as code (not vague)?\n\n"
        f"{_build_query_context(state)}"
        f"Proposed Plan:\n{state.get('plan', '')}\n\n"
        "Return a JSON object with:\n"
        '  "passed": true/false\n'
        '  "issues": list of identified problems (empty if passed)\n'
        '  "severity": "success", "minor_revision", "major_revision", or "fail"\n'
        '  "repair_hints": suggestions for fixing the plan (MUST be "" when passed=true)\n'
        '  "pass_reasons": concise list of why the plan passed (2-4 items when passed=true, empty when passed=false)\n'
        '  "critique_status": "completed"\n\n'
        "Set passed=true only for severity=success. Set passed=false for all revision or fail outcomes.\n\n"
        "Respond ONLY with the JSON object, wrapped in ```json ```."
    )


def _build_step_critique_prompt(state: AgentState) -> str:
    """Build a critique prompt for the most recently executed step of a multi-step plan."""
    prompt = (
        "You are a strict reviewer for one intermediate step of a multi-step data-analysis plan.\n"
        "Judge only the most recent executed step against the plan. Do not require the final answer yet.\n\n"
        "Checks (all required):\n"
        "  1. Does the most recent code correspond to the next unfinished step of the plan?\n"
        "  2. Did execution succeed without errors?\n"
        "  3. Are the computed values realistic and internally consistent for this step?\n"
        "  4. Is the step's result usable as input for the plan's remaining steps?\n\n"
        f"{_build_query_context(state)}"
        f"Plan:\n{state.get('plan', '')}\n\n"
    )

    prompt += _format_tool_execution_context(state)

    prompt += (
        "Return a JSON object with:\n"
        '  "passed": true/false\n'
        '  "issues": list of identified problems (empty if passed)\n'
        '  "severity": "success", "minor_revision", "major_revision", or "fail"\n'
        '  "repair_hints": suggestions for fixing this step (MUST be "" when passed=true)\n'
        '  "pass_reasons": concise list of why the step passed (2-4 items when passed=true, empty when passed=false)\n'
        '  "critique_status": "completed"\n\n'
        "Set passed=true only for severity=success. Set passed=false for all revision or fail outcomes.\n\n"
        "Respond ONLY with the JSON object, wrapped in ```json ```."
    )

    return prompt


def _to_critique_result(decision: CritiqueDecision) -> CritiqueResult:
    """Convert validated schema object to the workflow state type."""
    return decision.model_dump()


def _skip_if_no_key(settings, reason: str) -> CritiqueResult | None:
    """Return a skipped-config result when the provider API key is missing, else None."""
    if settings.llm_provider.lower() == "openrouter" and not settings.openrouter_api_key:
        return _to_critique_result(CritiqueDecision.skipped_config(reason))
    return None


def _execution_failure_result(tool_result: dict) -> CritiqueResult:
    """Build a fail result for a tool execution that raised an error."""
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


def _run_critique(settings, prompt: str) -> CritiqueResult:
    """Invoke the critique model on a prompt, with a fallback-accept result on error."""
    try:
        model = build_model(
            settings,
            temperature=0.3,
            structured_output_schema=CritiqueDecision,
        )
        result = model.invoke(prompt)
        return _to_critique_result(result_to_model(result, CritiqueDecision))
    except Exception as exc:  # noqa: BLE001 - robust fallback on any error
        return _to_critique_result(CritiqueDecision.fallback_error(str(exc)))


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
    skipped = _skip_if_no_key(settings, "Critique step was skipped because OpenRouter API key is missing.")
    if skipped is not None:
        return skipped

    tool_result = state.get("tool_result")
    memory_usage_mode = state.get("memory_usage_mode", "none")

    if memory_usage_mode != "memory_only" and tool_result and not tool_result.get("success", True):
        return _execution_failure_result(tool_result)

    # Route to prompt matching the evidence available for this response.
    if memory_usage_mode == "memory_only":
        prompt = _build_memory_critique_prompt(state)
    elif memory_usage_mode == "partial":
        prompt = _build_partial_critique_prompt(state)
    else:
        prompt = _build_code_critique_prompt(state)

    return _run_critique(settings, prompt)


def critique_plan(state: AgentState) -> CritiqueResult:
    """
    Evaluate a proposed step-by-step plan before any code executes.

    Returns:
    CritiqueResult dict with passed, issues, severity, repair_hints,
    pass_reasons, and critique_status.
    """
    settings = get_settings()
    skipped = _skip_if_no_key(settings, "Plan critique was skipped because OpenRouter API key is missing.")
    if skipped is not None:
        return skipped

    return _run_critique(settings, _build_plan_critique_prompt(state))


def critique_step(state: AgentState) -> CritiqueResult:
    """
    Evaluate the most recently executed step of a multi-step plan.

    Returns:
    CritiqueResult dict with passed, issues, severity, repair_hints,
    pass_reasons, and critique_status.
    """
    settings = get_settings()
    skipped = _skip_if_no_key(settings, "Step critique was skipped because OpenRouter API key is missing.")
    if skipped is not None:
        return skipped

    tool_result = state.get("tool_result")
    if tool_result and not tool_result.get("success", True):
        return _execution_failure_result(tool_result)

    return _run_critique(settings, _build_step_critique_prompt(state))
