
# Dependency injection:
#   build_workflow() accepts optional callable overrides for the LLM generator
#   and critique evaluator. Tests pass fake callables so they run fast and
#   never touch external APIs.
# ---------------------------------------------------------------------------------
import uuid
import json
import re
from typing import Callable, Literal, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore

from smart_home_langgraph.evaluation.metrics import EpisodeRecord
from smart_home_langgraph.graph.state import AgentState, CritiqueResult, ToolExecutionResult
from smart_home_langgraph.memory.ltm_schema import MemoryType
from smart_home_langgraph.services.critique_client import critique_response
from smart_home_langgraph.services.planner_client import evaluate_memory_sufficiency
from smart_home_langgraph.services.response_client import generate_response, generate_tool_enabled_response
from smart_home_langgraph.services.summary_client import summarize_messages
from smart_home_langgraph.services.memory_extractor import extract_structured_memories
from smart_home_langgraph.tools.python_executor import get_smart_home_tools

# Type aliases for injectable callables (used for dependency injection in tests).
ResponseGenerator = Callable[[AgentState], tuple[str, bool]]
ResponseGeneratorWithTools = Callable[[AgentState, Sequence[BaseTool]], AIMessage]
CritiqueGenerator = Callable[[AgentState], CritiqueResult]

# Experiment toggle: when False, keep raw tool output (no formatting).
TOOL_OUTPUT_FORMATTING_ENABLED = False


def render_tool_output(tool_output: object, indent: int = 0) -> str:
    """Render tool output as readable text for the final response."""
    prefix = "  " * indent

    if isinstance(tool_output, str):
        stripped = tool_output.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                tool_output = json.loads(stripped)
            except Exception:
                return tool_output
        else:
            return tool_output

    if isinstance(tool_output, dict):
        lines: list[str] = []
        for key, value in tool_output.items():
            if isinstance(value, (dict, list, tuple, set)):
                lines.append(f"{prefix}- {key}:")
                lines.append(render_tool_output(value, indent + 1))
            else:
                lines.append(f"{prefix}- {key}: {value}")
        return "\n".join(lines)

    if isinstance(tool_output, (list, tuple, set)):
        lines: list[str] = []
        for value in tool_output:
            if isinstance(value, (dict, list, tuple, set)):
                lines.append(f"{prefix}-")
                lines.append(render_tool_output(value, indent + 1))
            else:
                lines.append(f"{prefix}- {value}")
        return "\n".join(lines)

    return f"{prefix}{tool_output}"


def initial_state(
    query: str,
    max_repairs: int = 2,
    messages: list[BaseMessage] | None = None,
) -> AgentState:
    """
    Return the default initial state for starting a new agent run.

    Call this in main.py and in tests to avoid repeating the boilerplate dict.
    """
    return {
        "user_query": query,
        "messages": list(messages or []),
        "summary": "",
        "intent": "",
        "sensor_context": "",
        "memory_context": "",
        "error_memory_context": "",
        "error_signature": "",
        # Memory evaluator decision
        "use_memory_only": False,
        "planner_reason": "",
        # Response
        "response": "",
        "used_live_llm": False,
        # Tool execution state
        "tool_result": None,
        "generated_code": "",
        "tool_execution_count": 0,
        # Critique and repair
        "critique_result": {
            "passed": False,
            "issues": [],
            "severity": "minor_revision",
            "repair_hints": "",
            "pass_reasons": [],
            "critique_status": "not_run",
        },
        "repair_count": 0,
        "max_repairs": max_repairs,
        "episode_record": {
            "episode_id": 1,
            "task_class": "",
            "critique_passed_first_try": False,
            "repeated_known_mistake": False,
            "preferences_respected": False,
            "used_existing_recipe": False,
        },
        "memory_written_count": 0,
        "error_memory_written_count": 0,
    }


def build_workflow(
    response_generator: ResponseGenerator | None = None,
    critique_generator: CritiqueGenerator | None = None,
    checkpointer: SqliteSaver | None = None,
    # Native LangGraph long-term store (e.g., PostgresStore).
    store: BaseStore | None = None,
    max_repairs: int = 2,
    tools: Sequence[BaseTool] | None = None,
):
    """
    Build and compile the smart-home agent graph.

    Parameters (all optional — defaults wire up the live system):
      response_generator  Callable that produces (response_text, used_live_llm).
                          Default: generate_response (real provider call).
      critique_generator  Callable that returns a CritiqueResult dict.
                          Default: critique_response (real Gemini critique).
      checkpointer        LangGraph checkpointer for persistent chat state.
                          Default: no checkpointing.
      store               LangGraph BaseStore for long-term memory.
                          Default: no persistent store.
      max_repairs         Maximum repair loop iterations. Default: 2.
      tools               List of tools for code execution. Default: smart home tools.
    """
    response_gen = response_generator or generate_response
    critique_gen = critique_generator or critique_response
    
    # Initialize tools for data analysis
    available_tools = list(tools) if tools else get_smart_home_tools()
    tool_node = ToolNode(tools=available_tools)

    # Memory namespace layout: (app, entity, user_id, collection)
    namespace_root = ("smart_home", "users")

    def memory_namespace(config: RunnableConfig | None) -> tuple[str, str, str, str]:
        user_id = (config or {}).get("configurable", {}).get("user_id", "default_user")
        return (*namespace_root, user_id, "typed_memories")

    def format_memories_for_prompt(items: list) -> str:
        if not items:
            return "No relevant long-term memories found."
        lines: list[str] = []
        for item in items:
            content = item.value.get("content", "").strip()
            memory_type = item.value.get("memory_type", MemoryType.GENERAL.value)
            if content:
                lines.append(f"- [{memory_type}] {content}")
        return "\n".join(lines) if lines else "No relevant long-term memories found."

    def build_error_signature(state: AgentState) -> str:
        """Create a normalized error signature for retrieval and dedupe."""
        tool_result = state.get("tool_result") or {}
        error_text = str(tool_result.get("error", "unknown_error"))
        normalized = re.sub(r"\d+", "<num>", error_text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        tool_name = str(tool_result.get("tool_name", "unknown_tool"))
        return f"{state['intent']}|{tool_name}|{normalized}"

    # ------------------------------------------------------------------
    # Node definitions (closures capture sim, retriever, response_gen, etc.)
    # ------------------------------------------------------------------

    def detect_intent(state: AgentState) -> AgentState:
        """Classify user query into a task class via keyword matching."""
        query = state["user_query"].lower()
        
        if any(word in query for word in ["energy", "power", "bill", "save"]):
            intent = "energy_optimization"
        elif any(word in query for word in ["temperature", "comfort", "hot", "cold"]):
            intent = "comfort_optimization"
        elif any(word in query for word in ["anomaly", "strange", "unusual", "fault"]):
            intent = "anomaly_explanation"
        else:
            intent = "general_question"
        
        return {**state, "intent": intent}

    def retrieve_context(state: AgentState,config: RunnableConfig | None = None,*,store: BaseStore | None = None,) -> AgentState:
        """
        Build memory context injected into the generation prompt:
          - memory_context  — mistakes + recipes + preferences from memory stores
        """
        if store is None:
            memory_context = "No long-term memory store configured."
        else:
            memories = store.search(memory_namespace(config), query=state["user_query"], limit=8)
            memory_context = format_memories_for_prompt(memories)
        return {**state, "memory_context": memory_context}

    def memory_evaluator_node(state: AgentState) -> AgentState:
        """
        Decide whether memory is sufficient to answer the question.
        
        Sets use_memory_only=True to skip tool binding in generate_response_node.
        """
        decision = evaluate_memory_sufficiency(state)
        return {
            **state,
            "use_memory_only": decision.use_memory,
            "planner_reason": decision.reason,
        }

    def retrieve_error_memory_node(state: AgentState,config: RunnableConfig | None = None,*,store: BaseStore | None = None,) -> AgentState:
        """Retrieve error-specific memories using failed tool execution context."""
        tool_result = state.get("tool_result")
        if tool_result is None or tool_result.get("success", True):
            return {**state, "error_memory_context": "No error-specific memories found.", "error_signature": ""}

        signature = build_error_signature(state)
        code_head = (tool_result.get("tool_input", "") or "").splitlines()
        first_code_line = code_head[0] if code_head else ""
        retrieval_query = (
            f"{state['intent']} "
            f"{tool_result.get('tool_name', 'tool')} "
            f"{tool_result.get('error', '')} "
            f"{first_code_line}"
        ).strip()

        if store is None:
            return {
                **state,
                "error_memory_context": "No long-term memory store configured.",
                "error_signature": signature,
            }

        memories = store.search(memory_namespace(config), query=retrieval_query, limit=5)
        return {
            **state,
            "error_memory_context": format_memories_for_prompt(memories),
            "error_signature": signature,
        }

    def generate_response_node(state: AgentState) -> AgentState:
        """
        Generate response using the configured LLM.
        
        If memory evaluator decided memory is sufficient (use_memory_only=True),
        generates response WITHOUT tools bound — LLM cannot call tools.
        Otherwise, binds tools and lets LLM decide autonomously.
        """
        # Memory evaluator decided memory is sufficient — generate without tools
        if state.get("use_memory_only", False):
            response_text, used_live_llm = response_gen(state)
            return {**state, "response": response_text, "used_live_llm": used_live_llm}

        # Tools available and memory evaluator says computation needed — bind tools
        if available_tools:
            llm_response = generate_tool_enabled_response(state, available_tools)
            
            if llm_response.tool_calls:
                # LLM chose to use a tool
                tool_args = llm_response.tool_calls[0].get("args", {})
                code = tool_args.get("query") or tool_args.get("code", "")
                return {
                    **state,
                    "messages": [llm_response],
                    "used_live_llm": True,
                    "generated_code": code,
                }
            else:
                # LLM chose to respond directly (no tools needed)
                return {
                    **state,
                    "messages": [llm_response],
                    "response": llm_response.content or "",
                    "used_live_llm": True,
                }
        else:
            # No tools available - standard generation
            response_text, used_live_llm = response_gen(state)
            return {**state, "response": response_text, "used_live_llm": used_live_llm}

    def process_tool_result(state: AgentState) -> AgentState:
        """
        Process results from any tool execution and format the response.
        
        Handles different tool types via format templates.
        """
        messages = state.get("messages", [])
        
        # Find the last tool message
        tool_output = ""
        tool_name = ""
        
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_output = msg.content if isinstance(msg.content, str) else str(msg.content)
                tool_name = getattr(msg, "name", "unknown_tool")
                break
        
        # Get generated code from state (set by generate_response_node)
        code = state.get("generated_code", "")
        
        # Detect errors (common patterns across tools)
        error_patterns = ["Error:", "Exception", "Traceback", "failed", "error:"]
        success = not any(pattern.lower() in tool_output.lower() for pattern in error_patterns)
        error = tool_output if not success else ""
        
        # Build the tool execution result
        tool_result: ToolExecutionResult = {
            "tool_name": tool_name,
            "tool_input": code,
            "tool_output": tool_output,
            "success": success,
            "error": error,
        }
        
        # Toggle formatter on/off for critique prompt experiments.
        formatted_output = (
            render_tool_output(tool_output)
            if TOOL_OUTPUT_FORMATTING_ENABLED
            else tool_output
        )

        # Build response based on tool type
        if tool_name == "python_repl":
            if success:
                response = (
                    f"Based on my analysis of the smart home data:\n\n"
                    f"**Result:**\n{formatted_output}\n\n"
                    f"This was calculated by executing:\n```python\n{code}\n```"
                )
            else:
                response = (
                    f"I attempted to analyze the data but encountered an error:\n\n"
                    f"**Result:**\n{formatted_output}\n\n"
                    f"**Error:**\n{error}\n\n"
                    f"Let me try a different approach."
                )
        else:
            if success:
                response = f"**{tool_name} Result:**\n{formatted_output}"
            else:
                response = f"**{tool_name} Error:**\n{formatted_output}"
        
        return {
            **state,
            "response": response,
            "tool_result": tool_result,
            "tool_execution_count": state.get("tool_execution_count", 0) + 1,
        }

    def record_turn(state: AgentState) -> AgentState:
        """Append the completed user/assistant turn to chat history."""
        return {
            **state,
            "messages": [
                HumanMessage(content=state["user_query"]),
                AIMessage(content=state["response"]),
            ],
        }

    def should_summarize(state: AgentState) -> str:
        """Route to summarize_node if conversation > 6 messages, else continue to generate."""
        if len(state["messages"]) > 6:
            return "summarize"
        return "continue"

    def summarize_node(state: AgentState) -> AgentState:
        """
        Summarize older messages to keep context window manageable.
        Keeps last 2 messages, summarizes the rest, removes originals using RemoveMessage.
        """
        history = state["messages"]
        
        if len(history) > 6:
            # Keep last 2 messages, summarize everything else
            msgs_to_summarize = history[:-2]
            last_2_msgs = history[-2:]
            
            summary_content, _ = summarize_messages(msgs_to_summarize)
            
            # Create summary message and remove old ones
            summary_msg = SystemMessage(
                content=f"[CONTEXT SUMMARY]\n{summary_content}"
            )
            
            # Build RemoveMessage for each summarized message that has an ID
            remove_messages = [
                RemoveMessage(id=m.id) 
                for m in msgs_to_summarize 
                if hasattr(m, "id") and m.id is not None
            ]
            
            return {
                **state,
                "messages": remove_messages + [summary_msg] + last_2_msgs,
                "summary": summary_content,
            }
        
        return state

    def critique_node(state: AgentState) -> AgentState:
        """Evaluate response quality; store structured result for routing."""
        result = critique_gen(state)
        return {**state, "critique_result": result}

    def repair_node(state: AgentState) -> AgentState:
        """
        Regenerate the response using critique hints.
        
        For tool-based queries, this includes code repair hints.
        Appends repair instructions to the original query so the generator
        knows what to fix. Increments repair_count for loop termination.
        
        Respects use_memory_only flag from memory evaluator — if True, repairs without tools.
        """
        tool_result = state.get("tool_result")
        
        # Build repair hints based on whether this was a tool execution
        if tool_result and not tool_result.get("success"):
            hints = (
                f"\n\nPrevious code execution failed.\n"
                f"Error: {tool_result.get('error', 'Unknown error')}\n"
                f"Failed code:\n```python\n{tool_result.get('tool_input', '')}\n```\n"
                f"Related past fixes/mistakes:\n{state.get('error_memory_context', 'No error-specific memories found.')}\n"
                f"Critique hints: {state['critique_result']['repair_hints']}\n"
                f"Please generate corrected Python code to answer the question."
            )
        else:
            hints = (
                f"\n\nPrevious response had issues. Repair hints: {state['critique_result']['repair_hints']}\n"
                f"Issues identified: {', '.join(state['critique_result']['issues'])}\n"
                "Please provide an improved response."
            )
        
        repair_state = {**state, "user_query": state["user_query"] + hints}
        
        # Respect memory evaluator decision — if memory-only, repair without tools
        if state.get("use_memory_only", False):
            response_text, used_live_llm = response_gen(repair_state)
            return {
                **state,
                "response": response_text,
                "used_live_llm": used_live_llm or state["used_live_llm"],
                "repair_count": state["repair_count"] + 1,
            }
        
        # Tools allowed — regenerate with tools bound
        if available_tools:
            llm_response = generate_tool_enabled_response(repair_state, available_tools)
            
            if llm_response.tool_calls:
                tool_args = llm_response.tool_calls[0].get("args", {})
                code = tool_args.get("query") or tool_args.get("code", "")
                return {
                    **state,
                    "messages": [llm_response],
                    "used_live_llm": True,
                    "generated_code": code,
                    "repair_count": state["repair_count"] + 1,
                }
            else:
                return {
                    **state,
                    "messages": [llm_response],
                    "response": llm_response.content or "",
                    "used_live_llm": True,
                    "repair_count": state["repair_count"] + 1,
                }
        else:
            response_text, used_live_llm = response_gen(repair_state)
            return {
                **state,
                "response": response_text,
                "used_live_llm": used_live_llm or state["used_live_llm"],
                "repair_count": state["repair_count"] + 1,
            }

    def memory_writer_node(state: AgentState,config: RunnableConfig | None = None,*,store: BaseStore | None = None,) -> AgentState:
        """
        Write typed long-term memories to the LangGraph store.

        Memory extraction is delegated to a structured-output LLM call and
        duplicate suppression is performed via the returned `is_new` flags.
        """
        memories_written = 0

        if store is not None:
            ns = memory_namespace(config)
            # Provide existing memory text so extractor can suppress semantic duplicates.
            existing_items = store.search(ns, limit=30)
            existing_memories_text = "\n".join(
                item.value.get("content", "") for item in existing_items if item.value.get("content")
            ) or "(empty)"

            # Reuse the same structured extractor path used across the app.
            decision = extract_structured_memories(
                user_query=state["user_query"],
                assistant_response=state["response"],
                intent=state["intent"],
                critique_passed=state["critique_result"]["passed"],
                critique_issues=state["critique_result"]["issues"],
                existing_memories_text=existing_memories_text,
            )

            if decision.should_write:
                for memory in decision.memories:
                    # Skip duplicates/empty memories exactly as flagged by extractor.
                    if not memory.is_new or not memory.text.strip():
                        continue
                    store.put(
                        ns,
                        str(uuid.uuid4()),
                        {
                            "content": memory.text.strip(),
                            "memory_type": memory.memory_type.value,
                            "intent": state["intent"],
                            "metadata": memory.metadata,
                        },
                    )
                    memories_written += 1

        episode = EpisodeRecord(
            episode_id=1,
            task_class=state["intent"],
            critique_passed_first_try=state["repair_count"] == 0 and state["critique_result"]["passed"],
            repeated_known_mistake=False,
            preferences_respected=False,
            used_existing_recipe="[recipe]" in state["memory_context"].lower(),
        )
        return {
            **state,
            "episode_record": episode,
            "memory_written_count": memories_written,
        }

    def error_memory_writer_node(state: AgentState,config: RunnableConfig | None = None,*,store: BaseStore | None = None,) -> AgentState:
        """Persist error-specific mistake memories when tool failures remain unresolved."""
        tool_result = state.get("tool_result")
        # Only write error memory for real unresolved tool failures.
        if store is None or tool_result is None or tool_result.get("success", True):
            return {**state, "error_memory_written_count": 0}

        ns = memory_namespace(config)
        signature = state.get("error_signature") or build_error_signature(state)
        # Fast dedupe guard: skip if this error signature already exists.
        existing = store.search(ns, query=signature, limit=5)
        for item in existing:
            if item.value.get("metadata", {}).get("error_signature") == signature:
                return {**state, "error_memory_written_count": 0}

        # Feed prior memories into extractor to avoid writing near-duplicates.
        existing_items = store.search(ns, limit=30)
        existing_memories_text = "\n".join(
            item.value.get("content", "") for item in existing_items if item.value.get("content")
        ) or "(empty)"

        # Compact failure transcript used as extraction source.
        assistant_response = (
            "Tool execution failed after repair attempts.\n"
            f"Tool: {tool_result.get('tool_name', 'unknown_tool')}\n"
            f"Error: {tool_result.get('error', '')}\n"
            f"Failed code:\n{tool_result.get('tool_input', '')}\n"
            f"Critique issues: {', '.join(state['critique_result']['issues'])}"
        )

        decision = extract_structured_memories(
            user_query=f"Tool failure signature: {signature}",
            assistant_response=assistant_response,
            intent=state["intent"],
            critique_passed=state["critique_result"]["passed"],
            critique_issues=state["critique_result"]["issues"],
            existing_memories_text=existing_memories_text,
        )

        errors_written = 0
        if decision.should_write:
            for memory in decision.memories:
                # Error node stores only mistake memories to keep semantics clean.
                if memory.memory_type != MemoryType.MISTAKE:
                    continue
                if not memory.is_new or not memory.text.strip():
                    continue
                metadata = dict(memory.metadata)
                metadata["error_signature"] = signature
                store.put(
                    ns,
                    str(uuid.uuid4()),
                    {
                        "content": memory.text.strip(),
                        "memory_type": MemoryType.MISTAKE.value,
                        "intent": state["intent"],
                        "metadata": metadata,
                    },
                )
                errors_written += 1
                break

        return {
            **state,
            "error_memory_written_count": errors_written,
            "memory_written_count": state.get("memory_written_count", 0) + errors_written,
        }

    def route_after_critique(state: AgentState) -> Literal["memory", "repair", "retrieve_error_memory", "error_memory_writer"]:
        """
        Conditional routing after critique with error-memory handling.
        """
        critique_passed = state["critique_result"]["passed"]
        tool_result = state.get("tool_result")
        tool_failed = tool_result is not None and not tool_result.get("success", True)

        if critique_passed and not tool_failed:
            return "memory"

        if state["repair_count"] >= state["max_repairs"]:
            if tool_failed:
                return "error_memory_writer"
            return "memory"

        if tool_failed:
            return "retrieve_error_memory"

        return "repair"

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("memory_evaluator", memory_evaluator_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("tools", tool_node)
    graph.add_node("process_tool_result", process_tool_result)
    graph.add_node("critique_response", critique_node)
    graph.add_node("retrieve_error_memory", retrieve_error_memory_node)
    graph.add_node("repair_response", repair_node)
    graph.add_node("error_memory_writer", error_memory_writer_node)
    graph.add_node("record_turn", record_turn)
    graph.add_node("memory_writer", memory_writer_node)

    # Set entry point
    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "retrieve_context")

    # Conditional summarization: routes to summarize or directly to memory evaluator
    graph.add_conditional_edges(
        "retrieve_context",
        should_summarize,
        {"summarize": "summarize", "continue": "memory_evaluator"},
    )
    graph.add_edge("summarize", "memory_evaluator")
    
    # Memory evaluator decides memory vs tools, then generates response
    graph.add_edge("memory_evaluator", "generate_response")
    
    # After generate_response: use tools_condition to route (LLM decides)
    # tools_condition returns "tools" if tool_calls present, END otherwise
    graph.add_conditional_edges(
        "generate_response",
        tools_condition,
        {"tools": "tools", END: "critique_response"},
    )
    
    # After tool execution, process the result
    graph.add_edge("tools", "process_tool_result")
    graph.add_edge("process_tool_result", "critique_response")

    # Critique/repair loop
    graph.add_conditional_edges(
        "critique_response",
        route_after_critique,
        {
            "memory": "memory_writer",
            "repair": "repair_response",
            "retrieve_error_memory": "retrieve_error_memory",
            "error_memory_writer": "error_memory_writer",
        },
    )

    # On tool failure, fetch error-specific memories before attempting repair.
    graph.add_edge("retrieve_error_memory", "repair_response")

    # After repair: use tools_condition again
    graph.add_conditional_edges(
        "repair_response",
        tools_condition,
        {"tools": "tools", END: "critique_response"},
    )
    
    # Final steps
    graph.add_edge("memory_writer", "record_turn")
    graph.add_edge("error_memory_writer", "record_turn")
    graph.add_edge("record_turn", END)

    # Compile with both STM (checkpointer) and optional LTM (store).
    return graph.compile(checkpointer=checkpointer, store=store)