
# Dependency injection:
#   build_workflow() accepts optional callable overrides for the LLM generator
#   and critique evaluator. Tests pass fake callables so they run fast and
#   never touch external APIs.
# ---------------------------------------------------------------------------------
import uuid
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore

from smart_home_langgraph.data.loader import HomeDataLoader
from smart_home_langgraph.evaluation.metrics import EpisodeRecord
from smart_home_langgraph.graph.state import AgentState, CritiqueResult
from smart_home_langgraph.memory.ltm_schema import MemoryType
from smart_home_langgraph.services.critique_client import critique_response
from smart_home_langgraph.services.gemini_client import generate_with_gemini
from smart_home_langgraph.services.memory_extractor import extract_structured_memories

# Type aliases for injectable callables (used for dependency injection in tests).
ResponseGenerator = Callable[[AgentState], tuple[str, bool]]
CritiqueGenerator = Callable[[AgentState], CritiqueResult]


def initial_state(
    query: str,
    max_repairs: int = 2,
    conversation_history: list[BaseMessage] | None = None,
) -> AgentState:
    """
    Return the default initial state for starting a new agent run.

    Call this in main.py and in tests to avoid repeating the boilerplate dict.
    """
    return {
        "user_query": query,
        "conversation_history": list(conversation_history or []),
        "summary": "",
        "intent": "",
        "sensor_context": "",
        "memory_context": "",
        "response": "",
        "used_live_llm": False,
        "critique_result": {
            "passed": False,
            "issues": [],
            "severity": "low",
            "repair_hints": "",
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
    }


def build_workflow(
    response_generator: ResponseGenerator | None = None,
    critique_generator: CritiqueGenerator | None = None,
    loader: HomeDataLoader | None = None,
    checkpointer: SqliteSaver | None = None,
    # Native LangGraph long-term store (e.g., PostgresStore).
    store: BaseStore | None = None,
    max_repairs: int = 2,
):
    """
    Build and compile the smart-home agent graph.

    Parameters (all optional — defaults wire up the live system):
      response_generator  Callable that produces (response_text, used_live_llm).
                          Default: generate_with_gemini (real Gemini call).
      critique_generator  Callable that returns a CritiqueResult dict.
                          Default: critique_response (real Gemini critique).
    loader              HomeDataLoader that serves telemetry summaries
                  backed by Postgres structured store.
      checkpointer        LangGraph checkpointer for persistent chat state.
                            Default: no checkpointing.
      store               LangGraph BaseStore for long-term memory.
                            Default: no persistent store.
      max_repairs         Maximum repair loop iterations. Default: 2.
    """
    data_loader = loader or HomeDataLoader()
    response_gen = response_generator or generate_with_gemini
    critique_gen = critique_generator or critique_response

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

    # ------------------------------------------------------------------
    # Node definitions (closures capture sim, retriever, response_gen, etc.)
    # ------------------------------------------------------------------

    @traceable(name="detect_intent", run_type="chain")
    def detect_intent(state: AgentState) -> AgentState:
        # add an llm here
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

    @traceable(name="retrieve_context", run_type="retriever")
    def retrieve_context(state: AgentState,config: RunnableConfig | None = None,*,store: BaseStore | None = None,) -> AgentState:
        """
        Build two context strings injected into the generation prompt:
          1. sensor_context  — recent telemetry summary from structured store
          2. memory_context  — mistakes + recipes + preferences from memory stores
        """
        sensor_context = data_loader.context_window(hours=24)
        if store is None:
            memory_context = "No long-term memory store configured."
        else:
            memories = store.search(memory_namespace(config), query=state["user_query"], limit=8)
            memory_context = format_memories_for_prompt(memories)
        return {**state, "sensor_context": sensor_context, "memory_context": memory_context}

    @traceable(name="generate_response", run_type="chain")
    def generate_response(state: AgentState) -> AgentState:
        """Call Gemini (or injected fake) to produce the initial response."""
        response_text, used_live_llm = response_gen(state)
        return {**state, "response": response_text, "used_live_llm": used_live_llm}

    @traceable(name="record_turn", run_type="chain")
    def record_turn(state: AgentState) -> AgentState:
        """Append the completed user/assistant turn to chat history."""
        return {
            **state,
            "conversation_history": [
                HumanMessage(content=state["user_query"]),
                AIMessage(content=state["response"]),
            ],
        }

    def should_summarize(state: AgentState) -> str:
        """Route to summarize_node if conversation > 6 messages, else continue to generate."""
        if len(state["conversation_history"]) > 6:
            return "summarize"
        return "continue"

    @traceable(name="summarize_history", run_type="chain")
    def summarize_node(state: AgentState) -> AgentState:
        """
        Summarize older messages to keep context window manageable.
        Keeps last 2 messages, summarizes the rest, removes originals using RemoveMessage.
        """
        history = state["conversation_history"]
        
        if len(history) > 6:
            # Keep last 2 messages, summarize everything else
            msgs_to_summarize = history[:-2]
            last_2_msgs = history[-2:]
            
            # Call generate_with_gemini with is_summary flag (uses _build_summary_prompt, no smart-home context pollution)
            summary_content, _ = generate_with_gemini(is_summary=True, messages_to_summarize=msgs_to_summarize)
            
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
                "conversation_history": remove_messages + [summary_msg] + last_2_msgs,
                "summary": summary_content,
            }
        
        return state

    @traceable(name="critique_node", run_type="chain")
    def critique_node(state: AgentState) -> AgentState:
        """Evaluate response quality; store structured result for routing."""
        result = critique_gen(state)
        return {**state, "critique_result": result}

    @traceable(name="repair_response", run_type="chain")
    def repair_node(state: AgentState) -> AgentState:
        """
        Regenerate the response using critique hints.
        Appends repair instructions to the original query so the generator
        knows what to fix. Increments repair_count for loop termination.
        """
        hints = (
            f"\n\nPrevious response had issues. Repair hints: {state['critique_result']['repair_hints']}\n"
            f"Issues identified: {', '.join(state['critique_result']['issues'])}\n"
            "Please provide an improved response."
        )
        repair_state = {**state, "user_query": state["user_query"] + hints}
        response_text, used_live_llm = response_gen(repair_state)
        return {
            **state,
            "response": response_text,
            "used_live_llm": used_live_llm or state["used_live_llm"],
            "repair_count": state["repair_count"] + 1,
        }

    @traceable(name="memory_writer", run_type="chain")
    def memory_writer_node(state: AgentState,config: RunnableConfig | None = None,*,store: BaseStore | None = None,) -> AgentState:
        """
        Write typed long-term memories to the LangGraph store.

        Memory extraction is delegated to a structured-output LLM call and
        duplicate suppression is performed via the returned `is_new` flags.
        """
        memories_written = 0

        if store is not None:
            ns = memory_namespace(config)
            existing_items = store.search(ns, limit=30)
            existing_memories_text = "\n".join(
                item.value.get("content", "") for item in existing_items if item.value.get("content")
            ) or "(empty)"

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

    def should_repair(state: AgentState) -> str:
        """
        Conditional routing after critique.
        Returns "memory" (accept + learn) or "repair" (regenerate).
        Exits to "memory" if critique passed OR repair attempts are exhausted.
        """
        if state["critique_result"]["passed"]:
            return "memory"
        if state["repair_count"] >= state["max_repairs"]:
            return "memory"
        return "repair"

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent", detect_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("summarize", summarize_node)
    graph.add_node("generate_response", generate_response)
    graph.add_node("critique_response", critique_node)
    graph.add_node("repair_response", repair_node)
    graph.add_node("record_turn", record_turn)
    graph.add_node("memory_writer", memory_writer_node)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "retrieve_context")

    graph.add_conditional_edges(
        "retrieve_context",
        should_summarize,
        {"summarize": "summarize", "continue": "generate_response"},
    )
    graph.add_edge("summarize", "generate_response")
    graph.add_edge("generate_response", "critique_response")

    graph.add_conditional_edges(
        "critique_response",
        should_repair,
        {"memory": "memory_writer", "repair": "repair_response"},
    )

    graph.add_edge("repair_response", "critique_response")
    graph.add_edge("memory_writer", "record_turn")
    graph.add_edge("record_turn", END)

    # Compile with both STM (checkpointer) and optional LTM (store).
    return graph.compile(checkpointer=checkpointer, store=store)