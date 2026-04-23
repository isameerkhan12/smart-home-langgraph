# ---------------------------------------------------------------------------
# graph/workflow.py
# Purpose: The unified smart-home agent workflow.
#
# Graph shape:
#
#   [START]
#      |
#      v
#   detect_intent            -- classify the user query into a task class
#      |
#      v
#   retrieve_context         -- pull sensor summary + long-term memory
#      |
#      v
#   generate_response        -- call Gemini (with fallback) to produce an answer
#      |
#      v
#   critique_response        -- evaluate answer quality (pass / fail + hints)
#      |
#      +-- (passed OR retries exhausted) --> memory_writer --> [END]
#      |
#      +-- (failed AND retries remain)   --> repair_response
#                                                 |
#                                                 v
#                                          critique_response  (loop)
#
# Dependency injection:
#   build_workflow() accepts optional callable overrides for the LLM generator
#   and critique evaluator. Tests pass fake callables so they run fast and
#   never touch external APIs.
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
from typing import Callable

from langgraph.graph import END, StateGraph

from smart_home_langgraph.data.loader import HomeDataLoader
from smart_home_langgraph.evaluation.metrics import EpisodeRecord
from smart_home_langgraph.graph.state import AgentState, CritiqueResult
from smart_home_langgraph.memory.retriever import MemoryRetriever
from smart_home_langgraph.memory.store import MistakeMemory, PreferenceMemory, RecipeMemory
from smart_home_langgraph.services.critique_client import critique_response
from smart_home_langgraph.services.gemini_client import generate_with_gemini
from smart_home_langgraph.services.memory_writer import write_learnings

# Type aliases for injectable callables (used for dependency injection in tests).
ResponseGenerator = Callable[[AgentState], tuple[str, bool]]
CritiqueGenerator = Callable[[AgentState], CritiqueResult]


def initial_state(query: str, max_repairs: int = 2) -> AgentState:
    """
    Return the default initial state for starting a new agent run.

    Call this in main.py and in tests to avoid repeating the boilerplate dict.
    """
    return {
        "user_query": query,
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


def _default_memory_retriever(base_dir: str) -> MemoryRetriever:
    """Create JSON-backed memory stores under a runtime directory."""
    os.makedirs(base_dir, exist_ok=True)
    return MemoryRetriever(
        mistake_store=MistakeMemory(os.path.join(base_dir, "mistakes.json")),
        recipe_store=RecipeMemory(os.path.join(base_dir, "recipes.json")),
        preference_store=PreferenceMemory(os.path.join(base_dir, "preferences.json")),
    )


def build_workflow(
    response_generator: ResponseGenerator | None = None,
    critique_generator: CritiqueGenerator | None = None,
    loader: HomeDataLoader | None = None,
    memory_retriever: MemoryRetriever | None = None,
    max_repairs: int = 2,
):
    """
    Build and compile the smart-home agent graph.

    Parameters (all optional — defaults wire up the live system):
      response_generator  Callable that produces (response_text, used_live_llm).
                          Default: generate_with_gemini (real Gemini call).
      critique_generator  Callable that returns a CritiqueResult dict.
                          Default: critique_response (real Gemini critique).
    loader              HomeDataLoader that reads sensor data from Excel.
                  Default: reads data/home_data.xlsx.
      memory_retriever    MemoryRetriever backed by JSON files.
                          Default: runtime_memory/ in the working directory.
      max_repairs         Maximum repair loop iterations. Default: 2.
    """
    data_loader = loader or HomeDataLoader()
    retriever = memory_retriever or _default_memory_retriever(
        os.path.join(os.getcwd(), "runtime_memory")
    )
    response_gen = response_generator or generate_with_gemini
    critique_gen = critique_generator or critique_response

    # ------------------------------------------------------------------
    # Node definitions (closures capture sim, retriever, response_gen, etc.)
    # ------------------------------------------------------------------

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

    def retrieve_context(state: AgentState) -> AgentState:
        """
        Build two context strings injected into the generation prompt:
          1. sensor_context  — recent 24h sensor summary from the simulator
          2. memory_context  — mistakes + recipes + preferences from memory stores
        """
        sensor_context = data_loader.context_window(hours=24)
        memory_context = retriever.retrieve(state["intent"]) # this type os task what should i remember
        return {**state, "sensor_context": sensor_context, "memory_context": memory_context}

    def generate_response(state: AgentState) -> AgentState:
        """Call Gemini (or injected fake) to produce the initial response."""
        response_text, used_live_llm = response_gen(state)
        return {**state, "response": response_text, "used_live_llm": used_live_llm}

    def critique_node(state: AgentState) -> AgentState:
        """Evaluate response quality; store structured result for routing."""
        result = critique_gen(state)
        return {**state, "critique_result": result}

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

    def memory_writer_node(state: AgentState) -> AgentState:
        """
        Write learnings to persistent memory and populate episode metrics.

        - Failed critique  → write issues + corrective rules to MistakeMemory
        - Passed critique  → write strategy to RecipeMemory (score 0.9 or 0.7)
        """
        outcome = write_learnings(state, retriever._mistakes, retriever._recipes)
        episode = EpisodeRecord(
            episode_id=1,
            task_class=state["intent"],
            critique_passed_first_try=outcome.critique_first_pass,
            repeated_known_mistake=False,
            preferences_respected=False,
            used_existing_recipe=outcome.recipes_written > 0,
        )
        return {
            **state,
            "episode_record": episode,
            "memory_written_count": outcome.mistakes_written + outcome.recipes_written,
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
    graph.add_node("generate_response", generate_response)
    graph.add_node("critique_response", critique_node)
    graph.add_node("repair_response", repair_node)
    graph.add_node("memory_writer", memory_writer_node)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", "critique_response")

    graph.add_conditional_edges(
        "critique_response",
        should_repair,
        {"memory": "memory_writer", "repair": "repair_response"},
    )

    graph.add_edge("repair_response", "critique_response")
    graph.add_edge("memory_writer", END)

    return graph.compile()
