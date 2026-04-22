# ---------------------------------------------------------------------------
# graph/workflow.py  —  Phase 0: Hello Graph
# Purpose: Build and wire the Phase 0 LangGraph workflow.
#
# Graph shape (sequential, no branches yet):
#
#   [START] --> detect_intent --> build_response --> [END]
#
# Each box is a "node" (a plain Python function).
# The arrows are "edges" (they tell LangGraph which node to run next).
# In later phases we will add conditional edges (if/else routing) and
# iterative loops (retry on critique failure).
# ---------------------------------------------------------------------------
from __future__ import annotations

from langgraph.graph import END, StateGraph
# StateGraph  – the main LangGraph class; you register nodes and edges on it,
#               then call .compile() to get a runnable app.
# END         – a special LangGraph sentinel that marks the finish of the graph.
#               An edge pointing to END means "stop after this node".

from smart_home_langgraph.graph.state import AgentState


# ---------------------------------------------------------------------------
# NODE 1 – detect_intent
# Role   : Read the user query and decide WHAT the user wants.
# Input  : state with user_query filled in, intent and response still empty.
# Output : same state but with intent set to one of the four categories.
# ---------------------------------------------------------------------------
def detect_intent(state: AgentState) -> AgentState:
    # Lowercase the query so keyword matching is case-insensitive.
    query = state["user_query"].lower()

    # Simple keyword-based intent classifier.
    # In later phases this will be replaced by an LLM call via Gemini.
    if any(word in query for word in ["energy", "power", "bill", "save"]):
        intent = "energy_optimization"   # user wants to cut energy usage
    elif any(word in query for word in ["temperature", "comfort", "hot", "cold"]):
        intent = "comfort_optimization"  # user wants better indoor climate
    elif any(word in query for word in ["anomaly", "strange", "unusual", "fault"]):
        intent = "anomaly_explanation"   # user notices something odd in sensor data
    else:
        intent = "general_question"      # fallback for everything else

    # {**state, "intent": intent} creates a NEW dict that copies all existing
    # state fields and overrides only the "intent" key.
    # LangGraph merges this returned dict back into the shared state.
    return {**state, "intent": intent}


# ---------------------------------------------------------------------------
# NODE 2 – build_response
# Role   : Use the detected intent to produce a starter recommendation.
# Input  : state with user_query and intent already filled in.
# Output : same state but with response filled in.
# ---------------------------------------------------------------------------
def build_response(state: AgentState) -> AgentState:
    intent = state["intent"]      # what the user wants (set by previous node)
    query = state["user_query"]   # original question (kept for context in the reply)

    # A lookup table: intent name -> starter advice string.
    # In later phases the Gemini LLM will generate richer, data-grounded answers.
    suggestions = {
        "energy_optimization": "Check peak-hour appliance usage and suggest shifting heavy loads.",
        "comfort_optimization": "Inspect occupancy and indoor climate trends to tune HVAC setpoints.",
        "anomaly_explanation": "Look for sudden deviations in sensor trends and explain likely causes.",
        "general_question": "Provide a concise, data-grounded smart-home recommendation.",
    }

    # Compose the final text response that the user will see.
    response = (
        f"Intent detected: {intent}. "
        f"Starter plan: {suggestions[intent]} "
        f"Original query: {query}"
    )

    return {**state, "response": response}


# ---------------------------------------------------------------------------
# GRAPH ASSEMBLY  –  build_workflow()
# This function wires all nodes together into a compiled runnable graph.
# ---------------------------------------------------------------------------
def build_workflow():
    # StateGraph(AgentState) creates an empty graph that will pass AgentState
    # dicts between nodes. Every node must accept and return AgentState.
    graph = StateGraph(AgentState)

    # add_node(name, function) registers a node.
    # The name is used in add_edge() calls and in debug logs.
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("build_response", build_response)

    # set_entry_point(name) marks which node runs first when we call app.invoke().
    graph.set_entry_point("detect_intent")

    # add_edge(from, to) creates an unconditional sequential connection.
    # In later phases we will use add_conditional_edges() for if/else routing.
    graph.add_edge("detect_intent", "build_response")
    graph.add_edge("build_response", END)  # after build_response the graph stops

    # compile() validates the graph (no disconnected nodes, entry point set, etc.)
    # and returns an executable object we can call with .invoke() or .stream().
    return graph.compile()
