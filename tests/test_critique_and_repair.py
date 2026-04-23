# ---------------------------------------------------------------------------
# tests/test_critique_and_repair.py
# Purpose: Test the critique node, repair loop, and conditional routing.
#
# Tests cover:
#   - Critique pass routes directly to memory_writer (no repair)
#   - Critique fail triggers repair; repair success passes second critique
#   - Repair exhaustion exits after max_repairs
#   - Repair count increments correctly through cycles
# ---------------------------------------------------------------------------
from __future__ import annotations

import pytest

from smart_home_langgraph.data.loader import HomeDataLoader
from smart_home_langgraph.graph.state import CritiqueResult
from smart_home_langgraph.graph.workflow import build_workflow, initial_state
from smart_home_langgraph.memory.retriever import MemoryRetriever
from smart_home_langgraph.memory.store import MistakeMemory, PreferenceMemory, RecipeMemory


@pytest.fixture
def temp_memory(tmp_path):
    return MemoryRetriever(
        mistake_store=MistakeMemory(str(tmp_path / "mistakes.json")),
        recipe_store=RecipeMemory(str(tmp_path / "recipes.json")),
        preference_store=PreferenceMemory(str(tmp_path / "preferences.json")),
    )


@pytest.fixture
def temp_loader():
    return HomeDataLoader()


def test_critique_pass_routes_to_end(temp_memory, temp_loader):
    """Response quality is good on first try; no repair triggered."""

    def fake_response_gen(state):
        return "Turn off the AC between 2-4 PM when outside temp is mild.", True

    def fake_critique_gen(state) -> CritiqueResult:
        return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}

    workflow = build_workflow(
        response_generator=fake_response_gen,
        critique_generator=fake_critique_gen,
        loader=temp_loader,
        memory_retriever=temp_memory,
        max_repairs=2,
    )
    result = workflow.invoke(initial_state("How can I reduce my evening power usage?"))

    assert result["intent"] == "energy_optimization"
    assert result["response"] == "Turn off the AC between 2-4 PM when outside temp is mild."
    assert result["critique_result"]["passed"] is True
    assert result["repair_count"] == 0


def test_critique_fail_triggers_repair(temp_memory, temp_loader):
    """Critique fails on first response; repair succeeds on second attempt."""

    def fake_response_gen(state):
        if "Please provide an improved response" in state.get("user_query", ""):
            return "Optimized: Lower AC temperature gradually to save 15% energy.", True
        return "Maybe try some things.", True

    def fake_critique_gen(state) -> CritiqueResult:
        if "Optimized:" in state["response"]:
            return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}
        return {
            "passed": False,
            "issues": ["Too vague", "Not actionable"],
            "severity": "medium",
            "repair_hints": "Provide specific temperature targets and timing.",
        }

    workflow = build_workflow(
        response_generator=fake_response_gen,
        critique_generator=fake_critique_gen,
        loader=temp_loader,
        memory_retriever=temp_memory,
        max_repairs=2,
    )
    result = workflow.invoke(initial_state("How can I reduce my evening power usage?"))

    assert result["intent"] == "energy_optimization"
    assert "Optimized:" in result["response"]
    assert result["critique_result"]["passed"] is True
    assert result["repair_count"] == 1


def test_repair_exhaustion_exits_after_max_repairs(temp_memory, temp_loader):
    """Critique always fails; workflow exits after max_repairs attempts."""
    call_count = {"n": 0}

    def fake_response_gen(state):
        call_count["n"] += 1
        return f"Response attempt {call_count['n']}", True

    def fake_critique_gen(state) -> CritiqueResult:
        return {
            "passed": False,
            "issues": ["Still not good"],
            "severity": "high",
            "repair_hints": "Try harder.",
        }

    workflow = build_workflow(
        response_generator=fake_response_gen,
        critique_generator=fake_critique_gen,
        loader=temp_loader,
        memory_retriever=temp_memory,
        max_repairs=2,
    )
    result = workflow.invoke(initial_state("How can I reduce my evening power usage?"))

    assert result["repair_count"] == 2
    assert result["critique_result"]["passed"] is False
    assert "Response attempt 3" in result["response"]  # initial + 2 repairs


def test_repair_count_increments_correctly(temp_memory, temp_loader):
    """Repair count correctly tracks multiple cycles."""
    repair_sequence = []

    def fake_response_gen(state):
        is_repair = "Please provide an improved response" in state.get("user_query", "")
        repair_sequence.append("repair" if is_repair else "initial")
        return "Response", True

    def fake_critique_gen(state) -> CritiqueResult:
        repair_count = len([x for x in repair_sequence if x == "repair"])
        if repair_count >= 2:
            return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}
        return {
            "passed": False,
            "issues": ["Issue"],
            "severity": "medium",
            "repair_hints": "Fix it.",
        }

    workflow = build_workflow(
        response_generator=fake_response_gen,
        critique_generator=fake_critique_gen,
        loader=temp_loader,
        memory_retriever=temp_memory,
        max_repairs=3,
    )
    result = workflow.invoke(initial_state("Test query", max_repairs=3))

    assert result["repair_count"] >= 1
    assert result["critique_result"]["passed"] is True
