# ---------------------------------------------------------------------------
# tests/test_workflow.py
# Purpose: End-to-end tests for the unified smart-home agent workflow.
#
# Tests cover:
#   - Intent detection from query keywords
#   - Context population (sensor + memory)
#   - Fallback when Gemini API key is absent
#   - Full workflow with injected fake generator
# ---------------------------------------------------------------------------
from __future__ import annotations

import pytest

from smart_home_langgraph.data.loader import HomeDataLoader
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


def test_workflow_populates_context_and_response(temp_memory, temp_loader):
    """Full workflow run with injected fake generator; verify all context fields populated."""
    captured: dict = {}

    def fake_generator(state):
        captured.update(state)
        return "FAKE_RESPONSE", False

    def fake_critique(state):
        return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}

    app = build_workflow(
        response_generator=fake_generator,
        critique_generator=fake_critique,
        loader=temp_loader,
        memory_retriever=temp_memory,
    )
    result = app.invoke(initial_state("How can I reduce energy usage at night?"))

    assert result["response"] == "FAKE_RESPONSE"
    assert result["intent"] == "energy_optimization"
    assert "Sensor summary" in captured["sensor_context"]
    assert "=== Memory Context ===" in captured["memory_context"]


def test_intent_detection_energy(temp_memory, temp_loader):
    """Queries containing energy keywords → energy_optimization intent."""

    def fake_gen(state):
        return "ok", False

    def fake_critique(state):
        return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}

    app = build_workflow(
        response_generator=fake_gen,
        critique_generator=fake_critique,
        loader=temp_loader,
        memory_retriever=temp_memory,
    )
    result = app.invoke(initial_state("My power bill is too high"))
    assert result["intent"] == "energy_optimization"


def test_intent_detection_comfort(temp_memory, temp_loader):
    """Queries containing comfort keywords → comfort_optimization intent."""

    def fake_gen(state):
        return "ok", False

    def fake_critique(state):
        return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}

    app = build_workflow(
        response_generator=fake_gen,
        critique_generator=fake_critique,
        loader=temp_loader,
        memory_retriever=temp_memory,
    )
    result = app.invoke(initial_state("It's too hot in the living room"))
    assert result["intent"] == "comfort_optimization"


def test_fallback_without_api_key(monkeypatch, temp_memory, temp_loader):
    """When GEMINI_API_KEY is absent, workflow completes safely with fallback response."""
    from smart_home_langgraph.config.settings import Settings

    monkeypatch.setattr(
        "smart_home_langgraph.services.gemini_client.get_settings",
        lambda: Settings(gemini_api_key=None),
    )

    def fake_critique(state):
        return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}

    app = build_workflow(
        critique_generator=fake_critique,
        loader=temp_loader,
        memory_retriever=temp_memory,
    )
    result = app.invoke(initial_state("Optimize comfort and power usage"))

    assert result["used_live_llm"] is False
    assert "fallback" in result["response"].lower()
