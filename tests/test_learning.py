# ---------------------------------------------------------------------------
# tests/test_learning.py
# Purpose: Test memory writing and episode metric population.
#
# Tests cover:
#   - Failed critiques write to MistakeMemory
#   - Successful critiques write to RecipeMemory
#   - EpisodeRecord is populated with metrics after a run
#   - Learnings persist in JSON files across separate retriever instances
#   - MetricsTracker computes critique_first_pass_rate from EpisodeRecords
#   - First-pass vs. repair get different success scores (0.9 vs 0.7)
# ---------------------------------------------------------------------------
from __future__ import annotations

import json
import os

import pytest

from smart_home_langgraph.data.loader import HomeDataLoader
from smart_home_langgraph.evaluation.metrics import EpisodeRecord, MetricsTracker
from smart_home_langgraph.graph.workflow import build_workflow, initial_state
from smart_home_langgraph.memory.retriever import MemoryRetriever
from smart_home_langgraph.memory.store import MistakeMemory, PreferenceMemory, RecipeMemory
from smart_home_langgraph.services.memory_writer import write_learnings


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


def _state_with_failed_critique(intent: str) -> dict:
    """Minimal state dict for write_learnings with a failed critique."""
    return {
        "user_query": "Test query",
        "intent": intent,
        "sensor_context": "",
        "memory_context": "",
        "response": "Maybe do something",
        "used_live_llm": False,
        "critique_result": {
            "passed": False,
            "issues": ["Too vague", "Not specific"],
            "severity": "medium",
            "repair_hints": "Provide concrete actions with timing",
        },
        "repair_count": 0,
        "max_repairs": 2,
        "episode_record": {},
        "memory_written_count": 0,
    }


def _state_with_passed_critique(intent: str, response: str, repair_count: int = 0) -> dict:
    """Minimal state dict for write_learnings with a passed critique."""
    return {
        "user_query": "Test query",
        "intent": intent,
        "sensor_context": "",
        "memory_context": "",
        "response": response,
        "used_live_llm": True,
        "critique_result": {
            "passed": True,
            "issues": [],
            "severity": "low",
            "repair_hints": "",
        },
        "repair_count": repair_count,
        "max_repairs": 2,
        "episode_record": {},
        "memory_written_count": 0,
    }


def test_write_learnings_writes_mistakes_on_fail(tmp_path):
    """Failed critique → two issue records written to MistakeMemory."""
    mistake_store = MistakeMemory(str(tmp_path / "mistakes.json"))
    recipe_store = RecipeMemory(str(tmp_path / "recipes.json"))

    outcome = write_learnings(
        _state_with_failed_critique("energy_optimization"), mistake_store, recipe_store
    )

    assert outcome.mistakes_written == 2
    assert outcome.had_critique_failures is True
    assert outcome.critique_first_pass is False

    with open(tmp_path / "mistakes.json") as f:
        data = json.load(f)
    assert len(data) == 2
    assert any("Too vague" in m.get("error_description", "") for m in data)


def test_write_learnings_writes_recipe_on_pass(tmp_path):
    """Passed critique → recipe written to RecipeMemory with strategy."""
    mistake_store = MistakeMemory(str(tmp_path / "mistakes.json"))
    recipe_store = RecipeMemory(str(tmp_path / "recipes.json"))

    outcome = write_learnings(
        _state_with_passed_critique(
            "energy_optimization",
            "Turn off AC when outside is mild.",
        ),
        mistake_store,
        recipe_store,
    )

    assert outcome.recipes_written == 1
    assert outcome.critique_first_pass is True

    with open(tmp_path / "recipes.json") as f:
        data = json.load(f)
    assert len(data) == 1
    assert "Turn off AC" in data[0].get("strategy_steps", [])[0]


def test_episode_record_populated_after_workflow_run(temp_memory, temp_loader):
    """Full workflow run populates episode_record with critique metrics."""

    def fake_gen(state):
        return "Practical energy-saving strategy with concrete steps.", True

    def fake_critique(state):
        return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}

    app = build_workflow(
        response_generator=fake_gen,
        critique_generator=fake_critique,
        loader=temp_loader,
        memory_retriever=temp_memory,
    )
    result = app.invoke(initial_state("How can I reduce my evening power usage?"))

    assert result["episode_record"].critique_passed_first_try is True
    assert result["episode_record"].task_class == "energy_optimization"
    assert result["memory_written_count"] >= 1


def test_memory_persistence_across_retrievers(tmp_path):
    """Recipes written by one retriever are readable by a new retriever from the same files."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    r1 = MemoryRetriever(
        mistake_store=MistakeMemory(str(memory_dir / "mistakes.json")),
        recipe_store=RecipeMemory(str(memory_dir / "recipes.json")),
        preference_store=PreferenceMemory(str(memory_dir / "preferences.json")),
    )
    write_learnings(
        _state_with_passed_critique("energy_optimization", "Turn off AC when outside is mild."),
        r1._mistakes,
        r1._recipes,
    )

    r2 = MemoryRetriever(
        mistake_store=MistakeMemory(str(memory_dir / "mistakes.json")),
        recipe_store=RecipeMemory(str(memory_dir / "recipes.json")),
        preference_store=PreferenceMemory(str(memory_dir / "preferences.json")),
    )
    best = r2._recipes.best_recipe("energy_optimization")
    assert best is not None
    assert "Turn off AC" in " ".join(best.strategy_steps)


def test_metrics_tracker_computes_critique_first_pass_rate():
    """MetricsTracker computes critique_first_pass_rate correctly."""
    tracker = MetricsTracker()
    tracker.add_episode(
        EpisodeRecord(episode_id=1, task_class="energy_optimization",
                      critique_passed_first_try=True, repeated_known_mistake=False,
                      preferences_respected=False, used_existing_recipe=False)
    )
    tracker.add_episode(
        EpisodeRecord(episode_id=2, task_class="comfort_optimization",
                      critique_passed_first_try=True, repeated_known_mistake=False,
                      preferences_respected=False, used_existing_recipe=False)
    )
    tracker.add_episode(
        EpisodeRecord(episode_id=3, task_class="energy_optimization",
                      critique_passed_first_try=False, repeated_known_mistake=False,
                      preferences_respected=False, used_existing_recipe=False)
    )
    summary = tracker.summary()
    assert 65 < summary["critique_first_pass_rate_%"] < 68  # 2/3 = 66.7%


def test_repair_lowers_success_score(tmp_path):
    """First-pass success gets score 0.9; success after repair gets 0.7."""
    ms = MistakeMemory(str(tmp_path / "mistakes.json"))

    # First pass
    rs1 = RecipeMemory(str(tmp_path / "recipes_fp.json"))
    write_learnings(_state_with_passed_critique("energy_optimization", "Good.", repair_count=0), ms, rs1)
    with open(tmp_path / "recipes_fp.json") as f:
        score_fp = json.load(f)[0]["critique_score"]

    # After repair
    rs2 = RecipeMemory(str(tmp_path / "recipes_ar.json"))
    write_learnings(_state_with_passed_critique("energy_optimization", "Good.", repair_count=1), ms, rs2)
    with open(tmp_path / "recipes_ar.json") as f:
        score_ar = json.load(f)[0]["critique_score"]

    assert score_fp == 0.9
    assert score_ar == 0.7
    assert score_fp > score_ar


def test_workflow_fail_then_repair_writes_recipe(temp_memory, temp_loader):
    """Critique fail → repair → pass writes recipe with lower success score."""
    calls = {"gen": 0, "crit": 0}

    def fake_gen(state):
        calls["gen"] += 1
        return "Vague response" if calls["gen"] == 1 else "Concrete: Turn off AC at 2 PM.", True

    def fake_critique(state):
        calls["crit"] += 1
        if calls["crit"] == 1:
            return {"passed": False, "issues": ["Too vague"], "severity": "medium",
                    "repair_hints": "Provide specific actions."}
        return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}

    app = build_workflow(
        response_generator=fake_gen,
        critique_generator=fake_critique,
        loader=temp_loader,
        memory_retriever=temp_memory,
    )
    result = app.invoke(initial_state("How to save energy?"))

    assert result["episode_record"].critique_passed_first_try is False
    assert result["repair_count"] == 1
    assert result["memory_written_count"] >= 1
