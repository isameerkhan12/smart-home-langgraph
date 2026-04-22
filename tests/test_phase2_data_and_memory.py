# ---------------------------------------------------------------------------
# tests/test_phase2_data_and_memory.py  –  Phase 2 Tests
#
# What we test:
#   Part A – SmartHomeSimulator
#     1. generate() produces the right number of rows and expected columns
#     2. daily_summary() produces one row per day
#     3. context_window() returns a non-empty text summary
#     4. Anomaly rows exist when anomaly_probability is high
#
#   Part B – Memory Stores (PreferenceMemory, MistakeMemory, RecipeMemory)
#     5. Save and reload a preference; verify round-trip
#     6. Updating the same preference key overwrites instead of duplicating
#     7. Save and reload a mistake; find_by_task_class filters correctly
#     8. Save and reload a recipe; best_recipe picks highest critique_score
#
#   Part C – MemoryRetriever
#     9. retrieve() returns a string containing all three sections
#     10. has_recipe() and has_mistakes() return correct booleans
# ---------------------------------------------------------------------------
import os
import tempfile   # creates temporary directories for test files;
                  # deleted automatically after each test

import pytest

from smart_home_langgraph.data.simulator import SmartHomeSimulator
from smart_home_langgraph.memory.store import (
    PreferenceMemory, UserPreference,
    MistakeMemory,    MistakeRecord,
    RecipeMemory,     RecipeRecord,
)
from smart_home_langgraph.memory.retriever import MemoryRetriever


# ===========================================================================
# Part A — SmartHomeSimulator
# ===========================================================================

def test_generate_row_count_and_columns() -> None:
    sim = SmartHomeSimulator(seed=0)
    df  = sim.generate(days=3)

    # 3 days × 24 hours × 4 intervals/hour = 288 rows
    assert len(df) == 288, f"Expected 288 rows, got {len(df)}"

    # All six expected columns must be present.
    expected_cols = {"timestamp", "temperature_c", "humidity_pct",
                     "occupancy", "power_kw", "anomaly"}
    assert expected_cols == set(df.columns)


def test_generate_value_ranges() -> None:
    sim = SmartHomeSimulator(seed=1)
    df  = sim.generate(days=7)

    # Humidity must always be within the clamped range.
    assert df["humidity_pct"].between(20, 90).all(), "Humidity out of range"
    # Power must never be negative.
    assert (df["power_kw"] >= 0).all(), "Negative power values found"
    # Occupancy must only be 0 or 1.
    assert df["occupancy"].isin([0, 1]).all(), "Occupancy has values other than 0/1"


def test_daily_summary_one_row_per_day() -> None:
    sim = SmartHomeSimulator(seed=2)
    df  = sim.generate(days=5)
    summary = sim.daily_summary(df)

    # One summary row per calendar day.
    assert len(summary) == 5
    # Must include the key computed columns.
    assert "total_power_kwh" in summary.columns
    assert "anomaly_count"   in summary.columns


def test_context_window_returns_text() -> None:
    sim  = SmartHomeSimulator(seed=3)
    df   = sim.generate(days=2)
    text = sim.context_window(df, hours=24)

    # Must be a non-empty string with at least the header.
    assert isinstance(text, str)
    assert "Sensor data window" in text


def test_anomaly_rows_present_with_high_probability() -> None:
    sim = SmartHomeSimulator(seed=7)
    # Set anomaly_probability very high so anomalies are almost certain.
    df  = sim.generate(days=3, anomaly_probability=0.5)
    assert df["anomaly"].sum() > 0, "Expected at least one anomaly row"


# ===========================================================================
# Part B — Memory Stores
# ===========================================================================

# pytest fixture: creates a fresh temporary directory for each test.
# This means each test gets its own isolated JSON files and they are cleaned
# up automatically — no leftover files between test runs.
@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)   # tmp_path is a pathlib.Path; convert to string


def test_preference_save_and_reload(tmp_dir) -> None:
    mem  = PreferenceMemory(os.path.join(tmp_dir, "prefs.json"))
    pref = UserPreference(key="temperature_range", value="20-22 Celsius", reason="comfort")
    mem.save(pref)

    loaded = mem.load_all()
    assert len(loaded) == 1
    assert loaded[0].key   == "temperature_range"
    assert loaded[0].value == "20-22 Celsius"


def test_preference_update_overwrites_same_key(tmp_dir) -> None:
    mem = PreferenceMemory(os.path.join(tmp_dir, "prefs.json"))
    mem.save(UserPreference(key="quiet_hours", value="22:00-07:00"))
    mem.save(UserPreference(key="quiet_hours", value="23:00-06:00"))  # update same key

    loaded = mem.load_all()
    # Must still be only ONE entry for this key, with the updated value.
    assert len(loaded) == 1
    assert loaded[0].value == "23:00-06:00"


def test_mistake_save_and_filter_by_task_class(tmp_dir) -> None:
    mem = MistakeMemory(os.path.join(tmp_dir, "mistakes.json"))
    mem.save(MistakeRecord(
        task_class="energy_optimization",
        error_description="Suggested dishwasher at 23:00",
        corrective_rule="Avoid quiet hours 22:00-07:00",
    ))
    mem.save(MistakeRecord(
        task_class="comfort_optimization",
        error_description="Ignored occupancy data",
        corrective_rule="Always check occupancy before recommending HVAC change",
    ))

    energy_mistakes = mem.find_by_task_class("energy_optimization")
    assert len(energy_mistakes) == 1
    assert "dishwasher" in energy_mistakes[0].error_description

    # Different task class should not appear.
    assert len(mem.find_by_task_class("anomaly_explanation")) == 0


def test_recipe_best_picks_highest_score(tmp_dir) -> None:
    mem = RecipeMemory(os.path.join(tmp_dir, "recipes.json"))
    # Save two recipes with different scores.
    mem.save(RecipeRecord(
        task_class="energy_optimization",
        strategy_steps=["check peak hours", "shift load"],
        critique_score=0.75,
    ))
    mem.save(RecipeRecord(
        task_class="energy_optimization",
        strategy_steps=["identify waste", "schedule off-peak", "notify user"],
        critique_score=0.93,
    ))

    best = mem.best_recipe("energy_optimization")
    assert best is not None
    # Must return the one with the higher critique_score.
    assert best.critique_score == 0.93


def test_recipe_returns_none_when_empty(tmp_dir) -> None:
    mem = RecipeMemory(os.path.join(tmp_dir, "recipes.json"))
    # No recipes saved yet → should return None, not raise an error.
    assert mem.best_recipe("energy_optimization") is None


# ===========================================================================
# Part C — MemoryRetriever
# ===========================================================================

def test_retriever_context_contains_all_three_sections(tmp_dir) -> None:
    # Set up three stores with one entry each.
    pref_store = PreferenceMemory(os.path.join(tmp_dir, "prefs.json"))
    pref_store.save(UserPreference(key="quiet_hours", value="22:00-07:00"))

    mistake_store = MistakeMemory(os.path.join(tmp_dir, "mistakes.json"))
    mistake_store.save(MistakeRecord(
        task_class="energy_optimization",
        error_description="Ignored quiet hours",
        corrective_rule="Respect quiet hours",
    ))

    recipe_store = RecipeMemory(os.path.join(tmp_dir, "recipes.json"))
    recipe_store.save(RecipeRecord(
        task_class="energy_optimization",
        strategy_steps=["identify peak usage", "shift to off-peak"],
        critique_score=0.88,
    ))

    retriever = MemoryRetriever(mistake_store, recipe_store, pref_store)
    context   = retriever.retrieve("energy_optimization")

    # The context string must contain markers from all three sections.
    assert "AVOID"          in context   # from mistake section
    assert "Reuse"          in context   # from recipe section
    assert "quiet_hours"    in context   # from preference section


def test_retriever_has_recipe_and_has_mistakes_flags(tmp_dir) -> None:
    pref_store    = PreferenceMemory(os.path.join(tmp_dir, "prefs.json"))
    mistake_store = MistakeMemory(os.path.join(tmp_dir, "mistakes.json"))
    recipe_store  = RecipeMemory(os.path.join(tmp_dir, "recipes.json"))

    retriever = MemoryRetriever(mistake_store, recipe_store, pref_store)

    # Nothing stored yet → both flags should be False.
    assert retriever.has_recipe("energy_optimization")   is False
    assert retriever.has_mistakes("energy_optimization") is False

    # Add a mistake → has_mistakes must flip to True.
    mistake_store.save(MistakeRecord(
        task_class="energy_optimization",
        error_description="Test error",
        corrective_rule="Test fix",
    ))
    assert retriever.has_mistakes("energy_optimization") is True

    # has_recipe must still be False until a recipe is added.
    assert retriever.has_recipe("energy_optimization") is False
