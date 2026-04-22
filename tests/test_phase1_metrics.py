# ---------------------------------------------------------------------------
# tests/test_phase1_metrics.py  –  Phase 1 Tests
#
# What we test here:
#   1. scope.yaml loads correctly and contains expected keys.
#   2. MetricsTracker computes all four metrics accurately.
#   3. Metrics return None when no episodes exist yet (guard against /0 error).
# ---------------------------------------------------------------------------
from smart_home_langgraph.evaluation.metrics import (
    EpisodeRecord,    # the per-run data container
    MetricsTracker,   # the collector that computes percentages
    load_scope,       # utility that reads scope.yaml
)


# ---- scope.yaml tests -----------------------------------------------------

def test_scope_loads_and_has_expected_task_classes() -> None:
    scope = load_scope()
    # The YAML must define at least these four task classes (matching workflow.py intents).
    expected = {"energy_optimization", "comfort_optimization", "anomaly_explanation", "general_question"}
    actual = set(scope["in_scope_task_classes"])
    assert expected == actual, f"Unexpected task classes: {actual}"


def test_scope_has_all_four_metrics_defined() -> None:
    scope = load_scope()
    required_metrics = {
        "critique_first_pass_rate",
        "repeated_mistake_rate",
        "preference_adherence_rate",
        "recipe_reuse_rate",
    }
    assert required_metrics == set(scope["success_metrics"].keys())


# ---- MetricsTracker tests --------------------------------------------------

def test_metrics_return_none_when_no_episodes() -> None:
    tracker = MetricsTracker()  # empty tracker, no episodes yet
    # All four metrics must return None safely (no ZeroDivisionError).
    assert tracker.critique_first_pass_rate() is None
    assert tracker.repeated_mistake_rate() is None
    assert tracker.preference_adherence_rate() is None
    assert tracker.recipe_reuse_rate() is None


def test_metrics_compute_correctly_with_known_episodes() -> None:
    tracker = MetricsTracker()

    # Episode 1: perfect run — critique passed first try, no repeat mistake,
    #            preferences respected, used an existing recipe.
    tracker.add_episode(EpisodeRecord(
        episode_id=1,
        task_class="energy_optimization",
        critique_passed_first_try=True,
        repeated_known_mistake=False,
        preferences_respected=True,
        used_existing_recipe=True,
    ))

    # Episode 2: a bad run — critique failed, agent repeated a mistake,
    #            ignored preferences, had to generate a new plan.
    tracker.add_episode(EpisodeRecord(
        episode_id=2,
        task_class="comfort_optimization",
        critique_passed_first_try=False,
        repeated_known_mistake=True,
        preferences_respected=False,
        used_existing_recipe=False,
    ))

    # With 1 out of 2 episodes being "good" each metric should be 50.0%.
    assert tracker.critique_first_pass_rate() == 50.0
    assert tracker.repeated_mistake_rate() == 50.0     # 50% of runs had a repeat mistake
    assert tracker.preference_adherence_rate() == 50.0
    assert tracker.recipe_reuse_rate() == 50.0

    # summary() must include all four keys plus total episode count.
    summary = tracker.summary()
    assert summary["total_episodes"] == 2
    assert "critique_first_pass_rate_%" in summary
    assert "repeated_mistake_rate_%" in summary
    assert "preference_adherence_rate_%" in summary
    assert "recipe_reuse_rate_%" in summary
