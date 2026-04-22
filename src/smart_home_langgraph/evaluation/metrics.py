# ---------------------------------------------------------------------------
# evaluation/metrics.py  –  Phase 1: Metrics Tracker
#
# Purpose:
#   Keep a running record of each evaluation episode and compute the four
#   success metrics defined in scope.yaml.
#
# How it fits into the project:
#   - Phase 1 (now): We just define and track the metrics.
#   - Phase 4+: The critique and memory nodes will write real values here
#     so we can measure whether the agent is actually improving.
#
# What is an "episode"?
#   One complete run of the graph for a single user query.
#   Example: user asks "Why is my power bill high?" -> graph runs -> response
#   produced -> critique result recorded -> episode stored here.
# ---------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field  # dataclass = auto-generated __init__, repr, etc.
                                           # field() lets us give mutable defaults safely
from typing import Optional

import yaml  # used to load scope.yaml so we can validate against defined task classes
import os


# ---------------------------------------------------------------------------
# EpisodeRecord
# One record per graph run. Fill in each field as the graph produces results.
# ---------------------------------------------------------------------------
@dataclass
class EpisodeRecord:
    episode_id: int               # sequential number, e.g. 1, 2, 3 …
    task_class: str               # intent label from detect_intent, e.g. "energy_optimization"
    critique_passed_first_try: bool   # True = no repair needed; False = at least one retry was used
    repeated_known_mistake: bool      # True = mistake memory flagged this as a repeat error
    preferences_respected: bool       # True = response complied with all stored user preferences
    used_existing_recipe: bool        # True = agent reused a past successful recipe


# ---------------------------------------------------------------------------
# MetricsTracker
# Collects EpisodeRecords and computes summary statistics.
# ---------------------------------------------------------------------------
@dataclass
class MetricsTracker:
    # episodes is a list; field(default_factory=list) means each new
    # MetricsTracker instance gets its OWN empty list (avoids the classic
    # Python mutable-default-argument bug).
    episodes: list[EpisodeRecord] = field(default_factory=list)

    def add_episode(self, record: EpisodeRecord) -> None:
        """Append one completed episode to the tracker."""
        self.episodes.append(record)

    # ------------------------------------------------------------------
    # The four metrics from scope.yaml, computed as percentages (0-100).
    # Each method returns None if there are no episodes yet.
    # ------------------------------------------------------------------

    def critique_first_pass_rate(self) -> Optional[float]:
        """Percentage of episodes where the critique node passed on the first try."""
        if not self.episodes:
            return None  # not enough data yet
        # sum() counts the True values (Python treats True as 1, False as 0)
        passed = sum(e.critique_passed_first_try for e in self.episodes)
        return round(passed / len(self.episodes) * 100, 1)

    def repeated_mistake_rate(self) -> Optional[float]:
        """Percentage of episodes where the agent repeated a previously known mistake."""
        if not self.episodes:
            return None
        repeated = sum(e.repeated_known_mistake for e in self.episodes)
        return round(repeated / len(self.episodes) * 100, 1)

    def preference_adherence_rate(self) -> Optional[float]:
        """Percentage of episodes where user preferences were fully respected."""
        if not self.episodes:
            return None
        respected = sum(e.preferences_respected for e in self.episodes)
        return round(respected / len(self.episodes) * 100, 1)

    def recipe_reuse_rate(self) -> Optional[float]:
        """Percentage of episodes where the agent reused a stored successful recipe."""
        if not self.episodes:
            return None
        reused = sum(e.used_existing_recipe for e in self.episodes)
        return round(reused / len(self.episodes) * 100, 1)

    def summary(self) -> dict:
        """Return all four metrics as a single dictionary for easy printing or logging."""
        return {
            "total_episodes": len(self.episodes),
            # target: increasing over time
            "critique_first_pass_rate_%": self.critique_first_pass_rate(),
            # target: decreasing over time
            "repeated_mistake_rate_%": self.repeated_mistake_rate(),
            # target: increasing over time
            "preference_adherence_rate_%": self.preference_adherence_rate(),
            # target: increasing over time
            "recipe_reuse_rate_%": self.recipe_reuse_rate(),
        }


# ---------------------------------------------------------------------------
# load_scope()
# Utility to read scope.yaml and return it as a plain Python dictionary.
# Used by tests and future validation code to check task class names, etc.
# ---------------------------------------------------------------------------
def load_scope() -> dict:
    """Load and return the contents of scope.yaml as a dictionary."""
    # __file__ is the absolute path of THIS file (metrics.py).
    # We navigate one level up to find scope.yaml in the same folder.
    scope_path = os.path.join(os.path.dirname(__file__), "scope.yaml")
    with open(scope_path, "r", encoding="utf-8") as f:
        # yaml.safe_load parses YAML into Python dicts/lists.
        # safe_load (not load) avoids executing arbitrary Python in the YAML.
        return yaml.safe_load(f)
