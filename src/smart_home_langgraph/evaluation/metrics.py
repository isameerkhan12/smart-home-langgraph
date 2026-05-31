# ---------------------------------------------------------------------------
# evaluation/metrics.py
#
# Purpose:
#   Define the EpisodeRecord dataclass for tracking evaluation metrics.
# ---------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeRecord:
    """One record per graph run. Fill in each field as the graph produces results."""
    episode_id: int                   # sequential number, e.g. 1, 2, 3 …
    task_class: str                   # intent label, e.g. "energy_optimization"
    critique_passed_first_try: bool   # True = no repair needed
    repeated_known_mistake: bool      # True = mistake memory flagged this as a repeat
    preferences_respected: bool       # True = response complied with user preferences
    used_existing_recipe: bool        # True = agent reused a past successful recipe
