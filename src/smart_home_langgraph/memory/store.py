# ---------------------------------------------------------------------------
# memory/store.py  –  Phase 2: Long-Term Memory Stores
#
# Purpose:
#   Persist three types of long-term memory to JSON files on disk so the
#   agent can remember things across separate runs (sessions).
#
# Why JSON files?
#   Simple, human-readable, no database setup required.
#   In a later version this can be swapped for SQLite or a vector DB
#   without changing the interface — the rest of the code just calls
#   the same save/load methods.
#
# The three memory stores and what they hold:
#
#   1. PreferenceMemory  — what the user likes / dislikes
#      Examples: preferred temperature range, quiet hours, priority devices.
#      Requirement: "agent learns from previous preferences."
#
#   2. MistakeMemory  — errors the agent made and how to fix them
#      Fields: task_class, error_description, corrective_rule, confidence.
#      Requirement: "agent should not make mistakes after some experience."
#
#   3. RecipeMemory  — successful solution strategies worth reusing
#      Fields: task_class, strategy_steps, critique_score, conditions.
#      Requirement: "use previous knowledge (recipe) that it had used previously."
#
# How the retriever (retriever.py) uses these stores:
#   Priority order at generation time:
#     1. MistakeMemory  → avoid known errors first
#     2. RecipeMemory   → reuse proven approach
#     3. PreferenceMemory → personalise the answer
# ---------------------------------------------------------------------------
from __future__ import annotations

import json                         # read/write JSON files
import os                           # file path operations
from dataclasses import dataclass, field, asdict  # structured data containers
from typing import Optional


# ---------------------------------------------------------------------------
# Helper: _load_json / _save_json
# Low-level read/write so individual store classes don't repeat this logic.
# ---------------------------------------------------------------------------

def _load_json(path: str) -> list[dict]:
    """Read a JSON file and return its contents as a list of dicts.
    Returns an empty list if the file does not exist yet."""
    if not os.path.exists(path):
        return []   # first run — no data yet, that's fine
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)   # json.load parses the file into Python objects


def _save_json(path: str, data: list[dict]) -> None:
    """Write a list of dicts to a JSON file, creating it if needed."""
    # indent=2 makes the file human-readable (not one long line)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ===========================================================================
# 1. PreferenceMemory
#    Stores user preferences that should influence every generated response.
# ===========================================================================

@dataclass
class UserPreference:
    """One user preference entry."""
    key: str           # what the preference is about, e.g. "temperature_range"
    value: str         # the preference itself, e.g. "20-22 Celsius"
    reason: str = ""   # optional: why the user stated this preference


class PreferenceMemory:
    """
    Persists user preferences to a JSON file.

    Usage:
        mem = PreferenceMemory("data/preferences.json")
        mem.save(UserPreference(key="quiet_hours", value="22:00-07:00"))
        prefs = mem.load_all()
    """

    def __init__(self, path: str) -> None:
        # path is the file where preferences are stored, e.g. "memory/preferences.json"
        self._path = path

    def save(self, preference: UserPreference) -> None:
        """Add or update a preference. If the same key already exists, overwrite it."""
        records = _load_json(self._path)

        # Check if this key already exists and update it rather than duplicating.
        for i, rec in enumerate(records):
            if rec["key"] == preference.key:
                records[i] = asdict(preference)   # asdict() converts dataclass → dict
                _save_json(self._path, records)
                return

        # Key not found → append as a new entry.
        records.append(asdict(preference))
        _save_json(self._path, records)

    def load_all(self) -> list[UserPreference]:
        """Return every stored preference as a list of UserPreference objects."""
        records = _load_json(self._path)
        # dict(**rec) unpacks the dict back into the dataclass constructor.
        return [UserPreference(**rec) for rec in records]

    def as_text(self) -> str:
        """Return all preferences as a single readable string for LLM prompts."""
        prefs = self.load_all()
        if not prefs:
            return "No user preferences stored yet."
        lines = [f"- {p.key}: {p.value}" + (f" ({p.reason})" if p.reason else "")
                 for p in prefs]
        return "User preferences:\n" + "\n".join(lines)


# ===========================================================================
# 2. MistakeMemory
#    Records errors the agent made and the corrective rule to apply next time.
# ===========================================================================

@dataclass
class MistakeRecord:
    """One mistake the agent made on a specific task type."""
    task_class: str          # which intent this mistake occurred on
    error_description: str   # what went wrong (from critique node output)
    corrective_rule: str     # what to do differently next time
    confidence: float = 1.0  # how certain we are this rule is correct (0.0-1.0)
                              # starts at 1.0; can be lowered if rule is uncertain


class MistakeMemory:
    """
    Persists past mistakes and their fixes so the agent can avoid repeating them.

    Usage:
        mem = MistakeMemory("data/mistakes.json")
        mem.save(MistakeRecord(
            task_class="energy_optimization",
            error_description="Recommended running dishwasher at 23:00 (quiet hours)",
            corrective_rule="Never suggest appliance use during quiet hours 22:00-07:00",
        ))
        mistakes = mem.find_by_task_class("energy_optimization")
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def save(self, record: MistakeRecord) -> None:
        """Append a new mistake record. Duplicates are allowed (same error can recur)."""
        records = _load_json(self._path)
        records.append(asdict(record))
        _save_json(self._path, records)

    def load_all(self) -> list[MistakeRecord]:
        """Return all stored mistakes."""
        return [MistakeRecord(**r) for r in _load_json(self._path)]

    def find_by_task_class(self, task_class: str) -> list[MistakeRecord]:
        """Return only mistakes that belong to the given intent/task class."""
        # List comprehension: loop over all mistakes, keep only matching ones.
        return [m for m in self.load_all() if m.task_class == task_class]

    def as_text(self, task_class: str) -> str:
        """
        Return relevant mistakes as a string to inject into the LLM prompt.
        This tells the LLM "here are errors you made before — avoid them".
        """
        mistakes = self.find_by_task_class(task_class)
        if not mistakes:
            return f"No known mistakes for task class '{task_class}'."
        lines = [f"- AVOID: {m.error_description} → Rule: {m.corrective_rule}"
                 for m in mistakes]
        return f"Known mistakes to avoid ({task_class}):\n" + "\n".join(lines)


# ===========================================================================
# 3. RecipeMemory
#    Stores successful solution strategies so the agent can reuse them.
# ===========================================================================

@dataclass
class RecipeRecord:
    """One successful solution recipe for a given task type."""
    task_class: str              # which intent this recipe applies to
    strategy_steps: list[str]   # ordered list of steps the agent took
    critique_score: float        # quality score from critique node (0.0-1.0); higher is better
    conditions: str = ""         # optional: when this recipe is valid
                                 # e.g. "applies when occupancy > 4h and power > 3 kWh"


class RecipeMemory:
    """
    Persists successful solution strategies so the agent can reuse them
    instead of generating a brand-new plan from scratch every time.

    Usage:
        mem = RecipeMemory("data/recipes.json")
        mem.save(RecipeRecord(
            task_class="energy_optimization",
            strategy_steps=["identify peak hours", "shift dishwasher to 06:00"],
            critique_score=0.92,
        ))
        best = mem.best_recipe("energy_optimization")
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def save(self, record: RecipeRecord) -> None:
        """Append a successful recipe."""
        records = _load_json(self._path)
        records.append(asdict(record))
        _save_json(self._path, records)

    def load_all(self) -> list[RecipeRecord]:
        """Return all stored recipes."""
        return [RecipeRecord(**r) for r in _load_json(self._path)]

    def find_by_task_class(self, task_class: str) -> list[RecipeRecord]:
        """Return all recipes for the given task class."""
        return [r for r in self.load_all() if r.task_class == task_class]

    def best_recipe(self, task_class: str) -> Optional[RecipeRecord]:
        """
        Return the highest-scoring recipe for the given task class.
        Returns None if no recipe exists yet.
        The critique_score field (0-1) is used to rank recipes.
        """
        candidates = self.find_by_task_class(task_class)
        if not candidates:
            return None
        # max() with key= picks the recipe with the highest critique_score.
        return max(candidates, key=lambda r: r.critique_score)

    def as_text(self, task_class: str) -> str:
        """
        Return the best recipe as a string to inject into the LLM prompt.
        This tells the LLM "here is a strategy that worked before — prefer it".
        """
        recipe = self.best_recipe(task_class)
        if recipe is None:
            return f"No stored recipe for task class '{task_class}'."
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(recipe.strategy_steps))
        return (
            f"Reuse this proven recipe (score={recipe.critique_score}):\n"
            + steps
            + (f"\n  Conditions: {recipe.conditions}" if recipe.conditions else "")
        )
