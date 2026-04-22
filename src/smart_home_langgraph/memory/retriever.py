# ---------------------------------------------------------------------------
# memory/retriever.py  –  Phase 2: Similar-Task Retriever
#
# Purpose:
#   Before the LLM generates a response, query all three memory stores and
#   assemble a single "memory context" string to inject into the prompt.
#
# Retrieval priority (most important first):
#   1. MistakeMemory   → what to AVOID (safety first)
#   2. RecipeMemory    → what WORKED before (reuse over reinvention)
#   3. PreferenceMemory → what the USER LIKES (personalisation)
#
# Why this order?
#   Avoiding a known error is more urgent than reusing a recipe.
#   Reusing a recipe is more useful than starting from scratch.
#   Personalisation refines the final answer.
#
# Similarity strategy in Phase 2:
#   Simple keyword matching on task_class (the intent label).
#   Phase 5 will upgrade this to embedding-based semantic similarity so
#   "reduce electricity cost" and "lower power bill" are treated as the same.
# ---------------------------------------------------------------------------
from __future__ import annotations

from smart_home_langgraph.memory.store import (
    MistakeMemory,       # errors + corrective rules
    RecipeMemory,        # successful strategy templates
    PreferenceMemory,    # user likes/dislikes
)


class MemoryRetriever:
    """
    Queries all three memory stores and returns a combined context string
    ready to be injected into an LLM prompt.

    Usage:
        retriever = MemoryRetriever(
            mistake_store=MistakeMemory("data/mistakes.json"),
            recipe_store=RecipeMemory("data/recipes.json"),
            preference_store=PreferenceMemory("data/preferences.json"),
        )
        context = retriever.retrieve(task_class="energy_optimization")
        # → pass `context` to the LLM so it knows what to avoid and reuse
    """

    def __init__(
        self,
        mistake_store: MistakeMemory,
        recipe_store: RecipeMemory,
        preference_store: PreferenceMemory,
    ) -> None:
        # Store references to the three memory objects.
        # We do NOT create them here — the caller creates and passes them in
        # (this pattern is called "dependency injection" and makes testing easier
        # because tests can pass in stores backed by temp files).
        self._mistakes   = mistake_store
        self._recipes    = recipe_store
        self._preferences = preference_store

    def retrieve(self, task_class: str) -> str:
        """
        Assemble a memory context string for the given task class.

        The string contains three sections, separated by blank lines:
          Section 1 — mistakes to avoid (from MistakeMemory)
          Section 2 — best proven recipe (from RecipeMemory)
          Section 3 — user preferences (from PreferenceMemory)

        This string is pasted into the LLM prompt in Phase 3 so the model
        "knows" what went wrong before, what worked before, and what the user likes.
        """
        # Each .as_text() call returns either a useful string or a polite
        # "nothing stored yet" message — never raises an exception on empty stores.
        mistakes_text    = self._mistakes.as_text(task_class)
        recipe_text      = self._recipes.as_text(task_class)
        preferences_text = self._preferences.as_text()

        # Combine into one block with clear section headers.
        context = (
            "=== Memory Context ===\n\n"
            f"{mistakes_text}\n\n"     # section 1: what to avoid
            f"{recipe_text}\n\n"       # section 2: what worked before
            f"{preferences_text}"      # section 3: personalisation
        )

        return context

    def has_recipe(self, task_class: str) -> bool:
        """
        Return True if a successful recipe exists for this task class.
        The generate node uses this to decide whether to reuse or create fresh.
        """
        return self._recipes.best_recipe(task_class) is not None

    def has_mistakes(self, task_class: str) -> bool:
        """
        Return True if any known mistakes exist for this task class.
        The generate node uses this to know whether to apply avoidance rules.
        """
        return len(self._mistakes.find_by_task_class(task_class)) > 0
