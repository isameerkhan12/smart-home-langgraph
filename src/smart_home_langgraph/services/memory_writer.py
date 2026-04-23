# ---------------------------------------------------------------------------
# services/memory_writer.py
#
# Purpose:
#   Analyze critique results and write learnings to long-term memory.
#
# Strategy:
#   1. If critique FAILED: write to MistakeMemory (what went wrong + fix)
#   2. If critique PASSED: write to RecipeMemory (what worked + success score)
#   3. Track which memories were written for metrics
# ---------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass

from smart_home_langgraph.graph.state import AgentState
from smart_home_langgraph.memory.store import (
    MistakeMemory,
    MistakeRecord,
    RecipeMemory,
    RecipeRecord,
)


@dataclass
class LearningOutcome:
    """Tracks what was learned from this episode."""

    # Number of records written to MistakeMemory.
    mistakes_written: int

    # Number of records written to RecipeMemory.
    recipes_written: int

    # True if any mistakes were detected (critique failed).
    had_critique_failures: bool

    # True if critique passed on first attempt (no repairs).
    critique_first_pass: bool

    # Repair count at end of episode.
    final_repair_count: int


def write_learnings(
    state: AgentState,
    mistake_store: MistakeMemory,
    recipe_store: RecipeMemory,
) -> LearningOutcome:
    """
    Write learnings from this episode to memory stores.

    Logic:
      - If critique FAILED: extract issue descriptions and write to MistakeMemory
      - If critique PASSED: write successful strategy to RecipeMemory
      - Track counts for metric computation
    """
    critique = state["critique_result"]
    intent = state["intent"]
    response = state["response"]

    mistakes_written = 0
    recipes_written = 0
    critique_first_pass = state["repair_count"] == 0 and critique["passed"]

    # If critique failed, write what went wrong to MistakeMemory.
    if not critique["passed"]:
        for issue in critique["issues"]:
            mistake_record = MistakeRecord(
                task_class=intent,
                error_description=issue,
                corrective_rule=critique["repair_hints"],
                confidence=1.0,
            )
            try:
                mistake_store.save(mistake_record)
                mistakes_written += 1
            except Exception as exc:  # noqa: BLE001 - graceful fallback
                # Log but don't crash; learning is optional.
                print(f"Warning: Failed to write mistake: {exc}")

    # If critique eventually PASSED, write successful strategy to RecipeMemory.
    if critique["passed"]:
        # Split response into steps (one per bullet point or sentence).
        steps = [line.strip() for line in response.split("\n") if line.strip()]
        # If no line breaks, try splitting by periods or use whole response as one step.
        if len(steps) <= 1:
            steps = [response]

        # Compute success score: higher for first-pass, lower for repairs.
        success_score = 0.9 if state["repair_count"] == 0 else 0.7

        recipe_record = RecipeRecord(
            task_class=intent,
            strategy_steps=steps,
            critique_score=success_score,
            conditions="",
        )
        try:
            recipe_store.save(recipe_record)
            recipes_written += 1
        except Exception as exc:  # noqa: BLE001 - graceful fallback
            print(f"Warning: Failed to write recipe: {exc}")

    return LearningOutcome(
        mistakes_written=mistakes_written,
        recipes_written=recipes_written,
        had_critique_failures=not critique["passed"],
        critique_first_pass=critique_first_pass,
        final_repair_count=state["repair_count"],
    )
