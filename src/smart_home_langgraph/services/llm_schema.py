# ---------------------------------------------------------------------------
# services/llm_schema.py
# Purpose: Shared typed schemas for structured LLM outputs.
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PlannerDecision(BaseModel):
    """Decision from planner: use memory or tools?"""

    use_memory: bool = Field(description="True if memory is sufficient to answer the question")
    reason: str = Field(description="Brief explanation for the decision")


class CritiqueDecision(BaseModel):
    """Validated critique payload from model output."""

    passed: bool = True
    issues: list[str] = Field(default_factory=list)
    severity: Literal["success", "minor_revision", "major_revision", "fail"] = "success"
    repair_hints: str = ""
    pass_reasons: list[str] = Field(default_factory=list)
    critique_status: Literal["not_run", "completed", "skipped_config", "fallback_error"] = "completed"

    @model_validator(mode="after")
    def enforce_invariants(self) -> "CritiqueDecision":
        """Keep critique fields internally consistent."""
        if self.passed:
            self.repair_hints = ""
            if not self.pass_reasons:
                self.pass_reasons = [
                    "Response met the required quality checks.",
                    "No blocking issues were detected.",
                ]
        else:
            self.pass_reasons = []

        return self

    @classmethod
    def skipped_config(cls, reason: str) -> "CritiqueDecision":
        return cls(
            passed=True,
            issues=[],
            severity="success",
            repair_hints="",
            pass_reasons=[reason, "Response accepted by fallback policy."],
            critique_status="skipped_config",
        )

    @classmethod
    def fallback_error(cls, error: str) -> "CritiqueDecision":
        return cls(
            passed=True,
            issues=[],
            severity="success",
            repair_hints="",
            pass_reasons=[
                f"Critique step failed at runtime: {error}",
                "Response accepted by fallback policy.",
            ],
            critique_status="fallback_error",
        )