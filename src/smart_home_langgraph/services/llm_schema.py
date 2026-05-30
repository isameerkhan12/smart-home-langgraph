# ---------------------------------------------------------------------------
# services/llm_schema.py
# Purpose: Shared typed schemas for structured LLM outputs.
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CritiqueDecision(BaseModel):
    """Validated critique payload from model output."""

    passed: bool = True
    issues: list[str] = Field(default_factory=list)
    severity: Literal["critical", "medium", "low"] = "low"
    repair_hints: str = ""