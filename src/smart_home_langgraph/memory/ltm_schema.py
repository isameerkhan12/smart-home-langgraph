# ---------------------------------------------------------------------------
# memory/ltm_schema.py
# Purpose: Typed long-term memory schema for extractor output and storage.
# ---------------------------------------------------------------------------
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Supported long-term memory types."""

    PREFERENCE = "preference"
    RECIPE = "recipe"
    MISTAKE = "mistake"
    GENERAL = "general"


class MemoryItem(BaseModel):
    """One extracted memory item."""

    # Atomic sentence to persist and retrieve later via semantic search.
    text: str = Field(description="Atomic memory sentence")
    # Typed memory enables downstream filtering/routing (preference/recipe/etc.).
    memory_type: MemoryType = Field(description="Type of the memory")
    # Extractor marks duplicates as not new so writer can skip them.
    is_new: bool = Field(description="True if new, false if duplicate")
    # Optional structured tags (intent, confidence, source turn, etc.).
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryDecision(BaseModel):
    """Extractor decision output."""

    # Gate to avoid writing empty or low-value memory updates.
    should_write: bool
    # Candidate memory items extracted from the current interaction.
    memories: List[MemoryItem] = Field(default_factory=list)
