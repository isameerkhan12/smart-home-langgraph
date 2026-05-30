# ---------------------------------------------------------------------------
# services/json_utils.py
# Purpose: Shared helpers for parsing JSON from LLM text responses.
# ---------------------------------------------------------------------------
from __future__ import annotations

import json
import re
from typing import Any, TypeVar


TModel = TypeVar("TModel")


def _parse_json_from_text(text: str) -> Any:
    """Parse JSON from markdown-wrapped JSON block or raw JSON text."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


def result_to_text(result: Any) -> str:
    """Normalize model result content to plain text."""
    return result.content if isinstance(result.content, str) else str(result.content)


def result_to_model(result: Any, schema: type[TModel]) -> TModel:
    """Convert model output to a validated Pydantic model instance."""
    if isinstance(result, schema):
        return result

    if isinstance(result, dict):
        return schema.model_validate(result)

    payload = _parse_json_from_text(result_to_text(result))
    if not isinstance(payload, dict):
        raise ValueError("Model returned non-object JSON")
    return schema.model_validate(payload)