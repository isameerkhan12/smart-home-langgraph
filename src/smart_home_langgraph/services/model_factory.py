# ---------------------------------------------------------------------------
# services/model_factory.py
# Purpose: Shared chat-model construction for configured LLM providers.
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Any


def build_model(
    settings,
    temperature: float,
    structured_output_schema: Any | None = None,
):
    """Build configured provider model, optionally attaching structured output schema."""
    if settings.llm_provider.lower() == "ollama":
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
        )

    if structured_output_schema is not None:
        return model.with_structured_output(structured_output_schema)

    return model