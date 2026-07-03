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
    provider = settings.llm_provider.lower()
    
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )
    elif provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter."
            )

        model = ChatOpenRouter(
            model_name=settings.openrouter_model,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            app_url="https://github.com/smart-home-langgraph",
            app_title="Smart Home LangGraph",
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
            "Supported providers: openrouter, ollama."
        )

    if structured_output_schema is not None:
        return model.with_structured_output(structured_output_schema)

    return model