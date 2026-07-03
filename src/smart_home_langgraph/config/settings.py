# ---------------------------------------------------------------------------
# config/settings.py
# Purpose: Load secret keys and configuration from environment variables.
# Why env vars? So we never hardcode secrets in source code that gets
# committed to git. The .env file stays on your machine only.
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass  # dataclass turns a plain class into a structured
                                    # container with auto-generated __init__, __repr__, etc.

from dotenv import load_dotenv     # reads the .env file and loads each line as an
                                    # environment variable so os.getenv() can find them


# frozen=True makes the settings object immutable after creation —
# you cannot accidentally overwrite gemini_api_key later in the code.
@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None  # will be None if the env var is missing
    # Which LLM provider to use for generation/critique/memory extraction.
    llm_provider: str
    # Ollama settings used when LLM_PROVIDER=ollama.
    ollama_model: str
    ollama_base_url: str
    # OpenRouter settings used when LLM_PROVIDER=openrouter.
    openrouter_api_key: str | None
    openrouter_model: str
    # Postgres DSN used by LangGraph PostgresStore for LTM.
    postgres_uri: str | None
    # LangSmith tracing settings.
    langsmith_api_key: str | None
    langsmith_project: str
    langsmith_tracing_enabled: bool
    # Embedding model used for vectorizing memory content.
    gemini_embedding_model: str
    # Embedding vector dimension; must match the selected model.
    gemini_embedding_dims: int


def get_settings() -> Settings:
    """Load runtime settings from environment variables."""
    # load_dotenv() reads .env from the project root and puts each key=value
    # pair into the process environment so os.getenv() can read them.
    load_dotenv()
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        llm_provider=os.getenv("LLM_PROVIDER", "openrouter").strip().lower(),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free"),
        postgres_uri=os.getenv("POSTGRES_URI"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "smart-home-langgraph"),
        langsmith_tracing_enabled=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
        gemini_embedding_dims=int(os.getenv("GEMINI_EMBEDDING_DIMS", "768")),
    )
