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
    # Postgres DSN used by LangGraph PostgresStore for LTM.
    postgres_uri: str | None
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
        postgres_uri=os.getenv("POSTGRES_URI"),
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
        gemini_embedding_dims=int(os.getenv("GEMINI_EMBEDDING_DIMS", "768")),
    )
