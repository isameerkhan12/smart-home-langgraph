# ---------------------------------------------------------------------------
# data/loader.py
# Purpose: Thin facade used by workflow to fetch telemetry-backed context.
# ---------------------------------------------------------------------------
from __future__ import annotations

import os

import psycopg

from smart_home_langgraph.config.settings import get_settings
from smart_home_langgraph.data.telemetry_queries import format_context_window, get_metrics_for_window
from smart_home_langgraph.data.telemetry_store import ensure_tables, sync_csv_if_changed


# Path to the telemetry CSV in the data/ folder.
_DATA_FILE = os.path.join(os.path.dirname(__file__), "preprocessed_dataset.csv")


class HomeDataLoader:
    """Workflow-facing telemetry context loader."""

    def __init__(self, filepath: str = _DATA_FILE) -> None:
        self._filepath = filepath
        self._sync_checked = False

    def _connect(self) -> psycopg.Connection:
        settings = get_settings()
        if not settings.postgres_uri:
            raise RuntimeError("POSTGRES_URI is required for telemetry structured store.")
        return psycopg.connect(settings.postgres_uri)

    def context_window(self, hours: int = 24) -> str:
        """Return telemetry summary text for the most recent time window."""
        with self._connect() as conn:
            ensure_tables(conn)
            self._sync_checked = sync_csv_if_changed(
                conn=conn,
                csv_path=self._filepath,
                already_checked=self._sync_checked,
            )
            metrics = get_metrics_for_window(conn=conn, hours=hours)
        return format_context_window(metrics=metrics, hours=hours)
