from __future__ import annotations

import os

import pandas as pd
import psycopg

from smart_home_langgraph.data.telemetry_schema import (
    CANONICAL_COLUMNS_WITH_EVENT,
    INGEST_META_DDL,
    INSERT_COLUMNS_SQL,
    INSERT_VALUES_SQL,
    NUMERIC_COLUMNS,
    SOURCE_COLUMNS,
    SOURCE_TO_CANONICAL,
    TELEMETRY_EVENTS_DDL,
)


def ensure_tables(conn: psycopg.Connection) -> None:
    """Create telemetry and ingest metadata tables; safe to call repeatedly."""
    with conn.cursor() as cur:
        cur.execute(TELEMETRY_EVENTS_DDL)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_events_event_time ON telemetry_events (event_time)"
        )
        cur.execute(INGEST_META_DDL)
    conn.commit()


def _file_signature(csv_path: str) -> tuple[int, int]:
    """Return a cheap file fingerprint (mtime and size) for change detection."""
    stat = os.stat(csv_path)
    return stat.st_mtime_ns, stat.st_size


def sync_csv_if_changed(conn: psycopg.Connection, csv_path: str, already_checked: bool) -> bool:
    """
    Incremental sync:
    - Skip parse if metadata says file unchanged.
    - Parse + upsert when changed.

    Returns True when caller can mark sync as checked for this process.
    """
    if already_checked:
        return True

    current_mtime_ns, current_size = _file_signature(csv_path)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT file_mtime_ns, file_size_bytes
            FROM telemetry_ingest_meta
            WHERE source_path = %s
            """,
            (csv_path,),
        )
        saved = cur.fetchone()

    if saved and saved[0] == current_mtime_ns and saved[1] == current_size:
        return True

    try:
        raw_df = pd.read_csv(csv_path, usecols=SOURCE_COLUMNS)
    except ValueError as exc:
        raise ValueError(
            "CSV schema mismatch. Update SOURCE_TO_CANONICAL mapping in telemetry_schema.py. "
            f"Details: {exc}"
        ) from exc

    df = raw_df.rename(columns=SOURCE_TO_CANONICAL)
    df["event_time"] = pd.to_datetime(df["unix_timestamp"], unit="s", utc=True)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[*NUMERIC_COLUMNS, "event_time"])
    records = list(df[list(CANONICAL_COLUMNS_WITH_EVENT)].itertuples(index=False, name=None))

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO telemetry_events (
                {INSERT_COLUMNS_SQL}
            )
            VALUES ({INSERT_VALUES_SQL})
            ON CONFLICT (transaction_id) DO NOTHING
            """,
            records,
        )
        cur.execute(
            """
            INSERT INTO telemetry_ingest_meta (
                source_path,
                file_mtime_ns,
                file_size_bytes,
                last_synced_at
            )
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (source_path)
            DO UPDATE SET
                file_mtime_ns = EXCLUDED.file_mtime_ns,
                file_size_bytes = EXCLUDED.file_size_bytes,
                last_synced_at = NOW()
            """,
            (csv_path, current_mtime_ns, current_size),
        )
    conn.commit()

    return True
