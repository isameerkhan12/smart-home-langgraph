# ---------------------------------------------------------------------------
# data/loader.py
# Purpose: Load telemetry data into Postgres and produce deterministic context.
#
# V1 approach:
#   - Use the provided preprocessed CSV as telemetry source.
#   - Bootstrap a typed telemetry_events table in Postgres.
#   - Compute prompt context from SQL-backed metrics over a recent window.
# ---------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
import os

import pandas as pd
import psycopg

from smart_home_langgraph.config.settings import get_settings


# Path to the telemetry CSV in the data/ folder.
_DATA_FILE = os.path.join(os.path.dirname(__file__), "preprocessed_dataset.csv")


@dataclass(frozen=True)
class TelemetryMetrics:
    row_count: int
    start_time: pd.Timestamp | None
    end_time: pd.Timestamp | None
    avg_energy_kwh: float
    avg_bandwidth: float
    offloading_rate: float
    voltage_drop_rate: float
    high_load_rate: float
    top_combo: str


def _bandwidth_level(avg_bandwidth: float) -> str:
    # Convert raw bandwidth into a coarse category for easier prompting.
    if avg_bandwidth < 15000:
        return "low"
    if avg_bandwidth < 30000:
        return "medium"
    return "high"


class HomeDataLoader:
    """Bootstraps telemetry into Postgres and serves context summaries."""

    def __init__(self, filepath: str = _DATA_FILE) -> None:
        # Keep CSV path configurable for local experiments.
        self._filepath = filepath
        # Avoid repeated bootstrap checks inside the same process.
        self._bootstrapped = False

    def _connect(self) -> psycopg.Connection:
        # Use the same Postgres DSN as the rest of the app so telemetry and
        # memory can live in one database.
        settings = get_settings()
        if not settings.postgres_uri:
            raise RuntimeError("POSTGRES_URI is required for telemetry structured store.")
        return psycopg.connect(settings.postgres_uri)

    def _ensure_table(self, conn: psycopg.Connection) -> None:
        # Create the typed telemetry table once; repeated calls are safe.
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    unix_timestamp BIGINT NOT NULL,
                    transaction_id BIGINT PRIMARY KEY,
                    television SMALLINT NOT NULL,
                    dryer SMALLINT NOT NULL,
                    oven SMALLINT NOT NULL,
                    refrigerator SMALLINT NOT NULL,
                    microwave SMALLINT NOT NULL,
                    line_voltage DOUBLE PRECISION NOT NULL,
                    voltage DOUBLE PRECISION NOT NULL,
                    apparent_power DOUBLE PRECISION NOT NULL,
                    energy_kwh DOUBLE PRECISION NOT NULL,
                    offloading_decision SMALLINT NOT NULL,
                    bandwidth DOUBLE PRECISION NOT NULL,
                    event_time TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_events_event_time ON telemetry_events (event_time)"
            )
        conn.commit()

    def _bootstrap_csv_if_empty(self, conn: psycopg.Connection) -> None:
        # V1 bootstrap strategy: seed Postgres from CSV only when table is empty.
        # This keeps startup simple while preventing duplicate inserts.
        if self._bootstrapped:
            return

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM telemetry_events")
            row_count = cur.fetchone()[0]

        if row_count > 0:
            self._bootstrapped = True
            return

        raw_df = pd.read_csv(self._filepath)
        # Normalize CSV headers to SQL-friendly names used in queries.
        df = raw_df.rename(
            columns={
                "Unix Timestamp": "unix_timestamp",
                "Transaction_ID": "transaction_id",
                "Television": "television",
                "Dryer": "dryer",
                "Oven": "oven",
                "Refrigerator": "refrigerator",
                "Microwave": "microwave",
                "Line Voltage": "line_voltage",
                "Voltage": "voltage",
                "Apparent Power": "apparent_power",
                "Energy Consumption (kWh)": "energy_kwh",
                "Offloading Decision": "offloading_decision",
                "Bandwidth": "bandwidth",
            }
        )

        df["event_time"] = pd.to_datetime(df["unix_timestamp"], unit="s", utc=True)

        numeric_columns = [
            "unix_timestamp",
            "transaction_id",
            "television",
            "dryer",
            "oven",
            "refrigerator",
            "microwave",
            "line_voltage",
            "voltage",
            "apparent_power",
            "energy_kwh",
            "offloading_decision",
            "bandwidth",
        ]
        for col in numeric_columns:
            # Coerce bad values to NaN so they can be dropped deterministically.
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_columns + ["event_time"])

        records = list(
            df[
                [
                    "unix_timestamp",
                    "transaction_id",
                    "television",
                    "dryer",
                    "oven",
                    "refrigerator",
                    "microwave",
                    "line_voltage",
                    "voltage",
                    "apparent_power",
                    "energy_kwh",
                    "offloading_decision",
                    "bandwidth",
                    "event_time",
                ]
            ].itertuples(index=False, name=None)
        )

        with conn.cursor() as cur:
            # transaction_id is the natural key; ON CONFLICT keeps import idempotent.
            cur.executemany(
                """
                INSERT INTO telemetry_events (
                    unix_timestamp,
                    transaction_id,
                    television,
                    dryer,
                    oven,
                    refrigerator,
                    microwave,
                    line_voltage,
                    voltage,
                    apparent_power,
                    energy_kwh,
                    offloading_decision,
                    bandwidth,
                    event_time
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                records,
            )
        conn.commit()
        self._bootstrapped = True

    def _metrics_for_window(self, conn: psycopg.Connection, hours: int) -> TelemetryMetrics:
        # Compute deterministic aggregates from the latest N-hour window.
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH latest AS (
                    SELECT MAX(event_time) AS max_event_time FROM telemetry_events
                )
                SELECT
                    COUNT(*)::INT AS row_count,
                    MIN(event_time) AS start_time,
                    MAX(event_time) AS end_time,
                    AVG(energy_kwh)::DOUBLE PRECISION AS avg_energy_kwh,
                    AVG(bandwidth)::DOUBLE PRECISION AS avg_bandwidth,
                    AVG(offloading_decision)::DOUBLE PRECISION AS offloading_rate,
                    AVG(CASE WHEN (line_voltage - voltage) >= 8 THEN 1.0 ELSE 0.0 END)::DOUBLE PRECISION AS voltage_drop_rate,
                    AVG(CASE WHEN apparent_power >= 1800 THEN 1.0 ELSE 0.0 END)::DOUBLE PRECISION AS high_load_rate
                FROM telemetry_events
                WHERE event_time >= (
                    SELECT max_event_time - (%s * INTERVAL '1 hour') FROM latest
                )
                """,
                (hours,),
            )
            row = cur.fetchone()

            cur.execute(
                """
                WITH latest AS (
                    SELECT MAX(event_time) AS max_event_time FROM telemetry_events
                ),
                windowed AS (
                    SELECT
                        television,
                        dryer,
                        oven,
                        refrigerator,
                        microwave
                    FROM telemetry_events
                    WHERE event_time >= (
                        SELECT max_event_time - (%s * INTERVAL '1 hour') FROM latest
                    )
                ),
                combos AS (
                    SELECT
                        CONCAT(
                            CASE WHEN television = 1 THEN 'Television+' ELSE '' END,
                            CASE WHEN dryer = 1 THEN 'Dryer+' ELSE '' END,
                            CASE WHEN oven = 1 THEN 'Oven+' ELSE '' END,
                            CASE WHEN refrigerator = 1 THEN 'Refrigerator+' ELSE '' END,
                            CASE WHEN microwave = 1 THEN 'Microwave+' ELSE '' END
                        ) AS combo_text
                    FROM windowed
                    WHERE (television + dryer + oven + refrigerator + microwave) >= 2
                )
                SELECT RTRIM(combo_text, '+') AS combo
                FROM combos
                WHERE combo_text <> ''
                GROUP BY combo
                ORDER BY COUNT(*) DESC
                LIMIT 1
                """,
                (hours,),
            )
            combo_row = cur.fetchone()

        if not row or row[0] == 0:
            # Return an explicit zeroed structure so downstream formatting is stable.
            return TelemetryMetrics(
                row_count=0,
                start_time=None,
                end_time=None,
                avg_energy_kwh=0.0,
                avg_bandwidth=0.0,
                offloading_rate=0.0,
                voltage_drop_rate=0.0,
                high_load_rate=0.0,
                top_combo="none",
            )

        return TelemetryMetrics(
            row_count=row[0],
            start_time=row[1],
            end_time=row[2],
            avg_energy_kwh=float(row[3] or 0.0),
            avg_bandwidth=float(row[4] or 0.0),
            offloading_rate=float(row[5] or 0.0),
            voltage_drop_rate=float(row[6] or 0.0),
            high_load_rate=float(row[7] or 0.0),
            top_combo=(combo_row[0] if combo_row and combo_row[0] else "none"),
        )

    def context_window(self, hours: int = 24) -> str:
        """Return deterministic telemetry summary for the most recent window."""
        with self._connect() as conn:
            self._ensure_table(conn)
            self._bootstrap_csv_if_empty(conn)
            metrics = self._metrics_for_window(conn, hours=hours)

        if metrics.row_count == 0:
            return "Telemetry summary unavailable: no rows in telemetry_events."

        # Keep prompt context compact and numeric so the LLM reasons on facts.
        bandwidth_level = _bandwidth_level(metrics.avg_bandwidth)
        start = metrics.start_time.strftime("%Y-%m-%d %H:%M:%S %Z") if metrics.start_time else "n/a"
        end = metrics.end_time.strftime("%Y-%m-%d %H:%M:%S %Z") if metrics.end_time else "n/a"

        return (
            f"Telemetry summary ({hours}h window): {start} to {end}. "
            f"Rows: {metrics.row_count}. "
            f"Avg energy: {metrics.avg_energy_kwh:.2f} kWh. "
            f"Avg bandwidth: {metrics.avg_bandwidth:.0f} ({bandwidth_level}). "
            f"Offloading rate: {metrics.offloading_rate:.2%}. "
            f"High-load apparent-power rate (>=1800): {metrics.high_load_rate:.2%}. "
            f"Voltage-drop rate (line_voltage - voltage >= 8): {metrics.voltage_drop_rate:.2%}. "
            f"Most common multi-appliance combo: {metrics.top_combo}."
        )
