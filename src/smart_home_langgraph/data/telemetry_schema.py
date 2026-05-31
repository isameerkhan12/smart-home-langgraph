from __future__ import annotations

from dataclasses import dataclass

# Source CSV -> canonical telemetry schema mapping.
# Update this dictionary if CSV headers evolve.
SOURCE_TO_CANONICAL = {
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

SOURCE_COLUMNS = tuple(SOURCE_TO_CANONICAL.keys())
CANONICAL_COLUMNS = tuple(SOURCE_TO_CANONICAL.values())
# Canonical columns plus derived event timestamp used for DB insert order.
CANONICAL_COLUMNS_WITH_EVENT = (*CANONICAL_COLUMNS, "event_time")
# Canonical columns expected to be numeric; excludes derived event_time.
NUMERIC_COLUMNS = CANONICAL_COLUMNS

TELEMETRY_EVENTS_DDL = """
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

INGEST_META_DDL = """
CREATE TABLE IF NOT EXISTS telemetry_ingest_meta (
    source_path TEXT PRIMARY KEY,
    file_mtime_ns BIGINT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    last_synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

INSERT_COLUMNS_SQL = ",\n                    ".join(CANONICAL_COLUMNS_WITH_EVENT)
INSERT_VALUES_SQL = ", ".join(["%s"] * len(CANONICAL_COLUMNS_WITH_EVENT))


@dataclass(frozen=True)
class TelemetryMetrics:
    row_count: int
    start_time: object | None
    end_time: object | None
    avg_energy_kwh: float
    avg_bandwidth: float
    offloading_rate: float
    voltage_drop_rate: float
    high_load_rate: float
    top_combo: str


def bandwidth_level(avg_bandwidth: float) -> str:
    """Convert raw bandwidth into coarse levels for prompt readability."""
    if avg_bandwidth < 15000:
        return "low"
    if avg_bandwidth < 30000:
        return "medium"
    return "high"
