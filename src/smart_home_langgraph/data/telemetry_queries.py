from __future__ import annotations

import psycopg

from smart_home_langgraph.data.telemetry_schema import TelemetryMetrics, bandwidth_level


def get_metrics_for_window(conn: psycopg.Connection, hours: int) -> TelemetryMetrics:
    """Compute deterministic telemetry aggregates over the latest N-hour window."""
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


def format_context_window(metrics: TelemetryMetrics, hours: int) -> str:
    """Format telemetry metrics into compact model-facing context text."""
    if metrics.row_count == 0:
        return "Telemetry summary unavailable: no rows in telemetry_events."

    start = metrics.start_time.strftime("%Y-%m-%d %H:%M:%S %Z") if metrics.start_time else "n/a"
    end = metrics.end_time.strftime("%Y-%m-%d %H:%M:%S %Z") if metrics.end_time else "n/a"

    return (
        f"Telemetry summary ({hours}h window): {start} to {end}. "
        f"Rows: {metrics.row_count}. "
        f"Avg energy: {metrics.avg_energy_kwh:.2f} kWh. "
        f"Avg bandwidth: {metrics.avg_bandwidth:.0f} ({bandwidth_level(metrics.avg_bandwidth)}). "
        f"Offloading rate: {metrics.offloading_rate:.2%}. "
        f"High-load apparent-power rate (>=1800): {metrics.high_load_rate:.2%}. "
        f"Voltage-drop rate (line_voltage - voltage >= 8): {metrics.voltage_drop_rate:.2%}. "
        f"Most common multi-appliance combo: {metrics.top_combo}."
    )
