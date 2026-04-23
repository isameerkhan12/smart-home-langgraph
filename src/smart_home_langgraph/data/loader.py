# ---------------------------------------------------------------------------
# data/loader.py
# Purpose: Load smart-home sensor readings from a dummy Excel file.
#
# Why Excel instead of generating data?
#   Easier to understand and edit — just open home_data.xlsx in Excel or
#   LibreOffice, change some numbers, and the agent will use the new values.
#
# The Excel file has one row per hour with these columns:
#   timestamp      – date and time of the reading
#   temperature_c  – indoor temperature in Celsius
#   humidity_pct   – indoor relative humidity (%)
#   occupancy      – 1 if someone is home, 0 if empty
#   power_kw       – total appliance power draw in kilowatts
#   anomaly        – 1 if the reading looks unusual, 0 otherwise
#
# How to use:
#   loader = HomeDataLoader()
#   print(loader.context_window(hours=24))
# ---------------------------------------------------------------------------
from __future__ import annotations

import os

import pandas as pd

# Path to the Excel file, sitting next to this file in the data/ folder.
_DATA_FILE = os.path.join(os.path.dirname(__file__), "home_data.xlsx")


class HomeDataLoader:
    """Reads smart-home sensor data from an Excel file."""

    def __init__(self, filepath: str = _DATA_FILE) -> None:
        self._filepath = filepath

    def load(self) -> pd.DataFrame:
        """Return the full dataset as a DataFrame."""
        return pd.read_excel(self._filepath, parse_dates=["timestamp"])

    def context_window(self, hours: int = 24) -> str:
        """
        Return a plain-text summary of the most recent `hours` rows.

        This text is injected into the agent's prompt so it knows what the
        home sensors have been reading lately.
        """
        df = self.load()
        recent = df.tail(hours)  # last N rows (one per hour = last N hours)

        avg_temp  = recent["temperature_c"].mean()
        avg_humid = recent["humidity_pct"].mean()
        avg_power = recent["power_kw"].mean()
        occupied  = int(recent["occupancy"].sum())
        anomalies = int(recent["anomaly"].sum())

        summary = (
            f"Sensor summary (last {hours} hours): "
            f"avg temperature {avg_temp:.1f} °C, "
            f"avg humidity {avg_humid:.0f} %, "
            f"avg power draw {avg_power:.2f} kW, "
            f"home occupied for {occupied} of {len(recent)} hours"
        )
        if anomalies:
            summary += f", {anomalies} anomaly reading(s) detected"
        return summary + "."
