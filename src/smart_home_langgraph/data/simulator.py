# ---------------------------------------------------------------------------
# data/simulator.py  –  Phase 2: Synthetic Smart-Home Time-Series Generator
#
# Purpose:
#   Generate realistic-looking sensor and appliance data for a single
#   simulated home so we can test the agent without needing real hardware.
#
# Why simulate instead of using real data?
#   - No hardware or API access needed → faster to get started.
#   - We control the patterns (e.g. add an anomaly spike on purpose) so we
#     can write tests that check whether the agent detects it.
#   - Reproducible: same random seed → same data every time.
#
# What data is generated?
#   One row per 15-minute interval over a requested number of days.
#   Columns:
#     timestamp      – datetime of the reading
#     temperature_c  – indoor temperature in Celsius
#     humidity_pct   – indoor relative humidity (0-100)
#     occupancy      – 1 if someone is home, 0 if empty
#     power_kw       – total appliance power draw in kilowatts
#     anomaly        – 1 if this row was artificially spiked, else 0
#
# How to use:
#   sim = SmartHomeSimulator(seed=42)
#   df  = sim.generate(days=7)
#   print(df.head())
#   summary = sim.daily_summary(df)
# ---------------------------------------------------------------------------
from __future__ import annotations

import numpy as np          # numerical operations and random number generation
import pandas as pd         # DataFrame — the standard tool for tabular time-series data


class SmartHomeSimulator:
    """Generates synthetic smart-home sensor and appliance time-series data."""

    # Default realistic ranges used during generation.
    # These can be overridden by passing kwargs to __init__ if needed later.
    TEMP_BASE   = 20.0    # baseline indoor temperature in Celsius
    TEMP_NOISE  = 1.5     # random noise added per reading (±)
    HUMID_BASE  = 50.0    # baseline humidity %
    HUMID_NOISE = 5.0
    POWER_BASE  = 0.5     # baseline idle power draw in kW (standby devices)
    INTERVAL_MIN = 15     # minutes between readings

    def __init__(self, seed: int = 42) -> None:
        # seed makes the random generator deterministic:
        # same seed → same data every run → reproducible tests.
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # generate()
    # Main method. Returns a pandas DataFrame with one row per interval.
    # ------------------------------------------------------------------
    def generate(self, days: int = 7, anomaly_probability: float = 0.01) -> pd.DataFrame:
        """
        Generate 'days' worth of 15-minute smart-home readings.

        Parameters
        ----------
        days               : how many days of data to produce
        anomaly_probability: chance (0-1) that any given row is a spike anomaly

        Returns
        -------
        pd.DataFrame with columns: timestamp, temperature_c, humidity_pct,
                                   occupancy, power_kw, anomaly
        """
        # Total number of 15-minute intervals across the requested days.
        n = days * 24 * (60 // self.INTERVAL_MIN)   # e.g. 7 days → 672 rows

        # Build a regular timestamp index starting at midnight today.
        # pd.date_range creates evenly-spaced datetime values.
        timestamps = pd.date_range(
            start="2026-01-01 00:00",   # fixed start date keeps data reproducible
            periods=n,
            freq=f"{self.INTERVAL_MIN}min",
        )

        # --- Temperature -------------------------------------------------
        # Base temperature + sinusoidal daily cycle (cooler at night,
        # warmer in the afternoon) + small random noise per reading.
        hour_of_day = timestamps.hour + timestamps.minute / 60.0
        # np.sin produces a smooth wave; we shift it so the peak is at 14:00.
        daily_cycle = 2.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        temp = (
            self.TEMP_BASE
            + daily_cycle
            + self._rng.normal(0, self.TEMP_NOISE, n)
        )

        # --- Humidity ----------------------------------------------------
        # Humidity is inversely correlated with temperature (simple approximation).
        humidity = (
            self.HUMID_BASE
            - 0.5 * daily_cycle            # slightly lower when hot
            + self._rng.normal(0, self.HUMID_NOISE, n)
        )
        # Clamp to a realistic range so we never get -10% or 150% humidity.
        humidity = np.clip(humidity, 20.0, 90.0)

        # --- Occupancy ---------------------------------------------------
        # 1 = someone home, 0 = empty.
        # Simple rule: occupied 07:00-09:00 and 17:00-23:00 (commuter pattern).
        occupancy = np.where(
            ((hour_of_day >= 7) & (hour_of_day < 9)) |
            ((hour_of_day >= 17) & (hour_of_day < 23)),
            1, 0
        ).astype(int)

        # --- Power draw --------------------------------------------------
        # Idle baseline + occupancy-driven load + random variation.
        # When someone is home, appliances (TV, cooking, etc.) add ~1.5 kW on average.
        power = (
            self.POWER_BASE
            + occupancy * self._rng.uniform(0.5, 2.5, n)   # variable home-use load
            + self._rng.uniform(0.0, 0.3, n)                # small always-on fluctuation
        )

        # --- Anomaly injection -------------------------------------------
        # With probability anomaly_probability, spike the power to simulate
        # a fault (e.g. appliance left on, sensor glitch, heating runaway).
        anomaly_mask = self._rng.random(n) < anomaly_probability
        power = np.where(anomaly_mask, power * self._rng.uniform(4, 8, n), power)
        temp  = np.where(anomaly_mask, temp + self._rng.uniform(3, 8, n), temp)

        # --- Assemble DataFrame ------------------------------------------
        df = pd.DataFrame({
            "timestamp":     timestamps,
            "temperature_c": np.round(temp, 1),      # 1 decimal place
            "humidity_pct":  np.round(humidity, 1),
            "occupancy":     occupancy,
            "power_kw":      np.round(power, 3),
            "anomaly":       anomaly_mask.astype(int),  # 1 = spike row, 0 = normal
        })

        return df

    # ------------------------------------------------------------------
    # daily_summary()
    # Condense the 15-minute rows into one summary row per day.
    # The agent will use these summaries as context when answering questions.
    # ------------------------------------------------------------------
    def daily_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 15-minute readings into a per-day summary.

        Returns a DataFrame with one row per calendar day and columns:
          date, avg_temp_c, avg_humidity_pct, occupied_hours,
          total_power_kwh, anomaly_count
        """
        # Create a date column for grouping (strip the time part).
        df = df.copy()
        df["date"] = df["timestamp"].dt.date   # e.g. 2026-01-01

        summary = df.groupby("date").agg(
            avg_temp_c      =("temperature_c", "mean"),
            avg_humidity_pct=("humidity_pct",  "mean"),
            # Sum occupancy slots × interval length to get occupied hours.
            # Each interval is INTERVAL_MIN/60 hours.
            occupied_hours  =("occupancy",     lambda x: x.sum() * self.INTERVAL_MIN / 60),
            # Energy in kWh = power (kW) × time (h) summed across intervals.
            total_power_kwh =("power_kw",      lambda x: x.sum() * self.INTERVAL_MIN / 60),
            anomaly_count   =("anomaly",        "sum"),
        ).reset_index()

        # Round for readability.
        summary["avg_temp_c"]       = summary["avg_temp_c"].round(1)
        summary["avg_humidity_pct"] = summary["avg_humidity_pct"].round(1)
        summary["occupied_hours"]   = summary["occupied_hours"].round(1)
        summary["total_power_kwh"]  = summary["total_power_kwh"].round(2)

        return summary

    # ------------------------------------------------------------------
    # context_window()
    # Return the last N hours of raw readings as a compact text summary.
    # This text is what the retriever node will pass to the LLM as context.
    # ------------------------------------------------------------------
    def context_window(self, df: pd.DataFrame, hours: int = 24) -> str:
        """
        Return the most recent `hours` of readings as a plain-text summary
        suitable for pasting directly into an LLM prompt.
        """
        # Calculate how many rows cover the requested window.
        rows_needed = hours * (60 // self.INTERVAL_MIN)   # e.g. 24h → 96 rows
        window = df.tail(rows_needed)                     # last N rows

        if window.empty:
            return "No sensor data available."

        # Compute aggregate stats for the window.
        avg_temp     = window["temperature_c"].mean().round(1)
        avg_humidity = window["humidity_pct"].mean().round(1)
        total_power  = (window["power_kw"].sum() * self.INTERVAL_MIN / 60).round(2)
        occupied_h   = (window["occupancy"].sum() * self.INTERVAL_MIN / 60).round(1)
        anomalies    = int(window["anomaly"].sum())

        start = window["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M")
        end   = window["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M")

        # Build a human-readable summary string for the LLM prompt.
        return (
            f"Sensor data window: {start} to {end} ({hours}h)\n"
            f"  Avg temperature : {avg_temp} °C\n"
            f"  Avg humidity    : {avg_humidity} %\n"
            f"  Occupied hours  : {occupied_h} h\n"
            f"  Total energy    : {total_power} kWh\n"
            f"  Anomaly readings: {anomalies}"
        )
