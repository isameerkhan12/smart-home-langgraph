# ---------------------------------------------------------------------------
# tools/python_executor.py
# Purpose: Python code execution tool for DataFrame analysis.
#
# Uses LangChain's PythonAstREPLTool with pre-loaded smart home DataFrame.
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool
from langchain_experimental.tools import PythonAstREPLTool

# Path to the telemetry CSV in the data/ folder.
_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "preprocessed_dataset.csv",
)

# Tool description with column info for LLM context
_TOOL_DESCRIPTION = (
    "Execute Python code to analyze smart home telemetry data. "
    "The DataFrame is available as 'df' with columns: "
    "Unix_Timestamp, timestamp (datetime), Transaction_ID, "
    "Television, Dryer, Oven, Refrigerator, Microwave (appliance states 0/1), "
    "Line_Voltage, Voltage, Apparent_Power, Energy_Consumption_kWh, "
    "Offloading_Decision, Bandwidth. "
    "Use pandas/numpy for analysis. Always print the final result."
)


@lru_cache(maxsize=1)
def _load_dataframe(path: str = _DATA_FILE) -> pd.DataFrame:
    """Load and preprocess the telemetry DataFrame (cached)."""
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [
        col.strip().replace(" ", "_").replace("(", "").replace(")", "")
        for col in df.columns
    ]
    # Add datetime column for time-based analysis
    if "Unix_Timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Unix_Timestamp"], unit="s")
    return df


def get_smart_home_tools(dataframe_path: str | None = None) -> list[BaseTool]:
    """
    Get tools for the smart home agent.
    
    Returns a PythonAstREPLTool with:
    - Pre-loaded DataFrame as 'df'
    - pandas (pd) and numpy (np) available
    
    Args:
        dataframe_path: Optional custom path to telemetry CSV.
    """
    path = dataframe_path or _DATA_FILE
    df = _load_dataframe(path)
    
    tool = PythonAstREPLTool(
        locals={"df": df, "pd": pd, "np": np},
        name="python_repl",
        description=_TOOL_DESCRIPTION,
    )
    return [tool]
