# ---------------------------------------------------------------------------
# tools/python_executor.py
# Purpose: Python code execution tool for DataFrame analysis.
#
# Uses LangChain's PythonREPLTool with a pre-loaded smart home DataFrame.
# ---------------------------------------------------------------------------
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.utilities.python import PythonREPL

# Path to the telemetry CSV in the data/ folder.
_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "House_1.csv",
    "House_1.csv",
)

# Tool description with column info for LLM context
_TOOL_DESCRIPTION = (
    "Execute Python code to analyze smart home energy consumption data. "
    "The DataFrame is available as 'df' with columns: "
    "Time (datetime string), Unix (timestamp), Aggregate (total power), "
    "Fridge, Chest_Freezer, Upright_Freezer, Tumble_Dryer, Washing_Machine, "
    "Dishwasher, Computer_Site, Television_Site, Electric_Heater (all in Watts). "
    "Use pandas/numpy for analysis. Always print the final result."
)

# Appliance name mapping for column renaming
_APPLIANCE_NAMES = {
    "Appliance1": "Fridge",
    "Appliance2": "Chest_Freezer",
    "Appliance3": "Upright_Freezer",
    "Appliance4": "Tumble_Dryer",
    "Appliance5": "Washing_Machine",
    "Appliance6": "Dishwasher",
    "Appliance7": "Computer_Site",
    "Appliance8": "Television_Site",
    "Appliance9": "Electric_Heater",
}


def _load_dataframe(path: str = _DATA_FILE) -> pd.DataFrame:
    """Load and preprocess the energy consumption DataFrame."""
    df = pd.read_csv(path)
    # Normalize column names (strip whitespace, replace spaces)
    df.columns = [
        col.strip().replace(" ", "_").replace("(", "").replace(")", "")
        for col in df.columns
    ]
    # Rename Appliance columns to their real names
    df = df.rename(columns=_APPLIANCE_NAMES)
    # Add datetime column for time-based analysis
    if "Unix" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Unix"], unit="s")
    return df


# Load DataFrame at module import time (once per process)
_DF = _load_dataframe(_DATA_FILE)


def get_smart_home_tools(dataframe_path: str | None = None) -> list[BaseTool]:
    """
    Get tools for the smart home agent.
    
    Returns a PythonREPLTool with:
    - Pre-loaded DataFrame as 'df'
    - pandas (pd) and numpy (np) available
    
    Args:
        dataframe_path: Optional custom path to telemetry CSV (for testing).
                       If None, uses the globally loaded DataFrame.
    """
    # Use global DataFrame by default, or load custom one if path provided
    df = _DF if dataframe_path is None else _load_dataframe(dataframe_path)
    
    repl = PythonREPL(_globals={"df": df, "pd": pd, "np": np})
    tool = PythonREPLTool(
        python_repl=repl,
        name="python_repl",
        description=_TOOL_DESCRIPTION,
    )
    return [tool]
