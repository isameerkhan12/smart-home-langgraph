# ---------------------------------------------------------------------------
# tests/test_phase0_workflow.py
# Purpose: Verify the Phase 0 graph produces expected output.
#
# How to run:
#   $env:PYTHONPATH = "src"
#   pytest -q
# ---------------------------------------------------------------------------
from smart_home_langgraph.main import run  # import the helper that builds and runs the graph


def test_phase0_workflow_returns_intent_and_plan() -> None:
    # Send a query that contains the keyword "save" so detect_intent
    # should classify it as "energy_optimization".
    output = run("How do I save energy at night?")

    # assert = "I expect this to be true; fail the test if it is not."
    # We check that the response text contains the intent label ...
    assert "Intent detected: energy_optimization" in output
    # ... and that it contains a starter plan (not just an empty reply).
    assert "Starter plan:" in output
