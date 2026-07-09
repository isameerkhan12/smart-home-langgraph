#!/usr/bin/env python
"""Quick integration test for dataset and tool."""
import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("INTEGRATION TEST: House_1.csv Dataset")
print("=" * 60)

# Test 1: Import and load the tool
print("\n[1] Loading DataFrame...")
try:
    from smart_home_langgraph.tools.python_executor import _DF, get_smart_home_tools
    print(f"    ✓ DataFrame loaded successfully")
    print(f"    Shape: {_DF.shape[0]:,} rows × {_DF.shape[1]} columns")
except Exception as e:
    print(f"    ✗ Failed to load DataFrame: {e}")
    sys.exit(1)

# Test 2: Check columns
print("\n[2] Checking columns...")
expected_appliances = [
    'Fridge', 'Chest_Freezer', 'Upright_Freezer', 'Tumble_Dryer',
    'Washing_Machine', 'Dishwasher', 'Computer_Site', 'Television_Site', 'Electric_Heater'
]
found_appliances = [col for col in expected_appliances if col in _DF.columns]
print(f"    Appliance columns: {len(found_appliances)}/9 renamed")
for app in found_appliances[:3]:
    print(f"      - {app}")
print(f"      ... and {len(found_appliances)-3} more")

required_cols = ['Time', 'Unix', 'Aggregate', 'timestamp']
for col in required_cols:
    status = "✓" if col in _DF.columns else "✗"
    print(f"    {status} {col}")

# Test 3: Sample data
print("\n[3] Data preview...")
print(f"    First timestamp: {_DF['timestamp'].iloc[0]}")
print(f"    Last timestamp:  {_DF['timestamp'].iloc[-1]}")
print(f"    Fridge samples: {_DF['Fridge'].head(3).tolist()}")

# Test 4: Tool creation
print("\n[4] Creating tool...")
try:
    tools = get_smart_home_tools()
    print(f"    ✓ Tool created: {tools[0].name}")
    desc_preview = tools[0].description[:80]
    print(f"    Description: {desc_preview}...")
except Exception as e:
    print(f"    ✗ Failed to create tool: {e}")
    sys.exit(1)

# Test 5: Workflow
print("\n[5] Testing workflow build...")
try:
    from smart_home_langgraph.graph.workflow import build_workflow
    workflow = build_workflow()
    print(f"    ✓ Workflow built successfully")
except Exception as e:
    print(f"    ✗ Failed to build workflow: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - System ready!")
print("=" * 60)
