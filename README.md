# smart-home-langgraph

Beginner-friendly LangGraph project for a simulation-first smart-home agent.

## Phase 0 (implemented)

This phase gives you a minimal runnable graph with two nodes:
1. `detect_intent`
2. `build_response`

It is intentionally simple so you can understand graph flow before adding data, memory, critique, and repair loops.

## Virtual environment setup (Windows PowerShell)

Run these commands from the project root:

```powershell
C:/Users/Sameer-188/AppData/Local/Python/bin/python3.14.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment variables

1. Copy `.env.example` to `.env`.
2. Put your Gemini API key in `GEMINI_API_KEY`.

Note: Phase 0 does not call Gemini yet. The key is prepared for later phases.

## Run the Phase 0 graph

```powershell
$env:PYTHONPATH = "src"
python -m smart_home_langgraph.main --query "How can I reduce my evening power usage?"
```

## Run tests

```powershell
$env:PYTHONPATH = "src"
pytest -q
```

## Next phase preview

Phase 1 and Phase 2 will add:
1. Synthetic time-series data.
2. Long-term memory stores (preferences, mistakes, recipes).
3. Retrieval for similar past tasks.