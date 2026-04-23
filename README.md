# smart-home-langgraph

Beginner-friendly LangGraph smart-home agent with:

- intent detection
- context retrieval (Excel data + memory)
- response generation
- critique and repair loop
- memory writing for future runs

## Quick run

From project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m smart_home_langgraph.main --query "How can I reduce my evening power usage?"
```

## Full step-by-step guide

See RUN_QUERY_GUIDE.md for beginner-friendly setup and run steps.

## Run tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests -v
```