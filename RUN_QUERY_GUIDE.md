# Run Smart Home LangGraph With Your Query

This guide shows the exact steps to run the current unified project from PowerShell.

## 1) Open terminal in project root

You should be in this folder:

D:/Germany/Jobs/working-student/DFKI/ASR-Group/development/smart-home-langgraph

## 2) Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If you have not installed dependencies yet:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3) Set environment variable for imports

```powershell
$env:PYTHONPATH = "src"
```

## 4) Add Gemini API key (one-time setup)

1. Copy .env.example to .env
2. Put your key in GEMINI_API_KEY

Example line in .env:

GEMINI_API_KEY=your_real_key_here

If the key is missing, the app still runs using fallback output.

## 5) Run with your query

```powershell
python -m smart_home_langgraph.main --query "How can I reduce my evening power usage?"
```

Replace the query text with anything you want.

## 6) What happens after run

- The app detects intent from your query.
- It reads sensor context from src/smart_home_langgraph/data/home_data.xlsx.
- It retrieves long-term memory from runtime_memory/*.json.
- It generates a response (Gemini or fallback).
- It critiques and repairs if needed.
- It writes learning back to runtime_memory.

## 7) Useful checks

Run tests:

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests -v
```

Inspect memory files created after runs:

- runtime_memory/mistakes.json
- runtime_memory/recipes.json
- runtime_memory/preferences.json

## 8) Common issue

If PowerShell blocks script activation, run this once in an elevated terminal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then close and reopen terminal.
