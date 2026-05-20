# Run Smart Home LangGraph (Local + Docker Postgres)

This guide shows the exact steps to run the current project from PowerShell.

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

## 3) Start Postgres with Docker

From the project root:

```powershell
docker compose up -d postgres
docker compose ps
```

The included compose file maps container port 5432 to host port 5442.

## 4) Set environment variable for imports

```powershell
$env:PYTHONPATH = "src"
```

## 5) Add environment variables (one-time setup)

1. Copy .env.example to .env
2. Set at least these values:

Example .env:

GEMINI_API_KEY=your_real_key_here
POSTGRES_URI=postgresql://postgres:postgres@localhost:5442/postgres
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIMS=768

Notes:

- If GEMINI_API_KEY is missing, the app still runs with fallback generation.
- If POSTGRES_URI or GEMINI_API_KEY is missing, long-term memory store is disabled.

## 6) Run interactive chat

```powershell
python -m smart_home_langgraph.main --max-repairs 2 --thread default
```

Then type your question at the `You:` prompt.

Useful commands inside chat:

- `reset` starts a new conversation thread.
- `exit` or `quit` closes the session.

## 7) What happens after run

- The app detects intent from your query.
- It reads sensor context from src/smart_home_langgraph/data/home_data.xlsx.
- It loads per-thread short-term state from local SQLite (smart_home_agent.db).
- If configured, it retrieves long-term memory from Postgres (pgvector-backed).
- It generates a response (Gemini or fallback).
- It critiques and repairs if needed.
- It writes learned memory records into Postgres store.

## 8) Useful checks

Run tests:

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests -v
```

Check Postgres container status:

```powershell
docker compose ps
```

## 9) Common issues

If `docker compose` says it cannot connect to the Docker API:

- Open Docker Desktop.
- Wait until the engine status shows Running.
- Re-run `docker compose up -d postgres`.

If PowerShell blocks script activation, run this once in an elevated terminal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then close and reopen terminal.

If Docker is not running or Postgres is unreachable:

- Chat still works.
- Long-term memory retrieval/writes are skipped until Postgres is available.
