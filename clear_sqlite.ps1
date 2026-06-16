param(
    [string]$DbPath = "smart_home_agent.db"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DbPath)) {
    throw "SQLite file not found: $DbPath"
}

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$tempScript = Join-Path $PSScriptRoot "_clear_sqlite_tmp.py"

@'
import sqlite3
import sys

db = sys.argv[1]
con = sqlite3.connect(db)
cur = con.cursor()

tables = [
    r[0]
    for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
]

print("Tables:", tables)

cur.execute("PRAGMA foreign_keys=OFF")
for t in tables:
    cur.execute(f"DELETE FROM {t}")

has_sqlite_sequence = cur.execute(
    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
).fetchone() is not None

if has_sqlite_sequence and tables:
    placeholders = ",".join(["?"] * len(tables))
    cur.execute(f"DELETE FROM sqlite_sequence WHERE name IN ({placeholders})", tables)

con.commit()

counts = [(t, cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in tables]
print("Row counts after clear:", counts)

con.close()
'@ | Set-Content -Path $tempScript

try {
    & $pythonExe $tempScript $DbPath
}
finally {
    if (Test-Path $tempScript) {
        Remove-Item $tempScript -Force
    }
}

Write-Host "SQLite clear completed."