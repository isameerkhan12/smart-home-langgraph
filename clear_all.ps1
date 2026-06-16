param(
    [string]$SqlitePath = "smart_home_agent.db",
    [string]$PostgresService = "postgres",
    [string]$PostgresUser = "postgres",
    [string]$PostgresDatabase = "postgres"
)

$ErrorActionPreference = "Stop"

$clearPostgres = Join-Path $PSScriptRoot "clear_postgres.ps1"
$clearSqlite = Join-Path $PSScriptRoot "clear_sqlite.ps1"

if (-not (Test-Path $clearPostgres)) {
    throw "Missing script: $clearPostgres"
}
if (-not (Test-Path $clearSqlite)) {
    throw "Missing script: $clearSqlite"
}

Write-Host "Clearing Postgres..."
& $clearPostgres -Service $PostgresService -User $PostgresUser -Database $PostgresDatabase

Write-Host "Clearing SQLite..."
& $clearSqlite -DbPath $SqlitePath

Write-Host "All database clear operations completed."