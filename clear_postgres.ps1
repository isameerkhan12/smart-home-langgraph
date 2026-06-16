param(
    [string]$Service = "postgres",
    [string]$User = "postgres",
    [string]$Database = "postgres"
)

$ErrorActionPreference = "Stop"

Write-Host "Clearing Postgres tables via docker compose service '$Service'..."

$sql = "TRUNCATE TABLE public.store, public.store_migrations, public.store_vectors, public.telemetry_events, public.telemetry_ingest_meta, public.vector_migrations RESTART IDENTITY CASCADE;"

docker compose exec -T $Service psql -U $User -d $Database -v ON_ERROR_STOP=1 -c $sql

Write-Host "Verifying row counts..."
docker compose exec -T $Service psql -U $User -d $Database -c "SELECT 'public.store' AS table_name, count(*) AS row_count FROM public.store UNION ALL SELECT 'public.store_migrations', count(*) FROM public.store_migrations UNION ALL SELECT 'public.store_vectors', count(*) FROM public.store_vectors UNION ALL SELECT 'public.telemetry_events', count(*) FROM public.telemetry_events UNION ALL SELECT 'public.telemetry_ingest_meta', count(*) FROM public.telemetry_ingest_meta UNION ALL SELECT 'public.vector_migrations', count(*) FROM public.vector_migrations;"

Write-Host "Postgres clear completed."