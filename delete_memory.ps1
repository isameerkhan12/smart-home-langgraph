param(
    [Parameter(Mandatory = $true)]
    [string]$MemoryKey,
    [string]$UserId = "default_user",
    [string]$Service = "postgres",
    [string]$User = "postgres",
    [string]$Database = "postgres"
)

$ErrorActionPreference = "Stop"

$prefix = "smart_home.users.$UserId.typed_memories"

Write-Host "Deleting memory '$MemoryKey' from namespace '$prefix' via service '$Service'..."

$checkSql = "SELECT key, value->>'memory_type' AS memory_type, value->>'content' AS content FROM public.store WHERE prefix = '$prefix' AND key = '$MemoryKey';"
$deleteSql = "DELETE FROM public.store WHERE prefix = '$prefix' AND key = '$MemoryKey';"
$verifySql = "SELECT key FROM public.store WHERE prefix = '$prefix' AND key = '$MemoryKey';"

Write-Host "Checking target memory..."
docker compose exec -T $Service psql -U $User -d $Database -v ON_ERROR_STOP=1 -c $checkSql

Write-Host "Deleting..."
docker compose exec -T $Service psql -U $User -d $Database -v ON_ERROR_STOP=1 -c $deleteSql

Write-Host "Verifying deletion..."
docker compose exec -T $Service psql -U $User -d $Database -v ON_ERROR_STOP=1 -c $verifySql

Write-Host "Delete operation completed."