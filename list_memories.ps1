param(
    [string]$UserId,
    [int]$Limit = 200,
    [string]$Service = "postgres",
    [string]$User = "postgres",
    [string]$Database = "postgres"
)

$ErrorActionPreference = "Stop"

if ($Limit -lt 1) {
    throw "Limit must be >= 1"
}

if ([string]::IsNullOrWhiteSpace($UserId)) {
    Write-Host "Listing memories for all users (limit $Limit) via service '$Service'..."
    $sql = @"
SELECT
  split_part(prefix, '.', 3) AS user_id,
  prefix,
  key,
  value->>'memory_type' AS memory_type,
  value->>'content' AS content,
  created_at
FROM public.store
WHERE prefix LIKE 'smart_home.users.%.typed_memories'
ORDER BY created_at DESC
LIMIT $Limit;
"@
}
else {
    $prefix = "smart_home.users.$UserId.typed_memories"
    Write-Host "Listing memories for user '$UserId' (limit $Limit) via service '$Service'..."
    $sql = @"
SELECT
  split_part(prefix, '.', 3) AS user_id,
  prefix,
  key,
  value->>'memory_type' AS memory_type,
  value->>'content' AS content,
  created_at
FROM public.store
WHERE prefix = '$prefix'
ORDER BY created_at DESC
LIMIT $Limit;
"@
}

docker compose exec -T $Service psql -U $User -d $Database -v ON_ERROR_STOP=1 -c $sql

Write-Host "Memory list completed."
