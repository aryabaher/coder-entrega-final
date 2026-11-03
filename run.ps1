# Una sola instrucción en PowerShell: Redis, Phoenix, la API y el worker.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Se creó .env desde .env.example. Cargá la clave del proveedor antes de usar la API."
}
docker compose up --build
