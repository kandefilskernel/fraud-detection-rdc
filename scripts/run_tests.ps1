# Lance toutes les suites de tests, une par service (chaque service a son propre paquet `app`).
# Prérequis : venv activé ; PostgreSQL démarré pour les tests du backoffice (docker compose up -d postgres).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$suites = @("ml/tests", "services/scoring-service/tests", "services/integration-layer/tests",
            "services/backoffice-api/tests", "services/workers/tests")
$failed = @()
foreach ($s in $suites) {
    Write-Host "`n=== $s ===" -ForegroundColor Cyan
    python -W ignore -m pytest $s -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) { $failed += $s }
}
if ($failed.Count -gt 0) { Write-Host "`nÉCHECS : $($failed -join ', ')" -ForegroundColor Red; exit 1 }
Write-Host "`nToutes les suites passent." -ForegroundColor Green
