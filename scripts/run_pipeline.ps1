# Pipeline ML complet : phase 1 (données) -> phase 2 (variables) -> phase 3 (modèles)
# Usage (depuis la racine du projet) :
#   .\scripts\run_pipeline.ps1                 # complet
#   .\scripts\run_pipeline.ps1 -Quick          # essai rapide
#   .\scripts\run_pipeline.ps1 -SkipGeneration # réutilise ml/data/raw
param(
    [switch]$Quick,
    [switch]$SkipGeneration,
    [int]$Users = 3000,
    [int]$Days = 180,
    [double]$VisaAdoption = 2.5
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $SkipGeneration) {
    Write-Host "`n=== Phase 1 : génération des données synthétiques ===" -ForegroundColor Cyan
    python ml/generator/generate_synthetic_data.py --n-users $Users --n-days $Days --visa-adoption-multiplier $VisaAdoption
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`n=== Phase 2 : variables comportementales, split temporel, normalisation ===" -ForegroundColor Cyan
python -m ml.preprocessing.train_test_split_normalize --rebuild-features
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Phase 3 : entraînement et comparaison des modèles ===" -ForegroundColor Cyan
$trainArgs = @()
if ($Quick) { $trainArgs += "--quick" }
python -m ml.training.train_model @trainArgs
exit $LASTEXITCODE
