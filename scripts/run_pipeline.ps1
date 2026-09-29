# Pipeline ML complet : données -> variables -> modèles -> référence de dérive
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
# "Continue" et non "Stop" : sous Windows PowerShell 5.1, les journaux Python écrits sur stderr
# deviendraient des erreurs bloquantes dès que la sortie est redirigée (> log.txt). Les échecs
# réels sont détectés par $LASTEXITCODE après chaque étape.
$ErrorActionPreference = "Continue"
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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Référence de dérive (supervision du modèle en production) ===" -ForegroundColor Cyan
python -m ml.monitoring.build_reference
exit $LASTEXITCODE
