<#
.SYNOPSIS
  Mode démo : arrête les services non indispensables au temps réel pour libérer le processeur
  (PC de développement à 4 cœurs), ou les relance.

.EXAMPLE
  .\scripts\demo_mode.ps1                 # démo allégée (scoring, tableau de bord, alertes)
  .\scripts\demo_mode.ps1 -Supervision    # garde Prometheus + Grafana pour montrer les métriques
  .\scripts\demo_mode.ps1 -Restore        # relance toute la plateforme
#>
param([switch]$Supervision, [switch]$Restore)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Tâches de fond et outils : inutiles pour noter les transactions en temps réel
$fond = @("drift-monitor", "retrainer", "mlflow", "redpanda-console", "loki", "promtail",
          "redis-exporter", "postgres-exporter")
$outilsSupervision = @("prometheus", "grafana")

if ($Restore) {
    docker compose up -d --no-build
    Write-Host "`nPlateforme complète relancée." -ForegroundColor Green
    exit 0
}

$aArreter = if ($Supervision) { $fond } else { $fond + $outilsSupervision }
docker compose stop @aArreter
Write-Host "`nMode démo actif. Services arrêtés : $($aArreter -join ', ')" -ForegroundColor Green
Write-Host "Temps réel conservé : scoring, intégration, back-office, tableau de bord, Kafka, alertes, portefeuille."
Write-Host "Conseil : fermer Chrome et les applications lourdes pendant la démo."
Write-Host "Pour tout relancer : .\scripts\demo_mode.ps1 -Restore"
