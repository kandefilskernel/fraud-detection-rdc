"""
Déclencheur de réentraînement (service « retrainer »).

Vérifie toutes les CHECK_INTERVAL secondes et lance ml.retraining.retrain si :
    - assez de NOUVELLES étiquettes (verdicts + plaintes) depuis le dernier entraînement ;
    - ou une dérive est signalée par le détecteur (Prometheus : drift_alarm == 1) ;
    - ou la période maximale sans réentraînement est dépassée.

En production, le même enchaînement est orchestré par Airflow
(ml/retraining/airflow_retraining_dag.py) ; ce planificateur léger évite d'installer
Airflow (≈ 4 Go) sur un poste de développement.
"""
from __future__ import annotations

import logging
import os
import time

import httpx
from sqlalchemy import text

from ml.retraining import retrain
from shared.database.session import make_session_factory
from shared.logging.logger_config import configure_logging

CHECK_INTERVAL = int(os.getenv("RETRAIN_CHECK_INTERVAL_S", "600"))
NEW_LABELS = int(os.getenv("RETRAIN_NEW_LABELS", "500"))
MAX_AGE_H = float(os.getenv("RETRAIN_MAX_AGE_HOURS", "168"))   # une semaine
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
log = logging.getLogger("retrainer")


def drift_alarm() -> bool:
    try:
        r = httpx.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": "max(drift_alarm)"}, timeout=5)
        res = r.json()["data"]["result"]
        return bool(res) and float(res[0]["value"][1]) >= 1
    except Exception:  # noqa: BLE001
        return False


def state(engine) -> tuple[int, float | None]:
    with engine.connect() as c:
        last = c.execute(text("SELECT max(created_at) FROM model_registry")).scalar()
        new = c.execute(text("SELECT count(*) FROM scored_transactions WHERE label IS NOT NULL "
                             "AND (CAST(:t AS timestamptz) IS NULL OR labeled_at > :t)"), {"t": last}).scalar()
    age_h = (time.time() - last.timestamp()) / 3600 if last else None
    return int(new), age_h


def main():
    configure_logging("retrainer")
    engine, _ = make_session_factory()
    log.info("planificateur démarré : contrôle toutes les %ds, seuil %d nouvelles étiquettes", CHECK_INTERVAL, NEW_LABELS)
    while True:
        try:
            new_labels, age_h = state(engine)
            reasons = []
            if new_labels >= NEW_LABELS:
                reasons.append(f"{new_labels} nouvelles étiquettes")
            if drift_alarm():
                reasons.append("dérive détectée")
            if age_h is not None and age_h > MAX_AGE_H:
                reasons.append(f"modèle vieux de {age_h:.0f} h")
            if reasons:
                log.info("réentraînement déclenché : %s", ", ".join(reasons))
                report = retrain.run(engine)
                log.info("résultat : %s", {k: report.get(k) for k in ("status", "version", "reason")})
            else:
                log.info("rien à faire (nouvelles étiquettes=%d)", new_labels)
        except Exception:  # noqa: BLE001
            log.exception("échec du cycle de réentraînement")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
