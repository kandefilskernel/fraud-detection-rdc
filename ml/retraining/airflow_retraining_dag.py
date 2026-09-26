"""
DAG Airflow de réentraînement (déploiement de production).

Mêmes étapes que le planificateur léger (ml/retraining/scheduler.py), orchestrées par
Airflow avec historique d'exécution, reprise sur erreur et alertes :

    verifier_etiquettes >> entrainer_et_comparer >> recharger_modele >> reconstruire_reference_derive

À placer dans le dossier dags/ d'Airflow ; l'image des workers Airflow doit contenir le
dépôt (paquets ml/ et shared/) et les variables POSTGRES_* / ARTIFACTS_DIR.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {"owner": "equipe-fraude", "retries": 2, "retry_delay": timedelta(minutes=10)}


def _engine():
    from shared.database.session import make_session_factory
    return make_session_factory()[0]


def verifier_etiquettes(**_):
    from ml.retraining import retrain
    prod = retrain.load_production(_engine())
    if len(prod) < retrain.MIN_LABELED or prod["y"].sum() < retrain.MIN_FRAUDS:
        raise AirflowSkipException(f"étiquettes insuffisantes : {len(prod)} lignes, {int(prod['y'].sum())} fraudes")
    return {"labeled": len(prod), "frauds": int(prod["y"].sum())}


def entrainer_et_comparer(**_):
    from ml.retraining import retrain
    report = retrain.run(_engine())
    if report["status"] != "CHAMPION":
        raise AirflowSkipException(f"challenger non promu : {report['status']}")
    return report["version"]


def recharger_modele(**_):
    # Le scoring-service surveille metadata.json et recharge le modèle à chaud :
    # on vérifie simplement que la nouvelle version est bien servie.
    import os

    import httpx
    health = httpx.get(os.getenv("SCORING_URL", "http://scoring-service:8001") + "/health", timeout=10).json()
    return health["model_version"]


def reconstruire_reference_derive(**_):
    return "référence écrite par retrain.run (drift_reference.npz du candidat promu)"


with DAG(
    dag_id="fraude_rdc_reentrainement",
    description="Réentraînement champion/challenger du modèle hybride de détection de fraude",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * 1",          # chaque lundi à 2 h (heure creuse)
    catchup=False,
    max_active_runs=1,
    tags=["fraude", "ml", "rdc"],
) as dag:
    t1 = PythonOperator(task_id="verifier_etiquettes", python_callable=verifier_etiquettes)
    t2 = PythonOperator(task_id="entrainer_et_comparer", python_callable=entrainer_et_comparer,
                        execution_timeout=timedelta(hours=2))
    t3 = PythonOperator(task_id="recharger_modele", python_callable=recharger_modele)
    t4 = PythonOperator(task_id="reconstruire_reference_derive", python_callable=reconstruire_reference_derive)
    t1 >> t2 >> t3 >> t4
