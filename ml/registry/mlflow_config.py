"""Configuration MLflow : serveur de la plateforme si disponible, sinon fichier local.

    MLFLOW_TRACKING_URI=http://localhost:5000   (serveur docker compose, depuis Windows)
    MLFLOW_TRACKING_URI=http://mlflow:5000      (depuis les conteneurs)
    sans variable : sqlite:///mlflow.db          (dossier courant)
"""
import os

import mlflow

EXPERIMENT = "fraud_detection_rdc_experiment"


def init_mlflow(experiment: str = EXPERIMENT) -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri
