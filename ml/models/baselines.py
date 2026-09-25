"""
Modèles de référence : sans eux, impossible de démontrer l'apport du modèle hybride.

    - Règles métier    : ce que font beaucoup d'opérateurs aujourd'hui (seuils fixes)
    - Régression logistique
    - Random Forest
    - Isolation Forest : non supervisé (comparable à la branche autoencodeur)
    - XGBoost seul     : = branche 1 utilisée sans les deux autres
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.features.feature_engineering import FEATURE_NAMES


def train_logistic_regression(X, y, seed=42):
    return LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed).fit(X, y)


def train_random_forest(X, y, seed=42):
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
                                  max_features="sqrt", n_jobs=-1, random_state=seed).fit(X, y)


def train_isolation_forest(X_legit, seed=42):
    return IsolationForest(n_estimators=200, max_samples=4096, random_state=seed, n_jobs=-1).fit(X_legit)


def isolation_score(model: IsolationForest, X) -> np.ndarray:
    return -model.score_samples(X)  # plus élevé = plus anormal


def rule_based_score(X_raw: np.ndarray) -> np.ndarray:
    """Système de règles simple, sur les variables NON normalisées :
    montant > 5x l'habitude, nouvel appareil, nuit, vidage du solde, IP étrangère, rafale."""
    f = {name: X_raw[:, i] for i, name in enumerate(FEATURE_NAMES)}
    score = (
        (f["amount_ratio_user_mean"] > np.log1p(5)).astype(float)
        + f["is_new_device"]
        + f["is_night"]
        + f["is_near_full_drain"]
        + f["is_foreign_ip"]
        + (f["tx_count_1h"] >= 4).astype(float)
    )
    return score
