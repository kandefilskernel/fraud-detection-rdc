"""
Métriques adaptées à la détection de fraude (classes très déséquilibrées).

    - PR-AUC (average precision) : métrique principale ; la ROC-AUC est trop optimiste
      quand 99 % des transactions sont légitimes
    - Rappel à 1 % de faux positifs : « quelle part des fraudes attrape-t-on si l'on
      accepte d'alerter à tort sur 1 % des clients légitimes ? »
    - Précision / rappel / F1 au seuil choisi sur la VALIDATION (jamais sur le test)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "n_fraud": int(y.sum()), "note": "une seule classe présente"}
    pred = (scores >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "n_fraud": int(y.sum()),
        "pr_auc": round(float(average_precision_score(y, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "recall_at_1pct_fpr": round(recall_at_fpr(y, scores, 0.01), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "false_positive_rate": round(float(pred[y == 0].mean()), 5),
    }


def evaluate_by_group(y, scores, threshold, groups) -> dict:
    groups = np.asarray(groups)
    return {str(g): evaluate(y[groups == g], scores[groups == g], threshold) for g in np.unique(groups)}


def recall_by_fraud_type(y, scores, threshold, fraud_type) -> dict:
    """Taux de détection par typologie (transactions frauduleuses du test uniquement)."""
    df = pd.DataFrame({"y": y, "hit": scores >= threshold, "type": fraud_type})
    df = df[df.y == 1]
    return {t: {"n": int(len(g)), "recall": round(float(g.hit.mean()), 4)}
            for t, g in df.groupby("type")}
