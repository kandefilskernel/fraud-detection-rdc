"""
Inférence temps réel d'UNE transaction avec le modèle hybride entraîné.

    x (58 variables normalisées) ──► XGBoost ─────────┐
    [9 transactions précédentes + x] ──► LSTM-Attention ─┼─► méta-apprenant ─► P(fraude)
    x ──► Autoencodeur (erreur de reconstruction) ──────┘

Mode dégradé : si une branche neuronale échoue, sa contribution est remplacée par sa
valeur moyenne (score standardisé nul pour le méta-apprenant) et la réponse l'indique.

Explicabilité, calculée à chaque décision :
    - contributions SHAP exactes de XGBoost (TreeSHAP natif de xgboost, sans dépendance) ;
    - part de chaque branche dans la décision (coefficient × score standardisé) ;
    - poids d'attention du LSTM sur les transactions passées.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import xgboost as xgb

from ml.features.feature_engineering import FEATURE_NAMES
from ml.models.hybrid_ensemble import HybridEnsemble
from ml.models.meta_learner import BRANCH_NAMES

torch.set_num_threads(1)  # une requête = un cœur ; le parallélisme vient des workers uvicorn


@dataclass
class ScoreResult:
    probability: float
    branch_scores: dict
    branch_contributions: dict
    top_features: list
    attention: list
    degraded: bool
    degraded_branches: list


class HybridScorer:
    def __init__(self, artifacts_dir: str | Path):
        d = Path(artifacts_dir)
        self.model = HybridEnsemble.load(d)
        self.metadata = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        prep = json.loads((d / "preprocessing.json").read_text(encoding="utf-8"))
        self.scaler = joblib.load(d / "scaler.pkl")
        self.clip = float(prep["clip"])
        self.seq_len = int(prep["seq_len"])
        self.feature_names = json.loads((d / "feature_names.json").read_text(encoding="utf-8"))
        if self.feature_names != FEATURE_NAMES:
            raise RuntimeError("feature_names.json ne correspond pas au code des variables : réentraîner")
        self.threshold = float(self.metadata["threshold"])
        self._mean = self.scaler.mean_.astype(np.float64)
        self._scale = self.scaler.scale_.astype(np.float64)
        meta_scaler = self.model.meta.named_steps["scale"]
        self._branch_mean = meta_scaler.mean_.astype(np.float64)
        self._branch_std = meta_scaler.scale_.astype(np.float64)
        logreg = self.model.meta.named_steps["logreg"]
        self._coef = logreg.coef_[0].astype(np.float64)
        self._intercept = float(logreg.intercept_[0])
        self._booster = self.model.xgb.get_booster()
        # l'early stopping a retenu best_iteration : ne pas utiliser les arbres suivants
        best = getattr(self.model.xgb, "best_iteration", None)
        self._iter_range = (0, best + 1) if best is not None else (0, 0)
        self.model_version = str(self.metadata.get("trained_at", "inconnue"))

    # ------------------------------------------------------------------ prétraitement
    def scale(self, feats: dict) -> np.ndarray:
        raw = np.array([[feats[f] for f in self.feature_names]], dtype=np.float32)
        # même calcul qu'à l'entraînement (train_test_split_normalize) : transform puis clip
        return np.clip(self.scaler.transform(raw), -self.clip, self.clip).astype(np.float32)[0]

    def _sequence(self, history: list[np.ndarray], x: np.ndarray):
        prev = history[-(self.seq_len - 1):]
        seq = np.zeros((1, self.seq_len, len(x)), dtype=np.float32)
        mask = np.zeros((1, self.seq_len), dtype=bool)
        vecs = prev + [x]
        start = self.seq_len - len(vecs)
        for i, v in enumerate(vecs):
            seq[0, start + i] = v
            mask[0, start + i] = True
        return torch.from_numpy(seq), torch.from_numpy(mask), start

    # ------------------------------------------------------------------ scoring
    def score(self, x: np.ndarray, history: list[np.ndarray], explain: bool | str = "auto",
              top_k: int = 6) -> ScoreResult:
        """explain : True (toujours), False (jamais) ou "auto" : SHAP seulement à partir du
        risque MOYEN (p ≥ seuil/5). Le calcul SHAP exact coûte ~7 ms ; les ~98 % de
        transactions à risque faible n'en ont pas besoin en temps réel."""
        X1 = x.reshape(1, -1)
        degraded = []
        scores = {}

        # prédiction directe sans DMatrix (~0,5 ms au lieu de ~2 ms)
        scores["xgboost"] = float(self._booster.inplace_predict(
            X1, iteration_range=self._iter_range, predict_type="margin")[0])

        attention = []
        try:
            seq, mask, start = self._sequence(history, x)
            with torch.no_grad():
                logit, weights = self.model.lstm(seq, mask, return_attention=True)
            scores["lstm_attention"] = float(logit[0])
            w = weights[0].numpy()
            attention = [round(float(v), 4) for v in w[start:]]  # du plus ancien à l'actuelle
        except Exception:  # noqa: BLE001 — mode dégradé
            degraded.append("lstm_attention")

        try:
            with torch.no_grad():
                err = self.model.ae.reconstruction_error(torch.from_numpy(X1)).numpy()[0]
            scores["autoencoder"] = float(np.log(err + 1e-6))
        except Exception:  # noqa: BLE001
            degraded.append("autoencoder")

        b = np.array([scores.get(n, self._branch_mean[i]) for i, n in enumerate(BRANCH_NAMES)])
        z = (b - self._branch_mean) / self._branch_std
        contrib = self._coef * z
        logit_meta = self._intercept + float(contrib.sum())
        prob = 1.0 / (1.0 + math.exp(-logit_meta))

        top = []
        if explain is True or (explain == "auto" and prob >= self.threshold / 5):
            dm = xgb.DMatrix(X1)
            shap = self._booster.predict(dm, pred_contribs=True,
                                         iteration_range=self._iter_range)[0][:-1]  # dernière col. = biais
            order = np.argsort(-np.abs(shap))[:top_k]
            raw_vals = x.astype(np.float64) * self._scale + self._mean
            top = [{"feature": self.feature_names[i], "shap": round(float(shap[i]), 4),
                    "value": round(float(raw_vals[i]), 4)} for i in order]

        return ScoreResult(
            probability=prob,
            branch_scores={k: round(v, 4) for k, v in scores.items()},
            branch_contributions={n: round(float(c), 4) for n, c in zip(BRANCH_NAMES, contrib)},
            top_features=top,
            attention=attention,
            degraded=bool(degraded),
            degraded_branches=degraded,
        )
