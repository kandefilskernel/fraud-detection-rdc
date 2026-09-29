"""
Modèle hybride : branches combinées par un meta-learner (stacking).

    score_arbres = log-odds de la branche « arbres » sur la transaction courante :
                   XGBoost, ou forêt aléatoire (log(p / (1 − p)))
    score_lstm   = log-odds LSTM sur la séquence des SEQ_LEN dernières transactions
    score_ae     = log(erreur de reconstruction) de l'autoencodeur (branche facultative)
    P(fraude)    = meta_learner([scores des branches retenues])

La liste des branches est enregistrée dans metadata.json (« branch_order ») : le service de
scoring, le réentraînement et la surveillance suivent l'architecture du modèle en production.
Architecture retenue par ml.training.select_model : voir ml/reports/selection_modele.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from ml.models.autoencoder_branch import AutoencoderBranch, ae_score
from ml.models.lstm_attention_branch import LSTMAttentionBranch, lstm_logit
from ml.models.meta_learner import BRANCH_NAMES, meta_coefficients
from ml.models.xgboost_branch import xgb_logit

TREE_BRANCHES = ("xgboost", "random_forest")
TREE_FILES = {"xgboost": "xgb_branch.pkl", "random_forest": "rf_branch.pkl"}


def rf_logit(model, X: np.ndarray) -> np.ndarray:
    p = np.clip(model.predict_proba(X)[:, 1], 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p)).astype(np.float32)


class HybridEnsemble:
    def __init__(self, tree_model, lstm_model: LSTMAttentionBranch, ae_model: AutoencoderBranch | None = None,
                 meta=None, threshold: float = 0.5, tree_kind: str = "xgboost"):
        if tree_kind not in TREE_BRANCHES:
            raise ValueError(f"branche arbres inconnue : {tree_kind}")
        self.tree = tree_model
        self.tree_kind = tree_kind
        self.lstm = lstm_model
        self.ae = ae_model
        self.meta = meta
        self.threshold = threshold
        self.branches = [tree_kind, "lstm_attention"] + (["autoencoder"] if ae_model is not None else [])

    @property
    def xgb(self):
        """Compatibilité : la branche arbres quand c'est XGBoost."""
        return self.tree if self.tree_kind == "xgboost" else None

    def branch_scores(self, X: np.ndarray, seq_idx: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Matrice (n, nb de branches) des scores, dans l'ordre de self.branches."""
        compute = {
            "xgboost": lambda: xgb_logit(self.tree, X[rows]),
            "random_forest": lambda: rf_logit(self.tree, X[rows]),
            "lstm_attention": lambda: lstm_logit(self.lstm, X, seq_idx, rows),
            "autoencoder": lambda: ae_score(self.ae, X[rows]),
        }
        return np.column_stack([compute[b]() for b in self.branches])

    def predict_proba(self, X: np.ndarray, seq_idx: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return self.meta.predict_proba(self.branch_scores(X, seq_idx, rows))[:, 1]

    def predict(self, X, seq_idx, rows) -> np.ndarray:
        return (self.predict_proba(X, seq_idx, rows) >= self.threshold).astype(int)

    # ------------------------------------------------------------------ persistance
    def save(self, artifacts_dir: str | Path, metadata: dict | None = None) -> None:
        d = Path(artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.tree, d / TREE_FILES[self.tree_kind])
        joblib.dump(self.meta, d / "meta_learner.pkl")
        torch.save(self.lstm.state_dict(), d / "lstm_branch.pt")
        if self.ae is not None:
            torch.save(self.ae.state_dict(), d / "autoencoder_branch.pt")
        elif (d / "autoencoder_branch.pt").exists():
            (d / "autoencoder_branch.pt").unlink()          # ne pas laisser une branche d'un ancien modèle
        meta = {
            "threshold": self.threshold,
            "input_dim": self.lstm.lstm.input_size,
            "lstm_hidden_dim": self.lstm.lstm.hidden_size,
            "branch_order": self.branches,
            "meta_learner_coefficients": meta_coefficients(self.meta, self.branches),
            **(metadata or {}),
        }
        (d / "metadata.json").write_text(json.dumps(meta, indent=4, ensure_ascii=False, default=str),
                                         encoding="utf-8")

    @classmethod
    def load(cls, artifacts_dir: str | Path) -> "HybridEnsemble":
        d = Path(artifacts_dir)
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        branches = meta.get("branch_order", BRANCH_NAMES)
        tree_kind = branches[0] if branches[0] in TREE_BRANCHES else "xgboost"
        lstm = LSTMAttentionBranch(meta["input_dim"], hidden_dim=meta["lstm_hidden_dim"])
        lstm.load_state_dict(torch.load(d / "lstm_branch.pt", weights_only=True))
        lstm.eval()
        ae = None
        if "autoencoder" in branches:
            ae = AutoencoderBranch(meta["input_dim"])
            ae.load_state_dict(torch.load(d / "autoencoder_branch.pt", weights_only=True))
            ae.eval()
        return cls(joblib.load(d / TREE_FILES[tree_kind]), lstm, ae, joblib.load(d / "meta_learner.pkl"),
                   threshold=meta["threshold"], tree_kind=tree_kind)
