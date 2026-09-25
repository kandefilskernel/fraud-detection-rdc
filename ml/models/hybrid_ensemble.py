"""
Modèle hybride : XGBoost + LSTM-Attention + Autoencodeur, combinés par un meta-learner.

    score_xgb   = log-odds XGBoost sur la transaction courante
    score_lstm  = log-odds LSTM sur la séquence des SEQ_LEN dernières transactions
    score_ae    = log(erreur de reconstruction) de l'autoencodeur
    P(fraude)   = meta_learner([score_xgb, score_lstm, score_ae])
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


class HybridEnsemble:
    def __init__(self, xgb_model, lstm_model: LSTMAttentionBranch, ae_model: AutoencoderBranch,
                 meta=None, threshold: float = 0.5):
        self.xgb = xgb_model
        self.lstm = lstm_model
        self.ae = ae_model
        self.meta = meta
        self.threshold = threshold

    def branch_scores(self, X: np.ndarray, seq_idx: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Matrice (n, 3) des scores des branches, dans l'ordre BRANCH_NAMES."""
        return np.column_stack([
            xgb_logit(self.xgb, X[rows]),
            lstm_logit(self.lstm, X, seq_idx, rows),
            ae_score(self.ae, X[rows]),
        ])

    def predict_proba(self, X: np.ndarray, seq_idx: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return self.meta.predict_proba(self.branch_scores(X, seq_idx, rows))[:, 1]

    def predict(self, X, seq_idx, rows) -> np.ndarray:
        return (self.predict_proba(X, seq_idx, rows) >= self.threshold).astype(int)

    # ------------------------------------------------------------------ persistance
    def save(self, artifacts_dir: str | Path, metadata: dict | None = None) -> None:
        d = Path(artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.xgb, d / "xgb_branch.pkl")
        joblib.dump(self.meta, d / "meta_learner.pkl")
        torch.save(self.lstm.state_dict(), d / "lstm_branch.pt")
        torch.save(self.ae.state_dict(), d / "autoencoder_branch.pt")
        meta = {
            "threshold": self.threshold,
            "input_dim": self.lstm.lstm.input_size,
            "lstm_hidden_dim": self.lstm.lstm.hidden_size,
            "branch_order": BRANCH_NAMES,
            "meta_learner_coefficients": meta_coefficients(self.meta),
            **(metadata or {}),
        }
        (d / "metadata.json").write_text(json.dumps(meta, indent=4, ensure_ascii=False, default=str),
                                         encoding="utf-8")

    @classmethod
    def load(cls, artifacts_dir: str | Path) -> "HybridEnsemble":
        d = Path(artifacts_dir)
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        lstm = LSTMAttentionBranch(meta["input_dim"], hidden_dim=meta["lstm_hidden_dim"])
        lstm.load_state_dict(torch.load(d / "lstm_branch.pt", weights_only=True))
        lstm.eval()
        ae = AutoencoderBranch(meta["input_dim"])
        ae.load_state_dict(torch.load(d / "autoencoder_branch.pt", weights_only=True))
        ae.eval()
        return cls(joblib.load(d / "xgb_branch.pkl"), lstm, ae, joblib.load(d / "meta_learner.pkl"),
                   threshold=meta["threshold"])
