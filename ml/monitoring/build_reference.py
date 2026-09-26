"""
Distribution de référence pour la détection de dérive.

    - variables brutes : échantillon de la période d'ENTRAÎNEMENT ;
    - scores du modèle hybride : échantillon de la période de VALIDATION (hors
      entraînement des branches, donc représentatif des scores en production).

Écrit ml/artifacts/drift_reference.npz (lu par drift_detector.py).
Usage : python -m ml.monitoring.build_reference
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features.feature_engineering import FEATURE_NAMES
from ml.models.hybrid_ensemble import HybridEnsemble

ARTIFACTS = "ml/artifacts"
N = 10_000


def main(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    split = np.load("ml/data/processed/split.npy")
    train_rows = np.flatnonzero(split <= 1)
    val_rows = np.flatnonzero(split == 2)

    feats = pd.read_csv("ml/data/processed/features.csv", usecols=FEATURE_NAMES).to_numpy(np.float32)
    ref_features = feats[rng.choice(train_rows, size=min(N, len(train_rows)), replace=False)]

    X = np.load("ml/data/processed/X_all.npy")
    seq = np.load("ml/data/processed/seq_idx.npy")
    model = HybridEnsemble.load(ARTIFACTS)
    rows = np.sort(rng.choice(val_rows, size=min(N, len(val_rows)), replace=False))
    ref_scores = model.predict_proba(X, seq, rows).astype(np.float32)

    np.savez_compressed(f"{ARTIFACTS}/drift_reference.npz", features=ref_features, scores=ref_scores,
                        feature_names=np.array(FEATURE_NAMES))
    print(f"[OK] référence de dérive : {len(ref_features)} transactions (entraînement), "
          f"{len(ref_scores)} scores (validation) -> {ARTIFACTS}/drift_reference.npz")


if __name__ == "__main__":
    main()
