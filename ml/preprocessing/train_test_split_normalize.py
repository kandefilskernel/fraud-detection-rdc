"""
Phase 2 (suite) : split temporel, normalisation et séquences pour la branche LSTM.

    - Split TEMPOREL (jamais aléatoire) : on entraîne sur le passé, on évalue sur le futur.
          train (70 % de la période) = fit (85 % du train) + early_stop (15 % final du train)
          val   (15 % suivants)      -> apprentissage du meta-learner et choix du seuil
          test  (15 % finaux)        -> évaluation finale, touché une seule fois
    - StandardScaler ajusté sur le train UNIQUEMENT.
    - Séquences : pour chaque transaction, indices des SEQ_LEN dernières transactions du
      même utilisateur (la transaction courante en dernier, -1 = remplissage). Les données
      ne sont pas dupliquées : la branche LSTM rassemble les lignes à la volée.

Usage :
    python -m ml.preprocessing.train_test_split_normalize
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.features.feature_engineering import FEATURE_NAMES, PROCESSED_DIR, load_or_build_features

ARTIFACTS_DIR = Path("ml/artifacts")
SPLIT_NAMES = ["train", "early_stop", "val", "test"]
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
EARLY_STOP_FRAC = 0.15  # part finale de la période d'entraînement
SEQ_LEN = 10
CLIP = 10.0


def temporal_split(timestamps: pd.Series) -> tuple[np.ndarray, dict]:
    ts = pd.to_datetime(timestamps)
    t0, t1 = ts.min(), ts.max()
    span = t1 - t0
    cut_train = t0 + span * TRAIN_FRAC
    cut_es = t0 + span * TRAIN_FRAC * (1 - EARLY_STOP_FRAC)
    cut_val = t0 + span * (TRAIN_FRAC + VAL_FRAC)
    split = np.select([ts < cut_es, ts < cut_train, ts < cut_val], [0, 1, 2], default=3).astype(np.int8)
    bounds = {"start": t0, "early_stop_from": cut_es, "val_from": cut_train, "test_from": cut_val, "end": t1}
    return split, {k: str(v) for k, v in bounds.items()}


def build_sequence_index(user_ids: pd.Series, seq_len: int = SEQ_LEN) -> np.ndarray:
    """seq[i, -1] = i ; seq[i, -2] = transaction précédente du même utilisateur ; etc.
    Les lignes doivent être triées chronologiquement."""
    n = len(user_ids)
    rows = pd.Series(np.arange(n))
    prev = rows.groupby(user_ids.to_numpy()).shift(1).fillna(-1).astype(np.int64).to_numpy()
    seq = np.full((n, seq_len), -1, dtype=np.int32)
    cur = np.arange(n, dtype=np.int64)
    for k in range(seq_len - 1, -1, -1):
        seq[:, k] = cur
        valid = cur >= 0
        nxt = np.full(n, -1, dtype=np.int64)
        nxt[valid] = prev[cur[valid]]
        cur = nxt
    return seq


def run_preprocessing(seq_len: int = SEQ_LEN, rebuild_features: bool = False) -> None:
    features_csv = PROCESSED_DIR / "features.csv"
    if features_csv.exists() and not rebuild_features:
        table = pd.read_csv(features_csv)
        print(f"Variables lues depuis {features_csv}")
    else:
        print("Calcul des variables comportementales (rejeu chronologique)...")
        table = load_or_build_features(cache_path=features_csv)

    table["timestamp"] = pd.to_datetime(table["timestamp"])
    if not table["timestamp"].is_monotonic_increasing:
        raise ValueError("features.csv doit être trié chronologiquement")

    split, bounds = temporal_split(table["timestamp"])
    train_mask = split <= 1
    X = table[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = table["is_fraud"].to_numpy(dtype=np.int8)

    scaler = StandardScaler().fit(X[train_mask])
    X_scaled = np.clip(scaler.transform(X), -CLIP, CLIP).astype(np.float32)
    seq_idx = build_sequence_index(table["user_id"], seq_len)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_all.npy", X_scaled)
    np.save(PROCESSED_DIR / "y_all.npy", y)
    np.save(PROCESSED_DIR / "seq_idx.npy", seq_idx)
    np.save(PROCESSED_DIR / "split.npy", split)
    table[["transaction_id", "timestamp", "user_id", "channel", "tx_type", "amount_usd",
           "is_fraud", "fraud_type"]].to_csv(PROCESSED_DIR / "meta.csv", index=False)
    # Matrices tabulaires prêtes à l'emploi (baselines, notebooks)
    for name, mask in [("train", train_mask), ("val", split == 2), ("test", split == 3)]:
        np.save(PROCESSED_DIR / f"X_{name}.npy", X_scaled[mask])
        np.save(PROCESSED_DIR / f"y_{name}.npy", y[mask].astype(np.float32))

    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    (ARTIFACTS_DIR / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=4), encoding="utf-8")

    stats = {}
    for code, name in enumerate(SPLIT_NAMES):
        m = split == code
        stats[name] = {"n": int(m.sum()), "n_fraud": int(y[m].sum()),
                       "fraud_rate": round(float(y[m].mean()), 5) if m.any() else None,
                       "by_channel": {c: {"n": int((m & (table.channel == c)).sum()),
                                          "n_fraud": int(y[m & (table.channel == c).to_numpy()].sum())}
                                      for c in sorted(table.channel.unique())}}
    neg, pos = int((y[train_mask] == 0).sum()), int(y[train_mask].sum())
    (ARTIFACTS_DIR / "class_weights.json").write_text(json.dumps(
        {"negative": neg, "positive": pos, "scale_pos_weight": round(neg / max(pos, 1), 3)}, indent=4),
        encoding="utf-8")
    (ARTIFACTS_DIR / "preprocessing.json").write_text(json.dumps(
        {"split_bounds": bounds, "seq_len": seq_len, "clip": CLIP, "n_features": len(FEATURE_NAMES),
         "splits": stats}, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] {len(table):,} transactions, {len(FEATURE_NAMES)} variables, séquences de {seq_len}")
    for name, s in stats.items():
        print(f"  {name:<11} n={s['n']:>7,}  fraudes={s['n_fraud']:>5,}  taux={s['fraud_rate']}")


@dataclass
class ProcessedData:
    X: np.ndarray          # (n, F) normalisé, ordre chronologique
    y: np.ndarray          # (n,)
    seq_idx: np.ndarray    # (n, SEQ_LEN) indices dans X, -1 = remplissage
    split: np.ndarray      # (n,) code dans SPLIT_NAMES
    meta: pd.DataFrame

    def rows(self, *names: str) -> np.ndarray:
        codes = [SPLIT_NAMES.index(n) for n in names]
        return np.where(np.isin(self.split, codes))[0]


def load_processed(processed_dir: Path = PROCESSED_DIR) -> ProcessedData:
    return ProcessedData(
        X=np.load(processed_dir / "X_all.npy"),
        y=np.load(processed_dir / "y_all.npy").astype(np.int8),
        seq_idx=np.load(processed_dir / "seq_idx.npy"),
        split=np.load(processed_dir / "split.npy"),
        meta=pd.read_csv(processed_dir / "meta.csv"),
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--rebuild-features", action="store_true")
    a = p.parse_args()
    run_preprocessing(seq_len=a.seq_len, rebuild_features=a.rebuild_features)
