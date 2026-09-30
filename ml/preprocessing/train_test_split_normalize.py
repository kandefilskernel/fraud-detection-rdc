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

from ml.features.feature_engineering import FEATURE_NAMES, PROCESSED_DIR, RAW_DIR, load_or_build_features

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


def add_ground_truth(table: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Vérité terrain, pour l'ÉVALUATION seulement (jamais pour l'entraînement) :
    is_fraud = ce que l'opérateur sait (fraudes signalées) ; is_fraud_true = la réalité
    (fraudes jamais signalées incluses, contestations abusives exclues), lue dans label_noise.csv."""
    table = table.copy()
    table["is_fraud_true"] = table["is_fraud"].astype(int)
    table["fraud_type_true"] = table["fraud_type"]
    noise_path = Path(raw_dir) / "label_noise.csv"
    if noise_path.exists():
        noise = pd.read_csv(noise_path).set_index("transaction_id")
        ids = table["transaction_id"]
        known = ids.isin(noise.index)
        table.loc[known, "is_fraud_true"] = ids[known].map(noise["true_label"]).astype(int).to_numpy()
        table.loc[known, "fraud_type_true"] = ids[known].map(noise["true_fraud_type"]).to_numpy()
    return table


def run_preprocessing(seq_len: int = SEQ_LEN, rebuild_features: bool = False,
                      profile_scope: str = "unified", processed_dir: Path = PROCESSED_DIR,
                      artifacts_dir: Path = ARTIFACTS_DIR, raw_dir: Path = RAW_DIR) -> None:
    """profile_scope="silo" (expérience C2) : profils séparés par canal. À écrire dans un
    dossier d'expérience (processed_dir ET artifacts_dir), jamais dans ml/artifacts : le
    service de scoring utilise le profil unifié.
    raw_dir : données brutes (ex. export d'un opérateur importé par ml.onboarding)."""
    processed_dir, artifacts_dir, raw_dir = Path(processed_dir), Path(artifacts_dir), Path(raw_dir)
    if profile_scope != "unified" and artifacts_dir.absolute() == ARTIFACTS_DIR.absolute():
        raise ValueError("profil « silo » : choisir un --artifacts-dir d'expérience, pas ml/artifacts")
    features_csv = processed_dir / "features.csv"
    if features_csv.exists() and not rebuild_features:
        table = pd.read_csv(features_csv, low_memory=False)
        print(f"Variables lues depuis {features_csv}")
    else:
        print(f"Calcul des variables comportementales (rejeu chronologique, profil {profile_scope})...")
        table = load_or_build_features(raw_dir / "transactions.csv", raw_dir / "users.csv",
                                       cache_path=features_csv, profile_scope=profile_scope)

    table["timestamp"] = pd.to_datetime(table["timestamp"])
    if not table["timestamp"].is_monotonic_increasing:
        raise ValueError("features.csv doit être trié chronologiquement")
    table = add_ground_truth(table, raw_dir)

    split, bounds = temporal_split(table["timestamp"])
    train_mask = split <= 1
    X = table[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = table["is_fraud"].to_numpy(dtype=np.int8)

    scaler = StandardScaler().fit(X[train_mask])
    X_scaled = np.clip(scaler.transform(X), -CLIP, CLIP).astype(np.float32)
    # en silo, la séquence du LSTM ne contient que les transactions du même canal
    seq_key = table["user_id"] if profile_scope == "unified" else table["user_id"] + "|" + table["channel"]
    seq_idx = build_sequence_index(seq_key, seq_len)

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "X_all.npy", X_scaled)
    np.save(processed_dir / "y_all.npy", y)
    np.save(processed_dir / "seq_idx.npy", seq_idx)
    np.save(processed_dir / "split.npy", split)
    table[["transaction_id", "timestamp", "user_id", "channel", "tx_type", "amount_usd",
           "is_fraud", "fraud_type", "is_fraud_true", "fraud_type_true"]].to_csv(processed_dir / "meta.csv",
                                                                                index=False)
    # Matrices tabulaires prêtes à l'emploi (baselines, notebooks)
    for name, mask in [("train", train_mask), ("val", split == 2), ("test", split == 3)]:
        np.save(processed_dir / f"X_{name}.npy", X_scaled[mask])
        np.save(processed_dir / f"y_{name}.npy", y[mask].astype(np.float32))

    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    (artifacts_dir / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=4), encoding="utf-8")

    stats = {}
    for code, name in enumerate(SPLIT_NAMES):
        m = split == code
        stats[name] = {"n": int(m.sum()), "n_fraud": int(y[m].sum()),
                       "fraud_rate": round(float(y[m].mean()), 5) if m.any() else None,
                       "by_channel": {c: {"n": int((m & (table.channel == c)).sum()),
                                          "n_fraud": int(y[m & (table.channel == c).to_numpy()].sum())}
                                      for c in sorted(table.channel.unique())}}
    neg, pos = int((y[train_mask] == 0).sum()), int(y[train_mask].sum())
    (artifacts_dir / "class_weights.json").write_text(json.dumps(
        {"negative": neg, "positive": pos, "scale_pos_weight": round(neg / max(pos, 1), 3)}, indent=4),
        encoding="utf-8")
    (artifacts_dir / "preprocessing.json").write_text(json.dumps(
        {"split_bounds": bounds, "seq_len": seq_len, "clip": CLIP, "n_features": len(FEATURE_NAMES),
         "profile_scope": profile_scope, "splits": stats}, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] {len(table):,} transactions, {len(FEATURE_NAMES)} variables, séquences de {seq_len}, "
          f"profil {profile_scope} -> {processed_dir}")
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
    p.add_argument("--profile-scope", choices=["unified", "silo"], default="unified",
                   help="expérience C2 : 'silo' = profils séparés par canal")
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    p.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    p.add_argument("--raw-dir", default=str(RAW_DIR), help="données brutes (export opérateur importé)")
    a = p.parse_args()
    run_preprocessing(seq_len=a.seq_len, rebuild_features=a.rebuild_features, profile_scope=a.profile_scope,
                      processed_dir=Path(a.processed_dir), artifacts_dir=Path(a.artifacts_dir),
                      raw_dir=Path(a.raw_dir))
