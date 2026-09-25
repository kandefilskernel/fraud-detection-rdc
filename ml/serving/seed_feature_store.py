"""
Amorçage du feature store Redis : rejoue l'historique des transactions jusqu'à une date
de coupure, puis copie les profils comportementaux dans Redis.

Par défaut la coupure est le début de la période de TEST : l'API démarre alors avec les
profils connus à ce moment-là, et le simulateur peut rejouer la période de test en direct
(les scores obtenus doivent être identiques aux scores hors ligne : test de parité).

Usage :
    python -m ml.serving.seed_feature_store                 # coupure = début du test
    python -m ml.serving.seed_feature_store --until all     # tout l'historique
    python -m ml.serving.seed_feature_store --redis-url redis://localhost:6379/0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import redis

from ml.features.feature_engineering import (FEATURE_NAMES, RAW_DIR, BehavioralFeatureExtractor,
                                             _period_start)
from ml.serving.feature_store import PREFIX, RedisFeatureStore
from shared.schemas.unified_transaction import canonical_id

ARTIFACTS = Path("ml/artifacts")


def replay(until: str | None, raw_dir: Path = RAW_DIR, seq_len: int = 10):
    tx = pd.read_csv(raw_dir / "transactions.csv")
    users = pd.read_csv(raw_dir / "users.csv")
    period_start = _period_start(tx["timestamp"], raw_dir)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx = tx.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    if until:
        tx = tx[tx["timestamp"] < pd.Timestamp(until)].reset_index(drop=True)

    work = tx[["user_id", "tx_type", "channel", "amount_usd", "balance_before_usd", "status",
               "device_id", "device_type", "access_channel", "ip_country", "location_province",
               "counterparty_id", "agent_id", "merchant_id", "merchant_category",
               "merchant_country"]].copy()
    work = work.astype(object).where(work.notna(), None)
    for col in ("user_id", "device_id", "counterparty_id", "agent_id", "merchant_id"):
        work[col] = work[col].map(canonical_id)
    work["ts"] = (tx["timestamp"] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    work["hour"] = tx["timestamp"].dt.hour
    work["weekday"] = tx["timestamp"].dt.weekday
    work["day"] = tx["timestamp"].dt.day

    ext = BehavioralFeatureExtractor(users, period_start)
    rows = [ext.process(r) for r in work.to_dict("records")]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES).astype(np.float32).to_numpy()

    prep = json.loads((ARTIFACTS / "preprocessing.json").read_text(encoding="utf-8"))
    scaler = joblib.load(ARTIFACTS / "scaler.pkl")
    Xs = np.clip(scaler.transform(X), -prep["clip"], prep["clip"]).astype(np.float32)

    last = defaultdict(lambda: deque(maxlen=seq_len - 1))
    for i, uid in enumerate(tx["user_id"].to_numpy()):
        last[uid].append(i)
    sequences = {uid: [Xs[i] for i in idx] for uid, idx in last.items()}
    return ext, sequences, period_start, len(tx)


def main():
    p = argparse.ArgumentParser(description="Amorçage du feature store Redis")
    p.add_argument("--until", default="test",
                   help="'test' (début de la période de test), 'all', ou une date AAAA-MM-JJ")
    p.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    a = p.parse_args()

    until = None
    if a.until == "test":
        prep = json.loads((ARTIFACTS / "preprocessing.json").read_text(encoding="utf-8"))
        until = prep["split_bounds"]["test_from"]
    elif a.until != "all":
        until = a.until

    t0 = time.perf_counter()
    print(f"Rejeu de l'historique{' jusqu au ' + str(until) if until else ''}...")
    ext, sequences, period_start, n = replay(until)
    print(f"    {n:,} transactions rejouées en {time.perf_counter() - t0:.0f} s")

    r = redis.Redis.from_url(a.redis_url)
    old = list(r.scan_iter(f"{PREFIX}:*", count=5000))
    for i in range(0, len(old), 5000):
        r.delete(*old[i:i + 5000])
    store = RedisFeatureStore(r, period_start.timestamp())
    stats = store.bulk_load(ext, sequences)
    r.set(f"{PREFIX}:meta", json.dumps({"seeded_until": str(until), "period_start": str(period_start),
                                        "n_transactions": n, "seeded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    print(f"[OK] Redis amorcé en {time.perf_counter() - t0:.0f} s : {stats}")


if __name__ == "__main__":
    main()
