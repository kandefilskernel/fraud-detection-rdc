"""
Test de parité entraînement / production.

Rejoue les N premières transactions de la période de test à travers le chemin de
production (feature store Redis + HybridScorer, une transaction à la fois) et compare
les probabilités à celles calculées hors ligne par train_model.py (test_predictions.csv).

Prérequis : Redis amorcé jusqu'au début du test (python -m ml.serving.seed_feature_store).
Le rejeu modifie les profils : ré-amorcer ensuite avant une démo.

Usage : python -m ml.serving.parity_check --n 2000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import redis

from ml.serving.feature_store import PREFIX, RedisFeatureStore
from ml.serving.scorer import HybridScorer
from shared.schemas.unified_transaction import UnifiedTransaction


def load_test_transactions(n: int) -> pd.DataFrame:
    prep = json.loads(open("ml/artifacts/preprocessing.json", encoding="utf-8").read())
    tx = pd.read_csv("ml/data/raw/transactions.csv")
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx = tx.sort_values(["timestamp", "transaction_id"], kind="mergesort")
    tx = tx[tx["timestamp"] >= pd.Timestamp(prep["split_bounds"]["test_from"])].head(n)
    return tx.astype(object).where(tx.notna(), None)


def row_to_unified(row: dict) -> UnifiedTransaction:
    fields = UnifiedTransaction.model_fields
    data = {k: row[k] for k in fields if k in row and row[k] is not None}
    data["timestamp"] = pd.Timestamp(row["timestamp"]).to_pydatetime()
    return UnifiedTransaction(**data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    a = p.parse_args()

    r = redis.Redis.from_url(a.redis_url)
    meta = json.loads(r.get(f"{PREFIX}:meta"))
    scorer = HybridScorer("ml/artifacts")
    store = RedisFeatureStore(r, pd.Timestamp(meta["period_start"]).timestamp())
    offline = pd.read_csv("ml/reports/test_predictions.csv").set_index("transaction_id")["score_hybrid"]

    tx = load_test_transactions(a.n)
    diffs, lat = [], []
    for row in tx.to_dict("records"):
        u = row_to_unified(row)
        t0 = time.perf_counter()
        out = store.compute(u.to_feature_input(u.status or "SUCCESS"), scorer.scale,
                            lambda f, x, h: (scorer.score(x, h, explain=True), None))
        res = out["decided"]
        lat.append((time.perf_counter() - t0) * 1000)
        diffs.append(abs(res.probability - offline[u.transaction_id]))
    diffs, lat = np.array(diffs), np.array(lat)
    print(f"{len(diffs)} transactions rejouées via Redis")
    print(f"écart max de probabilité vs hors ligne : {diffs.max():.2e}  (moyen {diffs.mean():.2e})")
    print(f"latence bout-en-bout (Redis + variables + modèle + SHAP) : "
          f"p50={np.percentile(lat, 50):.1f} ms  p99={np.percentile(lat, 99):.1f} ms")
    ok = diffs.max() < 1e-4
    print("[OK] parité entraînement/production vérifiée" if ok else "[ÉCHEC] écart de parité")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
