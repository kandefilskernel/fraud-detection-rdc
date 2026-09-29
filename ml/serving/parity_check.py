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
from datetime import datetime

import numpy as np
import pandas as pd
import redis

from ml.features.feature_engineering import reported_seconds
from ml.serving.feature_store import PREFIX, RedisFeatureStore
from ml.serving.scorer import HybridScorer
from shared.schemas.unified_transaction import UnifiedTransaction, canonical_id


def load_test_transactions(n: int) -> tuple[pd.DataFrame, list[tuple[float, dict]]]:
    """Les n premières transactions du test, et les signalements de fraude reçus pendant le
    rejeu (ils alimentent le profil de réputation, comme le topic fraud.confirmed en production)."""
    prep = json.loads(open("ml/artifacts/preprocessing.json", encoding="utf-8").read())
    test_from = pd.Timestamp(prep["split_bounds"]["test_from"])
    tx = pd.read_csv("ml/data/raw/transactions.csv", low_memory=False)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx = tx.sort_values(["timestamp", "transaction_id"], kind="mergesort")
    test = tx[tx["timestamp"] >= test_from].head(n)
    reports = []
    rep_s = reported_seconds(tx)
    if rep_s is not None:
        to_s = lambda t: (t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)  # noqa: E731
        start_s, end_s = to_s(test_from), to_s(test["timestamp"].max())
        window = (rep_s >= start_s) & (rep_s <= end_s)
        for (_, row), s in zip(tx[window].iterrows(), rep_s[window]):
            reports.append((s, {"transaction_id": str(row["transaction_id"]), "ts": to_s(row["timestamp"]),
                                "tx_type": row["tx_type"],
                                **{k: canonical_id(row[k]) for k in ("user_id", "device_id", "counterparty_id",
                                                                     "agent_id", "merchant_id")}}))
        reports.sort(key=lambda e: e[0])
    return test.astype(object).where(test.notna(), None), reports


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
    preds = pd.read_csv("ml/reports/test_predictions.csv").set_index("transaction_id")
    # scores hors ligne du modèle EN PRODUCTION (score_production depuis la sélection de modèle)
    offline = preds["score_production"] if "score_production" in preds else preds["score_hybrid"]

    tx, reports = load_test_transactions(a.n)
    diffs, lat, j = [], [], 0
    for row in tx.to_dict("records"):
        u = row_to_unified(row)
        ts = (u.timestamp - datetime(1970, 1, 1)).total_seconds()
        while j < len(reports) and reports[j][0] < ts:   # signalements reçus avant cette transaction
            store.apply_report(reports[j][1])
            j += 1
        t0 = time.perf_counter()
        out = store.compute(u.to_feature_input(u.status or "SUCCESS"), scorer.scale,
                            lambda f, x, h: (scorer.score(x, h, explain=True), None))
        res = out["decided"]
        lat.append((time.perf_counter() - t0) * 1000)
        diffs.append(abs(res.probability - offline[u.transaction_id]))
    diffs, lat = np.array(diffs), np.array(lat)
    print(f"{len(diffs)} transactions rejouées via Redis, {j} signalements de fraude intégrés en cours de route")
    print(f"écart max de probabilité vs hors ligne : {diffs.max():.2e}  (moyen {diffs.mean():.2e})")
    print(f"latence bout-en-bout (Redis + variables + modèle + SHAP) : "
          f"p50={np.percentile(lat, 50):.1f} ms  p99={np.percentile(lat, 99):.1f} ms")
    ok = diffs.max() < 1e-4
    print("[OK] parité entraînement/production vérifiée" if ok else "[ÉCHEC] écart de parité")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
