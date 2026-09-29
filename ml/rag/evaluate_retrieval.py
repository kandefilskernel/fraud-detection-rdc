"""
Évaluation de la recherche de cas similaires (partie « R » de l'assistant d'enquête).

Sur la période de TEST (jamais vue par l'archive ni par le modèle) :

  1. Typologie : pour chaque fraude réelle, la typologie majoritaire des cas confirmés les
     plus proches est-elle la bonne ? Comparé à la réponse « typologie la plus fréquente du
     canal ». C'est ce que l'assistant propose comme « hypothèse principale ».
  2. Tri des alertes : parmi les transactions alertées (3 % les plus risquées selon le modèle
     hybride), la part de fraudes chez les voisins distingue-t-elle les vraies fraudes des
     faux positifs ? Comparé au score du modèle lui-même (même ensemble d'alertes).

    python -m ml.rag.evaluate_retrieval [--k 8]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.features.feature_engineering import FEATURE_NAMES
from ml.rag.build_case_archive import ARTIFACTS, OUT, PROCESSED
from shared.investigation.case_index import CaseIndex, summarize_neighbors

ROOT = Path(__file__).absolute().parents[2]


def evaluate(k: int = 8, alert_rate: float = 0.03) -> dict:
    index = CaseIndex.load(OUT)
    prep = json.loads((ARTIFACTS / "preprocessing.json").read_text(encoding="utf-8"))
    test_from = pd.Timestamp(prep["split_bounds"]["test_from"])
    feats = pd.read_csv(PROCESSED / "features.csv", usecols=["transaction_id", "timestamp", "channel", *FEATURE_NAMES],
                        parse_dates=["timestamp"], low_memory=False)
    feats = feats[feats.timestamp >= test_from]
    preds = pd.read_csv(ROOT / "ml" / "reports" / "test_predictions.csv",
                        usecols=["transaction_id", "is_fraud_true", "fraud_type_true", "score_hybrid"])
    df = feats.merge(preds, on="transaction_id")

    def neighbors(row) -> dict:
        f = {n: row[n] for n in FEATURE_NAMES}
        return summarize_neighbors(index.search(f, row["channel"], k=k))

    # 1) typologie des fraudes réelles
    archive_typo = Counter((c["channel"], c["typology"]) for c in index.cases if c.get("typology"))
    majority = {ch: max((t for (c, t) in archive_typo if c == ch), key=lambda t: archive_typo[(ch, t)])
                for ch in {c for c, _ in archive_typo}}
    frauds = df[(df.is_fraud_true == 1) & df.fraud_type_true.notna() & (df.fraud_type_true != "FRIENDLY_FRAUD")]
    rows = []
    for _, r in frauds.iterrows():
        s = neighbors(r)
        top = s["typologies"][0]["typology"] if s["typologies"] else None
        rows.append({"true": r["fraud_type_true"], "pred": top, "channel": r["channel"]})
    t = pd.DataFrame(rows)
    t["ok"] = t.true == t.pred
    t["baseline_ok"] = t.true == t.channel.map(majority)
    per_type = t.groupby("true").agg(n=("ok", "size"), accuracy=("ok", "mean")).round(3)

    # 2) tri des alertes
    alerts = pd.concat([g.nlargest(int(len(g) * alert_rate), "score_hybrid") for _, g in df.groupby("channel")])
    share = np.array([neighbors(r)["fraud_share"] or 0.0 for _, r in alerts.iterrows()])
    y = alerts.is_fraud_true.to_numpy()
    combo = 0.5 * share + 0.5 * pd.Series(alerts.score_hybrid.to_numpy()).rank(pct=True).to_numpy()

    result = {
        "k": k,
        "archive": {"n_cases": len(index.cases)},
        "typology": {
            "n_frauds": int(len(t)),
            "accuracy": round(float(t.ok.mean()), 3),
            "baseline_majority_accuracy": round(float(t.baseline_ok.mean()), 3),
            "per_type": per_type.to_dict(orient="index"),
        },
        "alert_triage": {
            "n_alerts": int(len(alerts)),
            "fraud_rate_in_alerts": round(float(y.mean()), 3),
            "neighbors_fraud_share": {"roc_auc": round(roc_auc_score(y, share), 3),
                                      "pr_auc": round(average_precision_score(y, share), 3)},
            "hybrid_score": {"roc_auc": round(roc_auc_score(y, alerts.score_hybrid), 3),
                             "pr_auc": round(average_precision_score(y, alerts.score_hybrid), 3)},
            "average_of_ranks": {"roc_auc": round(roc_auc_score(y, combo), 3),
                                 "pr_auc": round(average_precision_score(y, combo), 3)},
        },
    }
    out = ROOT / "ml" / "reports" / "rag_retrieval_eval.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=8)
    evaluate(ap.parse_args().k)
