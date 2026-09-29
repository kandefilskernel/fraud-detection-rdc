"""
Construit l'archive des cas passés de l'assistant d'enquête (RAG).

Une vraie cellule anti-fraude dispose d'un historique de dossiers instruits. On le
reconstitue à partir des données AVANT la période de test (aucune fuite : la démo et
l'évaluation se déroulent après). Deux origines, comme dans la réalité :

    alertes instruites : les 3 % de transactions (par canal) que la branche XGBoost aurait le
                         plus alertées, sur les périodes early_stop + validation que les arbres
                         n'ont pas apprises. L'analyste appelle le client : on suppose que
                         l'enquête établit la vérité, sauf pour 10 % des fraudes classées à
                         tort sans suite (client injoignable, complice, trop tard) - hypothèse
                         documentée dans docs/HYPOTHESES_DONNEES.md.
    plaintes clients   : fraudes signalées par les victimes (étiquette observée) sur toute la
                         période, y compris celles que le modèle n'aurait pas alertées.
                         Les « plaintes de complaisance » (FRIENDLY_FRAUD) sont des opérations
                         légitimes : l'enquête les classe sans suite.

Sorties (copiées dans l'image du back-office) :
    services/backoffice-api/knowledge/case_archive.npz   vecteurs, standardisation, poids
    services/backoffice-api/knowledge/case_archive.json  description des cas (sans identifiant)

    python -m ml.rag.build_case_archive
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features.feature_engineering import FEATURE_NAMES
from shared.investigation.case_index import CLEARED, CONFIRMED, CaseIndex
from shared.investigation.signals import describe, mitigating

ROOT = Path(__file__).absolute().parents[2]
PROCESSED = ROOT / "ml" / "data" / "processed"
RAW = ROOT / "ml" / "data" / "raw"
ARTIFACTS = ROOT / "ml" / "artifacts"
OUT = ROOT / "services" / "backoffice-api" / "knowledge"
EXCLUDED_TYPES = {"FRIENDLY_FRAUD"}
RAW_COLS = ["transaction_id", "operator", "access_channel", "location_province", "fraud_episode_id",
            "fraud_reported_at"]


def importance_weights(xgb_model, n: int, floor: float = 0.15) -> np.ndarray:
    """Poids par variable = racine du gain XGBoost normalisé (moyenne 1). Une variable jamais
    utilisée par les arbres garde un petit poids : elle peut rester utile à l'analyste."""
    gain = xgb_model.get_booster().get_score(importance_type="gain")
    g = np.array([gain.get(f"f{i}", 0.0) for i in range(n)], dtype=np.float64)
    w = np.sqrt(g / g.sum()) if g.sum() > 0 else np.ones(n)
    w = np.maximum(w / w.mean(), floor)
    return (w / w.mean()).astype(np.float32)


def load_period(test_from: pd.Timestamp) -> pd.DataFrame:
    usecols = ["transaction_id", "timestamp", "channel", "tx_type", "amount_usd", "is_fraud", "fraud_type",
               *FEATURE_NAMES]
    feats = pd.read_csv(PROCESSED / "features.csv", usecols=usecols, parse_dates=["timestamp"], low_memory=False)
    feats = feats[feats.timestamp < test_from]
    truth = pd.read_csv(PROCESSED / "meta.csv", usecols=["transaction_id", "is_fraud_true", "fraud_type_true"])
    raw = pd.read_csv(RAW / "transactions.csv", usecols=RAW_COLS)
    return feats.merge(truth, on="transaction_id", how="left").merge(raw, on="transaction_id", how="left")


def to_case(row, outcome: str, idx: int, source: str, score: float | None = None) -> dict:
    f = {n: row[n] for n in FEATURE_NAMES}
    case = {
        "id": f"H-{idx:05d}",
        "source_id": row["transaction_id"],       # jamais transmis au modèle de langage
        "source": source,                         # ALERTE_INSTRUITE | PLAINTE_CLIENT
        "date": row["timestamp"].strftime("%Y-%m-%d"),
        "hour": int(row["timestamp"].hour),
        "channel": row["channel"],
        "operator": row["operator"] if isinstance(row["operator"], str) else None,
        "tx_type": row["tx_type"],
        "access_channel": row["access_channel"],
        "amount_usd": round(float(row["amount_usd"]), 2),
        "province": row["location_province"],
        "outcome": outcome,
        "facts": describe(f, row["tx_type"], max_facts=6),
    }
    if score is not None:
        case["alert_score"] = round(float(score), 4)
    if outcome == CONFIRMED:
        case["typology"] = row["fraud_type_true"]
        case["episode"] = row["fraud_episode_id"] if isinstance(row["fraud_episode_id"], str) else None
        if isinstance(row["fraud_reported_at"], str):
            delay = (pd.Timestamp(row["fraud_reported_at"]) - row["timestamp"]).total_seconds() / 86400
            case["report_delay_days"] = round(max(delay, 0.0), 1)
    else:
        case["typology"] = None
        case["mitigating"] = mitigating(f, row["tx_type"])[:4]
    return case


def build(alert_rate: float = 0.03, investigation_miss_rate: float = 0.10, seed: int = 7,
          out_dir: Path = OUT) -> dict:
    prep = json.loads((ARTIFACTS / "preprocessing.json").read_text(encoding="utf-8"))
    bounds = {k: pd.Timestamp(v) for k, v in prep["split_bounds"].items()}
    scaler = joblib.load(ARTIFACTS / "scaler.pkl")
    xgb_model = joblib.load(ARTIFACTS / "xgb_branch.pkl")
    rng = np.random.default_rng(seed)
    df = load_period(bounds["test_from"])
    print(f"période archivée : {bounds['start'].date()} -> {bounds['test_from'].date()} ({len(df):,} transactions)")
    X = np.clip(scaler.transform(df[FEATURE_NAMES].to_numpy(np.float64)), -prep["clip"], prep["clip"])

    # 1) alertes instruites : périodes que les arbres n'ont pas apprises (early_stop + validation)
    unseen = (df.timestamp >= bounds["early_stop_from"]).to_numpy()
    cand = df[unseen].assign(_score=xgb_model.predict_proba(X[unseen].astype(np.float32))[:, 1])
    alerts = pd.concat([g.nlargest(int(len(g) * alert_rate), "_score") for _, g in cand.groupby("channel")])
    truly = alerts.is_fraud_true.fillna(0).astype(int).to_numpy() == 1
    missed = truly & (rng.random(len(alerts)) < investigation_miss_rate)
    alert_confirmed = truly & ~missed

    # 2) plaintes clients (étiquette observée) non déjà couvertes par une alerte
    complaints = df[(df.is_fraud == 1) & ~df.fraud_type.isin(EXCLUDED_TYPES)
                    & ~df.transaction_id.isin(alerts.transaction_id)]

    cases, rows = [], []
    for (_, r), conf in zip(alerts.iterrows(), alert_confirmed):
        cases.append(to_case(r, CONFIRMED if conf else CLEARED, len(cases), "ALERTE_INSTRUITE", r["_score"]))
        rows.append(r.name)
    for _, r in complaints.iterrows():
        cases.append(to_case(r, CONFIRMED, len(cases), "PLAINTE_CLIENT"))
        rows.append(r.name)
    vectors = X[df.index.get_indexer(rows)].astype(np.float32)

    weights = importance_weights(xgb_model, len(FEATURE_NAMES))
    index = CaseIndex(vectors, cases, FEATURE_NAMES, scaler.mean_.astype(np.float64),
                      scaler.scale_.astype(np.float64), weights, float(prep["clip"]))
    confirmed = [c for c in cases if c["outcome"] == CONFIRMED]
    meta = {
        "period": {"from": str(bounds["start"]), "to": str(bounds["test_from"])},
        "alert_rate": alert_rate,
        "investigation_miss_rate": investigation_miss_rate,
        "n_alerts": int(len(alerts)),
        "n_alerts_confirmed": int(alert_confirmed.sum()),
        "n_alerts_cleared": int((~alert_confirmed).sum()),
        "n_cleared_but_fraud": int(missed.sum()),
        "n_complaints": int(len(complaints)),
        "n_confirmed": len(confirmed),
        "typologies": pd.Series([c["typology"] for c in confirmed]).value_counts().to_dict(),
        "top_weighted_features": [FEATURE_NAMES[i] for i in np.argsort(-weights)[:12]],
        "note": "Archive reconstituée à partir de données synthétiques antérieures à la période de test.",
    }
    index.save(out_dir, vectors, meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "note"}, ensure_ascii=False, indent=1))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alert-rate", type=float, default=0.03)
    ap.add_argument("--miss-rate", type=float, default=0.10, help="fraudes alertées classées à tort")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    build(a.alert_rate, a.miss_rate, out_dir=a.out)
