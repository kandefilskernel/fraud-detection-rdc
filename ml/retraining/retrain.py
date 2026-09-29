"""
Réentraînement champion / challenger à partir des données de production.

Sources d'étiquettes :
    - verdicts des analystes sur les alertes (FRAUDE_CONFIRMEE = 1, FAUX_POSITIF = 0) ;
    - plaintes des clients (fraudes laissées passer par le modèle) = 1 ;
    - transactions « mûres » sans plainte après MATURITY_DAYS jours = 0 (pratique des
      émetteurs : une fraude non signalée après ce délai est rare ; bruit assumé).

Découpage TEMPOREL (aucune fuite du futur vers le passé) :
    fit         historique d'entraînement du champion + 1ʳᵉ moitié de la production
    early_stop  tranche d'arrêt précoce historique
    val         validation historique + tranche suivante de production (méta-apprenant, seuil)
    évaluation  DERNIÈRE tranche de production, vue par aucun des deux modèles

Le champion est évalué avec les probabilités qu'il a réellement produites en temps réel
(stockées dans scored_transactions) ; le challenger est promu seulement s'il fait mieux.

Le scaler n'est PAS réajusté : les vecteurs d'historique déjà stockés dans Redis (LSTM)
restent ainsi cohérents avec le nouveau modèle.

Usage :
    python -m ml.retraining.retrain                 # entraîne, compare, promeut si meilleur
    python -m ml.retraining.retrain --dry-run       # compare sans promouvoir
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features.feature_engineering import FEATURE_NAMES
from ml.models.autoencoder_branch import train_autoencoder_branch
from ml.models.baselines import train_random_forest
from ml.models.hybrid_ensemble import HybridEnsemble
from ml.models.lstm_attention_branch import train_lstm_branch
from ml.models.meta_learner import build_meta_learner, meta_coefficients
from ml.models.xgboost_branch import train_xgboost_branch
from ml.preprocessing.train_test_split_normalize import build_sequence_index
from ml.training.evaluation import best_f1_threshold, evaluate

ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", "ml/artifacts"))
PROCESSED = Path("ml/data/processed")
MATURITY_DAYS = float(os.getenv("LABEL_MATURITY_DAYS", "7"))
MIN_LABELED = int(os.getenv("RETRAIN_MIN_LABELED", "2000"))
MIN_FRAUDS = int(os.getenv("RETRAIN_MIN_FRAUDS", "30"))


def log(msg: str) -> None:
    print(f"[retrain] {msg}", flush=True)


# ---------------------------------------------------------------------- données
def load_history() -> pd.DataFrame:
    """Historique vu par le champion : tout ce qui précède sa période de test."""
    prep = json.loads((ARTIFACTS / "preprocessing.json").read_text(encoding="utf-8"))
    b = prep["split_bounds"]
    h = pd.read_csv(PROCESSED / "features.csv", usecols=["transaction_id", "timestamp", "user_id", "channel",
                                                          "is_fraud", *FEATURE_NAMES])
    h["timestamp"] = pd.to_datetime(h["timestamp"])
    h = h[h["timestamp"] < pd.Timestamp(b["test_from"])].copy()
    h["part"] = np.select([h["timestamp"] < pd.Timestamp(b["early_stop_from"]),
                           h["timestamp"] < pd.Timestamp(b["val_from"])], ["fit", "early_stop"], "val")
    h["champion_proba"] = np.nan
    return h.rename(columns={"is_fraud": "y"})


def load_production(engine) -> pd.DataFrame:
    from sqlalchemy import text
    with engine.connect() as c:
        rows = c.execute(text("""SELECT transaction_id, tx_time, user_id, channel, features, label,
                                        fraud_probability FROM scored_transactions ORDER BY tx_time""")).all()
    if not rows:
        return pd.DataFrame()
    p = pd.DataFrame([{"transaction_id": r[0], "timestamp": r[1], "user_id": r[2], "channel": r[3],
                       **{f: r[4].get(f, 0.0) for f in FEATURE_NAMES}, "label": r[5],
                       "champion_proba": r[6]} for r in rows])
    horizon = p["timestamp"].max() - pd.Timedelta(days=MATURITY_DAYS)
    matured = p["label"].isna() & (p["timestamp"] <= horizon)
    p["y"] = np.where(p["label"].notna(), p["label"], np.where(matured, 0, -1))
    p = p[p["y"] >= 0].drop(columns="label").reset_index(drop=True)
    # découpage temporel de la production : 50 % fit, 20 % val, 30 % évaluation
    q = p["timestamp"].rank(pct=True, method="first")
    p["part"] = np.select([q <= 0.5, q <= 0.7], ["fit", "val"], "eval")
    return p


# ---------------------------------------------------------------------- entraînement
def train_challenger(table: pd.DataFrame, seed: int = 42, quick: bool = False):
    table = table.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    prep = json.loads((ARTIFACTS / "preprocessing.json").read_text(encoding="utf-8"))
    scaler = joblib.load(ARTIFACTS / "scaler.pkl")          # figé (cohérence avec Redis)
    X = np.clip(scaler.transform(table[FEATURE_NAMES].to_numpy(np.float32)),
                -prep["clip"], prep["clip"]).astype(np.float32)
    y = table["y"].to_numpy(np.int8)
    seq = build_sequence_index(table["user_id"], prep["seq_len"])
    idx = {k: np.flatnonzero(table["part"].to_numpy() == k) for k in ("fit", "early_stop", "val", "eval")}
    log(f"lignes : " + ", ".join(f"{k}={len(v):,}" for k, v in idx.items()))

    xgb_m = train_xgboost_branch(X[idx["fit"]], y[idx["fit"]], X[idx["early_stop"]], y[idx["early_stop"]],
                                 200 if quick else 600, seed)
    fit, es = idx["fit"], idx["early_stop"]
    # le challenger reprend l'architecture du champion (branches listées dans metadata.json)
    champion = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    ae = None
    if "autoencoder" in champion.get("branch_order", ["autoencoder"]):
        ae = train_autoencoder_branch(X[fit][y[fit] == 0], X[es][y[es] == 0], epochs=3 if quick else 30,
                                      seed=seed, log=lambda m: None)
    lstm = train_lstm_branch(X, y, seq, fit, es, epochs=1 if quick else 8, neg_sample_rate=0.3,
                             seed=seed, log=lambda m: None)
    tree_kind = champion.get("branch_order", ["xgboost"])[0]
    tree = xgb_m
    if tree_kind == "random_forest":
        tree = train_random_forest(X[np.concatenate([fit, es])], y[np.concatenate([fit, es])], seed)
    ens = HybridEnsemble(tree, lstm, ae, tree_kind=tree_kind)
    S_val = ens.branch_scores(X, seq, idx["val"])
    ens.meta = build_meta_learner().fit(S_val, y[idx["val"]])
    ens.threshold = best_f1_threshold(y[idx["val"]], ens.meta.predict_proba(S_val)[:, 1])
    p_eval = ens.predict_proba(X, seq, idx["eval"])
    ref_rows = np.random.default_rng(seed).choice(idx["fit"], size=min(10_000, len(idx["fit"])), replace=False)
    val_sample = np.sort(np.random.default_rng(seed).choice(idx["val"], size=min(10_000, len(idx["val"])),
                                                            replace=False))
    drift_ref = {"features": table.loc[ref_rows, FEATURE_NAMES].to_numpy(np.float32),
                 "scores": ens.predict_proba(X, seq, val_sample).astype(np.float32),
                 "feature_names": np.array(FEATURE_NAMES)}
    return ens, table.iloc[idx["eval"]].reset_index(drop=True), p_eval, drift_ref, prep


# ---------------------------------------------------------------------- orchestration
def run(engine, dry_run: bool = False, quick: bool = False, force: bool = False) -> dict:
    t0 = time.perf_counter()
    prod = load_production(engine)
    n_lab = len(prod)
    n_fraud = int(prod["y"].sum()) if n_lab else 0
    log(f"production étiquetée : {n_lab:,} transactions dont {n_fraud} fraudes")
    if not force and (n_lab < MIN_LABELED or n_fraud < MIN_FRAUDS):
        return {"status": "IGNORE", "reason": f"pas assez d'étiquettes ({n_lab} / {n_fraud} fraudes)"}

    table = pd.concat([load_history(), prod], ignore_index=True)
    ens, ev, p_chal, drift_ref, prep = train_challenger(table, quick=quick)
    y_ev = ev["y"].to_numpy()
    champion_meta = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    m_champ = evaluate(y_ev, ev["champion_proba"].to_numpy(), float(champion_meta["threshold"]))
    m_chal = evaluate(y_ev, p_chal, ens.threshold)
    log(f"évaluation ({len(ev):,} tx, {int(y_ev.sum())} fraudes) : champion PR-AUC={m_champ['pr_auc']} "
        f"rappel={m_champ['recall']} | challenger PR-AUC={m_chal['pr_auc']} rappel={m_chal['recall']}")

    better = (m_chal["pr_auc"] >= m_champ["pr_auc"] + 0.002 and m_chal["recall"] >= m_champ["recall"] - 0.01)
    version = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    status = "CHAMPION" if better and not dry_run else ("CHALLENGER" if better else "REJETE")
    cand = ARTIFACTS / "candidates" / version.replace(":", "-")
    ens.save(cand, {"trained_at": version, "retrained_from": champion_meta.get("trained_at"),
                    "evaluation": {"champion": m_champ, "challenger": m_chal, "n_eval": int(len(ev))},
                    "training_rows": int((table["part"] != "eval").sum())})
    for f in ("scaler.pkl", "feature_names.json", "preprocessing.json", "class_weights.json"):
        if (ARTIFACTS / f).exists():
            shutil.copy2(ARTIFACTS / f, cand / f)
    np.savez_compressed(cand / "drift_reference.npz", **drift_ref)

    if status == "CHAMPION":
        promote(cand)
    report = {"status": status, "version": version, "champion": m_champ, "challenger": m_chal,
              "n_labeled_production": n_lab, "n_frauds_production": n_fraud,
              "duration_s": round(time.perf_counter() - t0, 1), "candidate_dir": str(cand)}
    register(engine, version, status, report)
    log(f"décision : {status} ({report['duration_s']} s)")
    return report


def promote(cand: Path) -> None:
    """Remplace les artefacts du champion. metadata.json est écrit EN DERNIER : c'est lui que
    le scoring-service surveille pour recharger le modèle à chaud."""
    archive = ARTIFACTS / "archive" / json.loads((ARTIFACTS / "metadata.json").read_text(
        encoding="utf-8"))["trained_at"].replace(":", "-")
    archive.mkdir(parents=True, exist_ok=True)
    for f in cand.iterdir():
        if (ARTIFACTS / f.name).exists():
            shutil.copy2(ARTIFACTS / f.name, archive / f.name)
    for f in cand.iterdir():
        if f.name != "metadata.json":
            tmp = ARTIFACTS / (f.name + ".tmp")
            shutil.copy2(f, tmp)
            os.replace(tmp, ARTIFACTS / f.name)
    tmp = ARTIFACTS / "metadata.json.tmp"
    shutil.copy2(cand / "metadata.json", tmp)
    os.replace(tmp, ARTIFACTS / "metadata.json")
    log(f"promu : {cand.name} (ancien champion archivé dans {archive})")


def register(engine, version: str, status: str, report: dict) -> None:
    from sqlalchemy import text
    with engine.begin() as c:
        if status == "CHAMPION":
            c.execute(text("UPDATE model_registry SET status = 'ARCHIVE' WHERE status = 'CHAMPION'"))
        c.execute(text("INSERT INTO model_registry (version, status, metrics, trained_on) "
                       "VALUES (:v, :s, CAST(:m AS JSONB), CAST(:t AS JSONB))"),
                  {"v": version, "s": status,
                   "m": json.dumps({"champion": report["champion"], "challenger": report["challenger"]}),
                   "t": json.dumps({"n_labeled_production": report["n_labeled_production"],
                                    "n_frauds_production": report["n_frauds_production"]})})
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("fraud-rdc-retraining")
            with mlflow.start_run(run_name=version):
                mlflow.log_param("status", status)
                for who in ("champion", "challenger"):
                    for k in ("pr_auc", "roc_auc", "recall", "precision", "f1"):
                        mlflow.log_metric(f"{who}_{k}", report[who][k])
        except Exception as e:  # noqa: BLE001 — le suivi MLflow est optionnel
            log(f"MLflow indisponible : {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true", help="ignorer les seuils minimaux d'étiquettes")
    a = p.parse_args()
    from shared.database.session import make_session_factory
    engine, _ = make_session_factory()
    print(json.dumps(run(engine, a.dry_run, a.quick, a.force), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
