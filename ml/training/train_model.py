"""
Phase 3 : entraînement et comparaison des modèles.

Étapes :
    1. Branches du modèle hybride, entraînées sur « train » (arrêt précoce sur « early_stop »)
    2. Meta-learner entraîné sur « val » (scores hors échantillon des branches)
    3. Seuil de décision choisi sur « val » (F1 maximal)
    4. Baselines entraînées sur la même période, seuil choisi de la même façon
    5. Ablations : meta-learner sans l'une ou l'autre branche
    6. Évaluation UNIQUE sur « test » : global, par canal (Mobile Money / Visa),
       par typologie de fraude

Sorties :
    ml/artifacts/        modèles + metadata.json (seuil, coefficients, métriques)
    ml/reports/          phase3_results.json, phase3_model_comparison.csv, test_predictions.csv

Usage (depuis la racine du projet) :
    python -m ml.training.train_model            # complet
    python -m ml.training.train_model --quick    # essai rapide
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.models.autoencoder_branch import train_autoencoder_branch
from ml.models.baselines import (isolation_score, rule_based_score, train_isolation_forest,
                                 train_logistic_regression, train_random_forest)
from ml.models.hybrid_ensemble import HybridEnsemble
from ml.models.lstm_attention_branch import train_lstm_branch
from ml.models.meta_learner import BRANCH_NAMES, build_meta_learner, meta_coefficients
from ml.models.xgboost_branch import train_xgboost_branch
from ml.preprocessing.train_test_split_normalize import ARTIFACTS_DIR, load_processed
from ml.training.evaluation import (best_f1_threshold, evaluate, evaluate_by_group,
                                    recall_by_fraud_type)

REPORTS_DIR = Path("ml/reports")


@dataclass
class TrainConfig:
    seed: int = 42
    xgb_estimators: int = 600
    lstm_epochs: int = 8
    lstm_neg_sample_rate: float = 0.3
    ae_epochs: int = 30
    skip_baselines: bool = False
    use_mlflow: bool = True


def log(msg: str) -> None:
    print(msg, flush=True)


def _score_block(name, y_val, s_val, y_test, s_test, test_meta) -> dict:
    thr = best_f1_threshold(y_val, s_val)
    return {
        "model": name,
        "threshold": thr,
        "test": evaluate(y_test, s_test, thr),
        "test_by_channel": evaluate_by_group(y_test, s_test, thr, test_meta["channel"].to_numpy()),
        "test_recall_by_fraud_type": recall_by_fraud_type(y_test, s_test, thr,
                                                          test_meta["fraud_type"].to_numpy()),
    }


def train_pipeline(cfg: TrainConfig) -> dict:
    t_start = time.perf_counter()
    np.random.seed(cfg.seed)
    data = load_processed()
    X, y, seq_idx, meta = data.X, data.y, data.seq_idx, data.meta
    fit, es, val, test = data.rows("train"), data.rows("early_stop"), data.rows("val"), data.rows("test")
    train_all = data.rows("train", "early_stop")
    log(f"Données : fit={len(fit):,}  early_stop={len(es):,}  val={len(val):,}  test={len(test):,}  "
        f"variables={X.shape[1]}")
    y_val, y_test = y[val], y[test]
    test_meta = meta.iloc[test].reset_index(drop=True)
    results: dict[str, dict] = {}
    test_scores: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- 1. branches
    log("\n[1/5] Branche XGBoost...")
    xgb_model = train_xgboost_branch(X[fit], y[fit], X[es], y[es], cfg.xgb_estimators, cfg.seed)
    log(f"    meilleure itération : {xgb_model.best_iteration}")

    log("\n[2/5] Branche Autoencodeur (transactions légitimes uniquement)...")
    ae_model = train_autoencoder_branch(X[fit][y[fit] == 0], X[es][y[es] == 0],
                                        epochs=cfg.ae_epochs, seed=cfg.seed, log=log)

    log("\n[3/5] Branche LSTM + Attention (séquences par utilisateur)...")
    lstm_model = train_lstm_branch(X, y, seq_idx, fit, es, epochs=cfg.lstm_epochs,
                                   neg_sample_rate=cfg.lstm_neg_sample_rate, seed=cfg.seed, log=log)

    # ---------------------------------------------------------------- 2-3. meta-learner + seuil
    log("\n[4/5] Meta-learner (stacking sur la validation) et seuil de décision...")
    ensemble = HybridEnsemble(xgb_model, lstm_model, ae_model)
    S_val = ensemble.branch_scores(X, seq_idx, val)
    S_test = ensemble.branch_scores(X, seq_idx, test)
    ensemble.meta = build_meta_learner().fit(S_val, y_val)
    p_val = ensemble.meta.predict_proba(S_val)[:, 1]
    p_test = ensemble.meta.predict_proba(S_test)[:, 1]
    results["hybrid"] = _score_block("Hybride (XGB + LSTM + AE)", y_val, p_val, y_test, p_test, test_meta)
    ensemble.threshold = results["hybrid"]["threshold"]
    test_scores["hybrid"] = p_test
    log(f"    coefficients : {meta_coefficients(ensemble.meta)}  seuil={ensemble.threshold:.4f}")

    for i, name in enumerate(BRANCH_NAMES):
        results[f"branch_{name}"] = _score_block(f"Branche seule : {name}", y_val, S_val[:, i],
                                                 y_test, S_test[:, i], test_meta)
        test_scores[f"branch_{name}"] = S_test[:, i]

    # ---------------------------------------------------------------- 5. ablations
    for keep in combinations(range(3), 2):
        dropped = [BRANCH_NAMES[i] for i in range(3) if i not in keep][0]
        m = build_meta_learner().fit(S_val[:, keep], y_val)
        results[f"ablation_without_{dropped}"] = _score_block(
            f"Hybride sans {dropped}", y_val, m.predict_proba(S_val[:, keep])[:, 1],
            y_test, m.predict_proba(S_test[:, keep])[:, 1], test_meta)

    # ---------------------------------------------------------------- 4. baselines
    if not cfg.skip_baselines:
        log("\n[5/5] Baselines...")
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
        raw = lambda rows: scaler.inverse_transform(X[rows])  # noqa: E731
        results["baseline_rules"] = _score_block("Règles métier", y_val, rule_based_score(raw(val)),
                                                 y_test, rule_based_score(raw(test)), test_meta)
        log("    régression logistique...")
        lr = train_logistic_regression(X[train_all], y[train_all], cfg.seed)
        results["baseline_logreg"] = _score_block("Régression logistique", y_val, lr.decision_function(X[val]),
                                                  y_test, lr.decision_function(X[test]), test_meta)
        log("    random forest...")
        rf = train_random_forest(X[train_all], y[train_all], cfg.seed)
        results["baseline_random_forest"] = _score_block("Random Forest", y_val, rf.predict_proba(X[val])[:, 1],
                                                         y_test, rf.predict_proba(X[test])[:, 1], test_meta)
        log("    isolation forest...")
        iso = train_isolation_forest(X[train_all][y[train_all] == 0], cfg.seed)
        results["baseline_isolation_forest"] = _score_block(
            "Isolation Forest", y_val, isolation_score(iso, X[val]), y_test, isolation_score(iso, X[test]),
            test_meta)
    else:
        log("\n[5/5] Baselines ignorées (--skip-baselines)")

    # ---------------------------------------------------------------- sauvegarde
    elapsed = time.perf_counter() - t_start
    comparison = pd.DataFrame([{"key": k, "model": r["model"], "threshold": round(r["threshold"], 5),
                                **{m: r["test"].get(m) for m in ["pr_auc", "roc_auc", "recall_at_1pct_fpr",
                                                                 "precision", "recall", "f1",
                                                                 "false_positive_rate"]},
                                "pr_auc_mobile_money": r["test_by_channel"].get("MOBILE_MONEY", {}).get("pr_auc"),
                                "pr_auc_visa_virtual": r["test_by_channel"].get("VISA_VIRTUAL", {}).get("pr_auc")}
                               for k, r in results.items()]).sort_values("pr_auc", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORTS_DIR / "phase3_model_comparison.csv", index=False)
    (REPORTS_DIR / "phase3_results.json").write_text(json.dumps(
        {"config": asdict(cfg), "training_seconds": round(elapsed), "results": results},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    preds = test_meta.copy()
    for k, s in test_scores.items():
        preds[f"score_{k}"] = s
    preds["pred_hybrid"] = (test_scores["hybrid"] >= ensemble.threshold).astype(int)
    preds.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    ensemble.save(ARTIFACTS_DIR, metadata={
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "train_config": asdict(cfg),
        "xgb_best_iteration": int(xgb_model.best_iteration),
        "test_metrics": results["hybrid"]["test"],
        "test_metrics_by_channel": results["hybrid"]["test_by_channel"],
    })

    if cfg.use_mlflow:
        _log_mlflow(cfg, results, ensemble)

    _print_summary(comparison, results, ensemble, elapsed)
    return results


def _log_mlflow(cfg: TrainConfig, results: dict, ensemble: HybridEnsemble) -> None:
    try:
        import mlflow
        from ml.registry.mlflow_config import init_mlflow
    except ImportError:
        log("\n(MLflow non installé : suivi d'expérience ignoré)")
        return
    init_mlflow()
    with mlflow.start_run(run_name="hybrid_ensemble"):
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("threshold", ensemble.threshold)
        for k, r in results.items():
            for m in ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "f1"]:
                if m in r["test"]:
                    mlflow.log_metric(f"{k}.{m}", r["test"][m])
        mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="artifacts")
        mlflow.log_artifacts(str(REPORTS_DIR), artifact_path="reports")
    log("\nRun enregistré dans MLflow")


def _print_summary(comparison: pd.DataFrame, results: dict, ensemble: HybridEnsemble, elapsed: float) -> None:
    pd.set_option("display.width", 200)
    log("\n" + "=" * 100)
    log("RÉSULTATS SUR LE JEU DE TEST (période la plus récente, jamais vue)")
    log("=" * 100)
    cols = ["model", "pr_auc", "roc_auc", "recall_at_1pct_fpr", "precision", "recall", "f1",
            "pr_auc_mobile_money", "pr_auc_visa_virtual"]
    log(comparison[cols].to_string(index=False))
    log("\nModèle hybride : détection par typologie de fraude")
    for t, r in sorted(results["hybrid"]["test_recall_by_fraud_type"].items(), key=lambda kv: -kv[1]["recall"]):
        log(f"    {t:<20} rappel={r['recall']:.3f}  (n={r['n']})")
    log(f"\nCoefficients du meta-learner : {meta_coefficients(ensemble.meta)}")
    log(f"Seuil (choisi sur la validation) : {ensemble.threshold:.4f}")
    log(f"Durée totale : {elapsed / 60:.1f} min")
    log(f"[OK] Modèles -> {ARTIFACTS_DIR}/   Rapports -> {REPORTS_DIR}/")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Phase 3 : entraînement du modèle hybride et des baselines")
    p.add_argument("--quick", action="store_true", help="essai rapide (peu d'époques)")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    cfg = TrainConfig(seed=a.seed, skip_baselines=a.skip_baselines, use_mlflow=not a.no_mlflow)
    if a.quick:
        cfg.xgb_estimators, cfg.lstm_epochs, cfg.ae_epochs = 150, 2, 5
    return cfg


if __name__ == "__main__":
    train_pipeline(parse_args())
