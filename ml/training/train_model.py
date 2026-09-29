"""
Phase 3 : entraînement et comparaison des modèles.

Étapes :
    1. Branches du modèle hybride, entraînées sur « train » (arrêt précoce sur « early_stop »)
    2. Meta-learner entraîné sur « val » (scores hors échantillon des branches) : hybride
       complet (comparaison) et architecture de PRODUCTION (branches choisies selon les
       besoins, docs/SELECTION_MODELE.md : forêt aléatoire + LSTM + autoencodeur)
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

from ml.features.feature_engineering import FEATURE_GROUPS, FEATURE_NAMES
from ml.models.autoencoder_branch import train_autoencoder_branch
from ml.models.baselines import (isolation_score, rule_based_score, train_isolation_forest,
                                 train_logistic_regression, train_random_forest)
from ml.models.hybrid_ensemble import HybridEnsemble, rf_logit
from ml.models.lstm_attention_branch import train_lstm_branch
from ml.models.meta_learner import BRANCH_NAMES, build_meta_learner, meta_coefficients
from ml.models.xgboost_branch import train_xgboost_branch
from ml.preprocessing.train_test_split_normalize import ARTIFACTS_DIR, PROCESSED_DIR, load_processed
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
    # Ablation par famille de variables (ex. ("recipient_network",) = modèle SANS C1) :
    # mêmes données, même split, mêmes graines -> comparaison contrôlée
    exclude_groups: tuple = ()
    reports_dir: str = "ml/reports"
    save_artifacts: bool = True
    # Expérience C2 : données prétraitées ailleurs (ex. profils « silo »), avec leur scaler
    processed_dir: str = str(PROCESSED_DIR)
    # Branches du modèle mis en production, choisies selon les besoins (docs/SELECTION_MODELE.md) :
    # forêt aléatoire (meilleure sur la vérité terrain et sur Visa, rendue rapide par FlatForest),
    # LSTM (séquence), autoencodeur (petit gain constant sur les typologies jamais vues)
    production_branches: tuple = ("random_forest", "lstm_attention", "autoencoder")


def log(msg: str) -> None:
    print(msg, flush=True)


def _score_block(name, y_val, s_val, y_test, s_test, test_meta) -> dict:
    thr = best_f1_threshold(y_val, s_val)
    block = {
        "model": name,
        "threshold": thr,
        "test": evaluate(y_test, s_test, thr),
        "test_by_channel": evaluate_by_group(y_test, s_test, thr, test_meta["channel"].to_numpy()),
        "test_recall_by_fraud_type": recall_by_fraud_type(y_test, s_test, thr,
                                                          test_meta["fraud_type"].to_numpy()),
    }
    # Vérité terrain (fraudes jamais signalées incluses) : ce que le modèle détecte VRAIMENT.
    # Évaluation seulement : le modèle n'a été entraîné que sur ce que l'opérateur sait.
    if "is_fraud_true" in test_meta and (test_meta["is_fraud_true"] != test_meta["is_fraud"]).any():
        y_true = test_meta["is_fraud_true"].to_numpy()
        block["test_truth"] = evaluate(y_true, s_test, thr)
        block["test_truth_by_channel"] = evaluate_by_group(y_true, s_test, thr, test_meta["channel"].to_numpy())
        block["test_truth_recall_by_fraud_type"] = recall_by_fraud_type(y_true, s_test, thr,
                                                                        test_meta["fraud_type_true"].to_numpy())
    return block


def train_pipeline(cfg: TrainConfig) -> dict:
    t_start = time.perf_counter()
    np.random.seed(cfg.seed)
    reports_dir = Path(cfg.reports_dir)
    unknown = set(cfg.exclude_groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"familles inconnues : {unknown} (disponibles : {list(FEATURE_GROUPS)})")
    processed_dir = Path(cfg.processed_dir)
    experiment = processed_dir.absolute() != PROCESSED_DIR.absolute()
    if (cfg.exclude_groups or experiment) and cfg.save_artifacts:
        log("(expérience : les modèles ne sont PAS sauvegardés, le service de scoring attend le "
            "profil unifié avec toutes les variables)")
        cfg.save_artifacts = False
    scaler_path = processed_dir / "scaler.pkl" if experiment else ARTIFACTS_DIR / "scaler.pkl"
    data = load_processed(processed_dir)
    X_full, y, seq_idx, meta = data.X, data.y, data.seq_idx, data.meta
    excluded = {f for g in cfg.exclude_groups for f in FEATURE_GROUPS[g]}
    used_features = [f for f in FEATURE_NAMES if f not in excluded]
    X = (np.ascontiguousarray(X_full[:, [FEATURE_NAMES.index(f) for f in used_features]])
         if excluded else X_full)
    fit, es, val, test = data.rows("train"), data.rows("early_stop"), data.rows("val"), data.rows("test")
    train_all = data.rows("train", "early_stop")
    log(f"Données : fit={len(fit):,}  early_stop={len(es):,}  val={len(val):,}  test={len(test):,}  "
        f"variables={X.shape[1]}" + (f"  (sans : {', '.join(cfg.exclude_groups)})" if excluded else ""))
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

    # forêt aléatoire : entraînée sur train + early_stop, sans arrêt précoce sur les étiquettes
    # connues de l'opérateur (branche de production possible, et modèle de référence)
    rf_model = None
    if "random_forest" in cfg.production_branches or not cfg.skip_baselines:
        log("    forêt aléatoire...")
        rf_model = train_random_forest(X[train_all], y[train_all], cfg.seed)
    cols_val = {n: S_val[:, i] for i, n in enumerate(BRANCH_NAMES)}
    cols_test = {n: S_test[:, i] for i, n in enumerate(BRANCH_NAMES)}
    if rf_model is not None:
        cols_val["random_forest"], cols_test["random_forest"] = rf_logit(rf_model, X[val]), rf_logit(rf_model, X[test])
    P_val = np.column_stack([cols_val[b] for b in cfg.production_branches])
    P_test = np.column_stack([cols_test[b] for b in cfg.production_branches])
    prod_meta = build_meta_learner().fit(P_val, y_val)
    p_val_prod = prod_meta.predict_proba(P_val)[:, 1]
    p_test_prod = prod_meta.predict_proba(P_test)[:, 1]
    results["production"] = _score_block(f"Production ({' + '.join(cfg.production_branches)})",
                                         y_val, p_val_prod, y_test, p_test_prod, test_meta)
    tree_kind = cfg.production_branches[0]
    production = HybridEnsemble(rf_model if tree_kind == "random_forest" else xgb_model, lstm_model,
                                ae_model if "autoencoder" in cfg.production_branches else None,
                                meta=prod_meta, threshold=results["production"]["threshold"], tree_kind=tree_kind)
    test_scores["production"] = p_test_prod
    log(f"    production : {meta_coefficients(prod_meta, production.branches)}  seuil={production.threshold:.4f}")

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
        scaler = joblib.load(scaler_path)
        raw = lambda rows: scaler.inverse_transform(X_full[rows])  # noqa: E731
        results["baseline_rules"] = _score_block("Règles métier", y_val, rule_based_score(raw(val)),
                                                 y_test, rule_based_score(raw(test)), test_meta)
        log("    régression logistique...")
        lr = train_logistic_regression(X[train_all], y[train_all], cfg.seed)
        results["baseline_logreg"] = _score_block("Régression logistique", y_val, lr.decision_function(X[val]),
                                                  y_test, lr.decision_function(X[test]), test_meta)
        rf = rf_model                      # déjà entraînée plus haut (mêmes données, même graine)
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
                                "pr_auc_visa_virtual": r["test_by_channel"].get("VISA_VIRTUAL", {}).get("pr_auc"),
                                "pr_auc_truth": r.get("test_truth", {}).get("pr_auc"),
                                "recall_truth": r.get("test_truth", {}).get("recall")}
                               for k, r in results.items()]).sort_values("pr_auc", ascending=False)

    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir / "phase3_model_comparison.csv", index=False)
    (reports_dir / "phase3_results.json").write_text(json.dumps(
        {"config": asdict(cfg), "training_seconds": round(elapsed), "features_used": used_features,
         "results": results}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    preds = test_meta.copy()
    for k, s in test_scores.items():
        preds[f"score_{k}"] = s
    preds["pred_hybrid"] = (test_scores["hybrid"] >= ensemble.threshold).astype(int)
    preds["pred_production"] = (test_scores["production"] >= production.threshold).astype(int)
    preds.to_csv(reports_dir / "test_predictions.csv", index=False)

    if cfg.save_artifacts:
        if production.tree_kind != "xgboost":
            # la branche XGBoost reste utile hors production : archive de l'assistant d'enquête
            joblib.dump(xgb_model, ARTIFACTS_DIR / "xgb_branch.pkl")
        production.save(ARTIFACTS_DIR, metadata={
            "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "train_config": asdict(cfg),
            "xgb_best_iteration": int(xgb_model.best_iteration),
            "test_metrics": results["production"]["test"],
            "test_metrics_by_channel": results["production"]["test_by_channel"],
            "test_metrics_truth": results["production"].get("test_truth"),
            "selection": "ml/reports/selection_modele.json",
        })

    if cfg.use_mlflow:
        _log_mlflow(cfg, results, ensemble, reports_dir)

    _print_summary(comparison, results, ensemble, elapsed, reports_dir, cfg.save_artifacts)
    return results


def _log_mlflow(cfg: TrainConfig, results: dict, ensemble: HybridEnsemble, reports_dir: Path) -> None:
    try:
        import mlflow
        from ml.registry.mlflow_config import init_mlflow
    except ImportError:
        log("\n(MLflow non installé : suivi d'expérience ignoré)")
        return
    init_mlflow()
    run_name = "hybrid_ensemble" + "".join(f"_sans_{g}" for g in cfg.exclude_groups)
    if Path(cfg.processed_dir).absolute() != PROCESSED_DIR.absolute():
        run_name += f"_{Path(cfg.processed_dir).name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("threshold", ensemble.threshold)
        for k, r in results.items():
            for m in ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "f1"]:
                if m in r["test"]:
                    mlflow.log_metric(f"{k}.{m}", r["test"][m])
        if cfg.save_artifacts:
            mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="artifacts")
        mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
    log("\nRun enregistré dans MLflow")


def _print_summary(comparison: pd.DataFrame, results: dict, ensemble: HybridEnsemble, elapsed: float,
                   reports_dir: Path, saved: bool) -> None:
    pd.set_option("display.width", 200)
    log("\n" + "=" * 100)
    log("RÉSULTATS SUR LE JEU DE TEST (période la plus récente, jamais vue)")
    log("=" * 100)
    cols = ["model", "pr_auc", "roc_auc", "recall_at_1pct_fpr", "precision", "recall", "f1",
            "pr_auc_mobile_money", "pr_auc_visa_virtual"]
    if comparison["pr_auc_truth"].notna().any():
        cols += ["pr_auc_truth", "recall_truth"]
    log(comparison[cols].to_string(index=False))
    if "test_truth" in results["hybrid"]:
        log("  (pr_auc / recall : étiquettes connues de l'opérateur ; *_truth : vérité terrain, fraudes non "
            "signalées incluses)")
    log("\nModèle hybride : détection par typologie de fraude")
    by_type = results["hybrid"].get("test_truth_recall_by_fraud_type") or results["hybrid"]["test_recall_by_fraud_type"]
    for t, r in sorted(by_type.items(), key=lambda kv: -kv[1]["recall"]):
        log(f"    {t:<20} rappel={r['recall']:.3f}  (n={r['n']})")
    log(f"\nCoefficients du meta-learner : {meta_coefficients(ensemble.meta)}")
    log(f"Seuil (choisi sur la validation) : {ensemble.threshold:.4f}")
    log(f"Durée totale : {elapsed / 60:.1f} min")
    log(f"[OK] Modèles -> {ARTIFACTS_DIR}/" if saved else "[OK] Modèles non sauvegardés (ablation)")
    log(f"     Rapports -> {reports_dir}/")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Phase 3 : entraînement du modèle hybride et des baselines")
    p.add_argument("--quick", action="store_true", help="essai rapide (peu d'époques)")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-groups", nargs="*", default=[],
                   help=f"familles de variables retirées (ablation) parmi : {', '.join(FEATURE_GROUPS)}")
    p.add_argument("--reports-dir", default="ml/reports")
    p.add_argument("--no-save", action="store_true", help="ne pas écraser les modèles de ml/artifacts")
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR),
                   help="données prétraitées d'une expérience (ex. profils silo, C2)")
    a = p.parse_args()
    cfg = TrainConfig(seed=a.seed, skip_baselines=a.skip_baselines, use_mlflow=not a.no_mlflow,
                      exclude_groups=tuple(a.exclude_groups), reports_dir=a.reports_dir,
                      save_artifacts=not a.no_save, processed_dir=a.processed_dir)
    if a.quick:
        cfg.xgb_estimators, cfg.lstm_epochs, cfg.ae_epochs = 150, 2, 5
    return cfg


if __name__ == "__main__":
    train_pipeline(parse_args())
