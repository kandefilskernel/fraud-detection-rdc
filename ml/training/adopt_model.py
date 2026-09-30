"""
Met en production l'architecture retenue par la sélection de modèle, sans réentraîner les
branches déjà évaluées :

    - branche arbres : forêt aléatoire de la sélection (ml/reports/selection_candidats/rf_branch.pkl)
    - LSTM (et autoencodeur si retenu) : ceux du modèle en production ; le LSTM est facultatif
    - méta-apprenant et seuil : ajustés sur la validation, comme à l'entraînement
    - --anomaly-detector : l'autoencodeur actuel devient une « veille des anomalies » HORS
      méta-apprenant (seuil = quantile des transactions de validation non signalées)

L'ancien modèle est archivé (ml/artifacts/archive/<version>/) ; metadata.json est écrit en
dernier : le scoring-service recharge alors le nouveau modèle à chaud.

    python -m ml.training.adopt_model --branches random_forest lstm_attention autoencoder
    python -m ml.training.adopt_model --branches random_forest lstm_attention --anomaly-detector
    python -m ml.training.adopt_model --branches random_forest --anomaly-detector

La commande à utiliser est proposée par ml.training.select_final (ml/reports/selection_finale.md).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.models.hybrid_ensemble import TREE_FILES, HybridEnsemble
from ml.models.meta_learner import build_meta_learner
from ml.preprocessing.train_test_split_normalize import ARTIFACTS_DIR, load_processed
from ml.training.evaluation import best_f1_threshold, evaluate, evaluate_by_group
from ml.training.select_model import CANDIDATES_DIR


ANOMALY_QUANTILE = 0.995


def main(branches: list[str], anomaly_detector: bool = False) -> dict:
    current = HybridEnsemble.load(ARTIFACTS_DIR)
    old_meta = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    tree_kind = branches[0]
    if tree_kind not in TREE_FILES:
        raise SystemExit(f"la première branche doit être une branche arbres ({list(TREE_FILES)}), pas {tree_kind}")
    unknown = set(branches[1:]) - {"lstm_attention", "autoencoder"}
    if unknown:
        raise SystemExit(f"branches inconnues : {unknown}")
    if tree_kind == current.tree_kind:
        tree = current.tree
    elif (ARTIFACTS_DIR / TREE_FILES[tree_kind]).exists() and not (CANDIDATES_DIR / TREE_FILES[tree_kind]).exists():
        tree = joblib.load(ARTIFACTS_DIR / TREE_FILES[tree_kind])
    else:
        tree = joblib.load(CANDIDATES_DIR / TREE_FILES[tree_kind])
    existing_ae = current.ae if current.ae is not None else current.anomaly
    if "lstm_attention" in branches and current.lstm is None:
        raise SystemExit("le modèle actuel n'a pas de LSTM à reprendre : lancer ml.training.train_model")
    if ("autoencoder" in branches or anomaly_detector) and existing_ae is None:
        raise SystemExit("le modèle actuel n'a pas d'autoencodeur à reprendre : lancer ml.training.train_model")
    if "autoencoder" in branches and anomaly_detector:
        raise SystemExit("autoencodeur : soit branche du méta-apprenant, soit veille séparée, pas les deux")
    lstm = current.lstm if "lstm_attention" in branches else None
    ae = existing_ae if "autoencoder" in branches else None
    new = HybridEnsemble(tree, lstm, ae, tree_kind=tree_kind)

    data = load_processed()
    X, y, seq_idx, meta = data.X, data.y, data.seq_idx, data.meta
    val, test = data.rows("val"), data.rows("test")
    S_val, S_test = new.branch_scores(X, seq_idx, val), new.branch_scores(X, seq_idx, test)
    new.meta = build_meta_learner().fit(S_val, y[val])
    p_val, p_test = new.meta.predict_proba(S_val)[:, 1], new.meta.predict_proba(S_test)[:, 1]
    new.threshold = best_f1_threshold(y[val], p_val)
    ch = meta["channel"].to_numpy()[test]
    truth = meta["is_fraud_true"].to_numpy().astype(int)[test]

    anomaly_report = None
    if anomaly_detector:
        new.anomaly = existing_ae
        a_val = new.anomaly_scores(X, val)
        # seuil calé sur ce que l'opérateur connaît : transactions de validation NON signalées
        new.anomaly_threshold = float(np.quantile(a_val[y[val] == 0], ANOMALY_QUANTILE))
        flag = new.anomaly_scores(X, test) >= new.anomaly_threshold
        alert = p_test >= new.threshold
        new.anomaly_info = {"quantile": ANOMALY_QUANTILE, "calibrated_on": "validation, transactions non signalées",
                            "effect_on_decision": "aucun (signal pour les analystes)"}
        anomaly_report = {"legit_flag_rate_truth": round(float(flag[truth == 0].mean()), 5),
                          "fraud_flag_rate_truth": round(float(flag[truth == 1].mean()), 4),
                          "frauds_missed_by_model": int(((truth == 1) & ~alert).sum()),
                          "of_which_flagged": int(((truth == 1) & ~alert & flag).sum())}

    archive = ARTIFACTS_DIR / "archive" / str(old_meta["trained_at"]).replace(":", "-")
    archive.mkdir(parents=True, exist_ok=True)
    for f in ARTIFACTS_DIR.iterdir():
        if f.is_file():
            shutil.copy2(f, archive / f.name)

    version = pd.Timestamp.now().isoformat(timespec="seconds")
    new.save(ARTIFACTS_DIR, metadata={
        "trained_at": version,
        "previous_version": old_meta.get("trained_at"),
        "selection": ["ml/reports/selection_finale.json (décision, validation seule)",
                      "ml/reports/selection_modele.json (exploratoire)", "ml/reports/selection_foret.json (exploratoire)"],
        "test_metrics": evaluate(y[test], p_test, new.threshold),
        "test_metrics_by_channel": evaluate_by_group(y[test], p_test, new.threshold, ch),
        "test_metrics_truth": evaluate(truth, p_test, new.threshold),
        "test_metrics_truth_by_channel": evaluate_by_group(truth, p_test, new.threshold, ch),
        "train_config": old_meta.get("train_config"),
        "adopted_branches": new.branches,
        **({"anomaly_detector_test": anomaly_report} if anomaly_report else {}),
    })
    # scores hors ligne du modèle en production : référence du contrôle de parité (ml.serving.parity_check)
    preds_path = Path("ml/reports/test_predictions.csv")
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        prod = pd.Series(p_test, index=meta["transaction_id"].to_numpy()[test])
        preds["score_production"] = preds["transaction_id"].map(prod)
        preds["pred_production"] = (preds["score_production"] >= new.threshold).astype(int)
        preds.to_csv(preds_path, index=False)
    print(f"[OK] modèle {version} : {' + '.join(new.branches)}  seuil={new.threshold:.4f}  "
          f"(ancien archivé dans {archive})")
    if anomaly_report:
        print(f"     veille des anomalies : seuil={new.anomaly_threshold:.4f}  {anomaly_report}")
    return {"version": version, "branches": new.branches, "threshold": new.threshold,
            "p_test_mean": float(np.mean(p_test))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branches", nargs="+", required=True,
                    help="ex. random_forest lstm_attention autoencoder (la branche arbres en premier)")
    ap.add_argument("--anomaly-detector", action="store_true",
                    help="garder l'autoencodeur comme veille des anomalies, hors méta-apprenant")
    a = ap.parse_args()
    main(a.branches, a.anomaly_detector)
