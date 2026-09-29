"""
Met en production l'architecture retenue par la sélection de modèle, sans réentraîner les
branches déjà évaluées :

    - branche arbres : forêt aléatoire de la sélection (ml/reports/selection_candidats/rf_branch.pkl)
    - LSTM (et autoencodeur si retenu) : ceux du modèle en production
    - méta-apprenant et seuil : ajustés sur la validation, comme à l'entraînement

L'ancien modèle est archivé (ml/artifacts/archive/<version>/) ; metadata.json est écrit en
dernier : le scoring-service recharge alors le nouveau modèle à chaud.

    python -m ml.training.adopt_model --branches random_forest lstm_attention autoencoder
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


def main(branches: list[str]) -> dict:
    current = HybridEnsemble.load(ARTIFACTS_DIR)
    old_meta = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    tree_kind = branches[0]
    tree = joblib.load(CANDIDATES_DIR / TREE_FILES[tree_kind]) if tree_kind != current.tree_kind else current.tree
    ae = current.ae if "autoencoder" in branches else None
    if "autoencoder" in branches and ae is None:
        raise SystemExit("le modèle actuel n'a pas d'autoencodeur à reprendre : lancer ml.training.train_model")
    new = HybridEnsemble(tree, current.lstm, ae, tree_kind=tree_kind)

    data = load_processed()
    X, y, seq_idx, meta = data.X, data.y, data.seq_idx, data.meta
    val, test = data.rows("val"), data.rows("test")
    S_val, S_test = new.branch_scores(X, seq_idx, val), new.branch_scores(X, seq_idx, test)
    new.meta = build_meta_learner().fit(S_val, y[val])
    p_val, p_test = new.meta.predict_proba(S_val)[:, 1], new.meta.predict_proba(S_test)[:, 1]
    new.threshold = best_f1_threshold(y[val], p_val)
    ch = meta["channel"].to_numpy()[test]
    truth = meta["is_fraud_true"].to_numpy().astype(int)[test]

    archive = ARTIFACTS_DIR / "archive" / str(old_meta["trained_at"]).replace(":", "-")
    archive.mkdir(parents=True, exist_ok=True)
    for f in ARTIFACTS_DIR.iterdir():
        if f.is_file():
            shutil.copy2(f, archive / f.name)

    version = pd.Timestamp.now().isoformat(timespec="seconds")
    new.save(ARTIFACTS_DIR, metadata={
        "trained_at": version,
        "previous_version": old_meta.get("trained_at"),
        "selection": ["ml/reports/selection_modele.json", "ml/reports/selection_foret.json"],
        "test_metrics": evaluate(y[test], p_test, new.threshold),
        "test_metrics_by_channel": evaluate_by_group(y[test], p_test, new.threshold, ch),
        "test_metrics_truth": evaluate(truth, p_test, new.threshold),
        "test_metrics_truth_by_channel": evaluate_by_group(truth, p_test, new.threshold, ch),
        "train_config": old_meta.get("train_config"),
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
    return {"version": version, "branches": new.branches, "threshold": new.threshold,
            "p_test_mean": float(np.mean(p_test))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branches", nargs="+", required=True,
                    help="ex. random_forest lstm_attention autoencoder (la branche arbres en premier)")
    main(ap.parse_args().branches)
