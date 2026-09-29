"""
Deuxième manche de la sélection de modèle (suite de ml/training/select_model.py).

La première manche a montré :
  - XGBoost + LSTM sans autoencodeur (choix de la règle) dégrade la carte Visa de façon
    significative sur le test : besoin B2 non respecté, candidat rejeté ;
  - forêt aléatoire + LSTM est le seul candidat meilleur partout (global, Mobile Money, Visa),
    mais écarté pour lenteur (~40 ms) et faute d'explication par variable ;
  - l'autoencodeur apporte un petit gain constant sur les typologies jamais vues.

Cette manche : forêt aplatie (ml/models/flat_forest.py) pour lever B3 et B4, puis
forêt + LSTM avec et sans autoencodeur, et test de fraude inconnue avec la forêt.

    python -m ml.training.select_forest [--skip-loto]
Sorties : ml/reports/selection_foret.json, ml/reports/selection_candidats/rf_branch.pkl
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from ml.models.baselines import train_random_forest
from ml.models.flat_forest import FlatForest
from ml.models.hybrid_ensemble import HybridEnsemble, rf_logit
from ml.models.meta_learner import build_meta_learner
from ml.preprocessing.train_test_split_normalize import ARTIFACTS_DIR, load_processed
from ml.training.select_model import (CANDIDATES_DIR, EXPLAIN_MS, LATENCY_P99_MS, REPORTS, log, metrics,
                                      paired_bootstrap, stack_scores, time_ms)

TYPOLOGIES = ["SIM_SWAP", "ACCOUNT_TAKEOVER", "SOCIAL_ENGINEERING", "AGENT_FRAUD", "SMURFING",
              "CARD_TESTING", "CNP_FRAUD", "TOPUP_DRAIN"]


def main(skip_loto: bool = False, n_boot: int = 1000, seed: int = 42) -> dict:
    t0 = time.perf_counter()
    data = load_processed()
    X, y, seq_idx, meta = data.X, data.y, data.seq_idx, data.meta
    fit, es, val, test = data.rows("train"), data.rows("early_stop"), data.rows("val"), data.rows("test")
    train_all = data.rows("train", "early_stop")
    y_val, y_test = y[val], y[test]
    truth = meta["is_fraud_true"].to_numpy().astype(int)
    t_val, t_test = truth[val], truth[test]
    ch_val, ch_test = meta["channel"].to_numpy()[val], meta["channel"].to_numpy()[test]
    users_test = meta["user_id"].to_numpy()[test]
    typ_true = meta["fraud_type_true"].fillna("").to_numpy()
    ftype_obs = meta["fraud_type"].fillna("").to_numpy()

    ens = HybridEnsemble.load(ARTIFACTS_DIR)
    S_val, S_test = ens.branch_scores(X, seq_idx, val), ens.branch_scores(X, seq_idx, test)
    base = {"xgb": (S_val[:, 0], S_test[:, 0]), "lstm": (S_val[:, 1], S_test[:, 1]), "ae": (S_val[:, 2], S_test[:, 2])}
    log("Forêt aléatoire (train + early_stop)...")
    rf = train_random_forest(X[train_all], y[train_all], seed)
    base["rf"] = (rf_logit(rf, X[val]), rf_logit(rf, X[test]))
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, CANDIDATES_DIR / "rf_branch.pkl", compress=3)

    # ------------------------------------------------------------------ forêt aplatie : B3, B4
    flat = FlatForest(rf)
    sample = test[:2000]
    diff = np.abs(np.array([flat.predict_proba_one(x) for x in X[sample]]) - rf.predict_proba(X[sample])[:, 1])
    torch.set_num_threads(1)
    x1 = X[test[:1]][0]
    lat_predict = time_ms(lambda: flat.predict_proba_one(x1), n=500)
    lat_explain = time_ms(lambda: flat.explain_one(x1), n=500)
    torch.set_num_threads(4)
    lstm_p99, ae_p99 = 5.246, 0.716      # mesurés à la première manche (selection_modele.json)
    flat_info = {"n_trees": flat.n_trees, "n_nodes": flat.n_nodes, "depth": flat.depth,
                 "memory_mb": round(sum(a.nbytes for a in (flat.feature, flat.threshold, flat.left, flat.right,
                                                            flat.prob, flat.is_leaf)) / 2**20, 1),
                 "max_abs_diff_vs_sklearn": float(diff.max()),
                 "predict_ms": lat_predict, "explain_ms": lat_explain}
    log(f"Forêt aplatie : {json.dumps(flat_info, ensure_ascii=False)}")

    # ------------------------------------------------------------------ candidats
    specs = {"Hybride actuel (XGB + LSTM + AE)": ["xgb", "lstm", "ae"],
             "Forêt aléatoire + LSTM": ["rf", "lstm"],
             "Forêt aléatoire + LSTM + AE": ["rf", "lstm", "ae"]}
    rows, metas = {}, {}
    for label, comps in specs.items():
        Sv = np.column_stack([base[c][0] for c in comps])
        St = np.column_stack([base[c][1] for c in comps])
        sv, st, m = stack_scores(Sv, y_val, St)
        metas[label] = m
        p99 = (lat_explain["p99"] if "rf" in comps else 0.897) + lstm_p99 + (ae_p99 if "ae" in comps else 0.0)
        rows[label] = {"components": comps, "val_observed": metrics(y_val, sv, ch_val), "val_truth": metrics(t_val, sv),
                       "test_observed": metrics(y_test, st, ch_test), "test_truth": metrics(t_test, st, ch_test),
                       "latency_p99_ms_with_explanation": round(p99, 2),
                       "meets_realtime": p99 <= LATENCY_P99_MS, "_st": st}
        r = rows[label]
        log(f"{label:<32} val {r['val_observed']['pr_auc']:.4f} (vérité {r['val_truth']['pr_auc']:.4f}) | test vérité "
            f"{r['test_truth']['pr_auc']:.4f} MM {r['test_truth']['pr_auc_mobile_money']:.4f} "
            f"Visa {r['test_truth']['pr_auc_visa_virtual']:.4f} | p99 {p99:.1f} ms")

    cur = "Hybride actuel (XGB + LSTM + AE)"
    confirm = {}
    for label in specs:
        if label == cur:
            continue
        confirm[label] = {}
        for scope, mask in (("global", np.ones_like(t_test, bool)), ("mobile_money", ch_test == "MOBILE_MONEY"),
                            ("visa", ch_test == "VISA_VIRTUAL")):
            confirm[label][scope] = paired_bootstrap(t_test[mask], rows[label]["_st"][mask], rows[cur]["_st"][mask],
                                                     users_test[mask], n_boot)
        log(f"Test (vérité) {label} − hybride actuel : {confirm[label]}")
    ae_effect = {scope: paired_bootstrap(t_test[mask], rows["Forêt aléatoire + LSTM + AE"]["_st"][mask],
                                         rows["Forêt aléatoire + LSTM"]["_st"][mask], users_test[mask], n_boot)
                 for scope, mask in (("global", np.ones_like(t_test, bool)), ("visa", ch_test == "VISA_VIRTUAL"))}
    log(f"Effet de l'autoencodeur avec la forêt (test, vérité) : {ae_effect}")

    # ------------------------------------------------------------------ fraude inconnue, avec la forêt
    loto = {}
    if not skip_loto:
        legit_test = t_test == 0
        for T in TYPOLOGIES:
            keep_tr, keep_val = train_all[ftype_obs[train_all] != T], val[ftype_obs[val] != T]
            log(f"Fraude inconnue (forêt) : entraînement sans {T}...")
            m = train_random_forest(X[keep_tr], y[keep_tr], seed)
            rv, rt = rf_logit(m, X[keep_val]), rf_logit(m, X[test])
            lv, lt = base["lstm"][0][np.isin(val, keep_val)], base["lstm"][1]
            av, at = base["ae"][0][np.isin(val, keep_val)], base["ae"][1]
            legit_val = truth[keep_val] == 0
            pos = typ_true[test] == T

            def rec(sv, st):
                thr = np.quantile(sv[legit_val], 0.99)
                return round(float((st[pos] >= thr).mean()), 3)

            def stacked(cols_v, cols_t):
                mm = build_meta_learner().fit(np.column_stack(cols_v), y[keep_val])
                return mm.predict_proba(np.column_stack(cols_v))[:, 1], mm.predict_proba(np.column_stack(cols_t))[:, 1]

            loto[T] = {"n_test": int(pos.sum()), "foret_sans_T": rec(rv, rt),
                       "foret_sans_T_plus_ae": rec(*stacked([rv, av], [rt, at])),
                       "ae_seul": rec(av, at)}
            log(f"    {T}: {loto[T]}  (le LSTM a vu {T} : exclu de ce test)")

    report = {"foret_aplatie": flat_info,
              "candidats": {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in rows.items()},
              "confirmation_test": confirm, "effet_autoencodeur_avec_foret": ae_effect, "fraude_inconnue_foret": loto,
              "duree_min": round((time.perf_counter() - t0) / 60, 1)}
    for label, m in metas.items():
        joblib.dump(m, CANDIDATES_DIR / f"meta_{'_'.join(specs[label])}.pkl")
    (REPORTS / "selection_foret.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
    log(f"\nRapport : {REPORTS / 'selection_foret.json'}  ({report['duree_min']} min)")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-loto", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.skip_loto, a.n_boot)
