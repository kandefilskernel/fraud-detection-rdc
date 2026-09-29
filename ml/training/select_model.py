"""
Choix du modèle de production à partir de BESOINS explicites, pas d'une préférence.

Besoins d'un opérateur Mobile Money (et émetteur de cartes Visa virtuelles) :

    B1  Détection      : PR-AUC maximale, mesurée sur la VALIDATION avec les étiquettes que
                          l'opérateur connaît (choix réaliste) ; le test ne sert qu'à confirmer.
    B2  Deux canaux    : la carte Visa ne doit pas s'effondrer (PR-AUC Visa rapportée).
    B3  Temps réel     : modèle seul ≤ 20 ms au 99e centile, sur 1 cœur (budget total 100 ms).
    B4  Explication    : explication par variable de chaque alerte ≤ 15 ms (audit BCC, analystes).
    B5  Fraude inconnue: tester ce qu'apporte l'autoencodeur sur une typologie jamais vue.
    B6  Simplicité     : à performance égale (écart < 0,005), le modèle le plus simple l'emporte.

Règle de choix : parmi les candidats qui respectent B3 et B4, le meilleur sur B1 ; B6 départage.
Confirmation sur le test (vérité terrain) : bootstrap apparié par client contre le modèle actuel.

    python -m ml.training.select_model [--skip-loto] [--n-boot 1000]
Sorties : ml/reports/selection_modele.json (+ modèles candidats dans ml/reports/selection_candidats/)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import cross_val_predict

from ml.models.autoencoder_branch import ae_score
from ml.models.baselines import train_random_forest
from ml.models.hybrid_ensemble import HybridEnsemble
from ml.models.lstm_attention_branch import gather_sequences, lstm_logit
from ml.models.meta_learner import build_meta_learner
from ml.preprocessing.train_test_split_normalize import ARTIFACTS_DIR, load_processed
from ml.training.evaluation import recall_at_fpr

REPORTS = Path("ml/reports")
CANDIDATES_DIR = REPORTS / "selection_candidats"
LATENCY_P99_MS, EXPLAIN_MS, TIE = 20.0, 15.0, 0.005
XGB_CONFIGS = {
    # configuration actuelle de la branche : pondération forte des fraudes (N- / N+)
    "xgb_actuel": None,
    "xgb_sans_ponderation": dict(scale_pos_weight=1.0, max_depth=6, min_child_weight=3, learning_rate=0.05),
    "xgb_ponderation_racine": dict(scale_pos_weight="sqrt", max_depth=6, min_child_weight=3, learning_rate=0.05),
    "xgb_profond": dict(scale_pos_weight=1.0, max_depth=8, min_child_weight=5, learning_rate=0.03),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def train_xgb(X_fit, y_fit, X_es, y_es, params: dict, seed: int = 42) -> xgb.XGBClassifier:
    p = dict(params)
    if p.get("scale_pos_weight") == "sqrt":
        p["scale_pos_weight"] = float(np.sqrt((y_fit == 0).sum() / max((y_fit == 1).sum(), 1)))
    model = xgb.XGBClassifier(n_estimators=3000, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                              eval_metric="aucpr", early_stopping_rounds=150, random_state=seed, n_jobs=-1, **p)
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    return model


def xgb_margin(model, X) -> np.ndarray:
    best = getattr(model, "best_iteration", None)
    rng = (0, best + 1) if best is not None else (0, 0)
    return model.get_booster().inplace_predict(X, iteration_range=rng, predict_type="margin")


def metrics(y, s, channel=None) -> dict:
    out = {"pr_auc": round(float(average_precision_score(y, s)), 4),
           "recall_at_1pct_fpr": round(recall_at_fpr(y, s, 0.01), 4)}
    if channel is not None:
        for ch in ("MOBILE_MONEY", "VISA_VIRTUAL"):
            m = channel == ch
            out[f"pr_auc_{ch.lower()}"] = round(float(average_precision_score(y[m], s[m])), 4)
    return out


def paired_bootstrap(y, s_a, s_b, users, n_boot=1000, seed=0) -> dict:
    """Écart de PR-AUC (a − b), rééchantillonnage des CLIENTS avec remise."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(users)
    groups = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], s_a[idx]) - average_precision_score(y[idx], s_b[idx]))
    d = np.array(diffs)
    delta = average_precision_score(y, s_a) - average_precision_score(y, s_b)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": round(float(delta), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "significant": bool(lo > 0 or hi < 0)}


def time_ms(fn, n=300) -> dict:
    for _ in range(20):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return {"p50": round(float(np.percentile(t, 50)), 3), "p99": round(float(np.percentile(t, 99)), 3)}


def stack_scores(S_val, y_val, S_test) -> tuple[np.ndarray, np.ndarray, object]:
    """Méta-apprenant : scores de validation par validation croisée (évaluation honnête du
    choix), scores de test par le méta-apprenant ajusté sur toute la validation."""
    meta = build_meta_learner()
    val = cross_val_predict(meta, S_val, y_val, cv=5, method="predict_proba")[:, 1]
    meta.fit(S_val, y_val)
    return val, meta.predict_proba(S_test)[:, 1], meta


def main(skip_loto: bool = False, n_boot: int = 1000, seed: int = 42) -> dict:
    t0 = time.perf_counter()
    torch.set_num_threads(4)
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
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    ens = HybridEnsemble.load(ARTIFACTS_DIR)
    log("Scores des branches actuelles (XGBoost, LSTM, autoencodeur)...")
    S_val_cur, S_test_cur = ens.branch_scores(X, seq_idx, val), ens.branch_scores(X, seq_idx, test)
    base = {"xgb_actuel": (S_val_cur[:, 0], S_test_cur[:, 0]), "lstm": (S_val_cur[:, 1], S_test_cur[:, 1]),
            "ae": (S_val_cur[:, 2], S_test_cur[:, 2])}
    xgb_models = {"xgb_actuel": ens.xgb}

    for name, params in XGB_CONFIGS.items():
        if params is None:
            continue
        log(f"XGBoost {name}...")
        m = train_xgb(X[fit], y[fit], X[es], y[es], params, seed)
        xgb_models[name] = m
        base[name] = (xgb_margin(m, X[val]), xgb_margin(m, X[test]))
        log(f"    itérations retenues : {m.best_iteration}  PR-AUC val : "
            f"{average_precision_score(y_val, base[name][0]):.4f}")
    best_xgb = max((k for k in xgb_models), key=lambda k: average_precision_score(y_val, base[k][0]))
    log(f"Meilleure configuration XGBoost sur la validation : {best_xgb}")

    log("Forêt aléatoire (train + early_stop, comme la référence de la phase 3)...")
    rf = train_random_forest(X[train_all], y[train_all], seed)
    base["rf"] = (logit(rf.predict_proba(X[val])[:, 1]), logit(rf.predict_proba(X[test])[:, 1]))

    # ------------------------------------------------------------------ candidats
    specs = {
        "Hybride actuel (XGB + LSTM + AE)": ["xgb_actuel", "lstm", "ae"],
        "XGB actuel + LSTM": ["xgb_actuel", "lstm"],
        f"XGB optimisé + LSTM": [best_xgb, "lstm"],
        f"XGB optimisé + LSTM + AE": [best_xgb, "lstm", "ae"],
        "Forêt aléatoire + LSTM": ["rf", "lstm"],
        "Forêt aléatoire seule": ["rf"],
        "XGB optimisé seul": [best_xgb],
        "LSTM seul": ["lstm"],
    }
    comps_cost = {}
    # latence et explication, 1 cœur, 1 transaction
    x1 = X[test[:1]].astype(np.float32)
    seq, mask = gather_sequences(X, seq_idx, test[:1])
    torch.set_num_threads(1)
    for k, m in xgb_models.items():
        b = m.get_booster()
        b.set_param({"nthread": 1})
        rng_it = (0, m.best_iteration + 1)
        comps_cost[k] = {"predict": time_ms(lambda: b.inplace_predict(x1, iteration_range=rng_it, predict_type="margin")),
                         "explain": time_ms(lambda: b.predict(xgb.DMatrix(x1), pred_contribs=True,
                                                              iteration_range=rng_it), n=150)}
    rf.set_params(n_jobs=1)
    comps_cost["rf"] = {"predict": time_ms(lambda: rf.predict_proba(x1), n=150),
                        "explain": None}   # pas de TreeSHAP dans la pile (paquet shap absent, lent sur forêt profonde)
    with torch.no_grad():
        comps_cost["lstm"] = {"predict": time_ms(lambda: ens.lstm(seq, mask)), "explain": 0.0}
        comps_cost["ae"] = {"predict": time_ms(lambda: ens.ae.reconstruction_error(torch.from_numpy(x1))),
                            "explain": 0.0}
    torch.set_num_threads(4)

    rows = {}
    stacked_models = {}
    for label, comps in specs.items():
        Sv = np.column_stack([base[c][0] for c in comps])
        St = np.column_stack([base[c][1] for c in comps])
        if len(comps) > 1:
            sv, st, m = stack_scores(Sv, y_val, St)
            stacked_models[label] = m
        else:
            sv, st = Sv[:, 0], St[:, 0]
        p99 = sum(comps_cost[c]["predict"]["p99"] for c in comps)
        explain = [comps_cost[c]["explain"] for c in comps if c in comps_cost]
        tree = [c for c in comps if c.startswith(("xgb", "rf"))]
        explain_ms = None if any(comps_cost[c]["explain"] is None for c in tree) else \
            round(sum(comps_cost[c]["explain"]["p99"] for c in tree), 2) if tree else 0.0
        rows[label] = {
            "components": comps,
            "val_observed": metrics(y_val, sv, ch_val), "val_truth": metrics(t_val, sv),
            "test_observed": metrics(y_test, st, ch_test), "test_truth": metrics(t_test, st, ch_test),
            "latency_p99_ms": round(p99, 2), "explain_p99_ms": explain_ms,
            "meets_realtime": p99 <= LATENCY_P99_MS,
            "meets_explain": explain_ms is not None and explain_ms <= EXPLAIN_MS,
            "_test_scores": st, "_val_scores": sv,
        }
        r = rows[label]
        log(f"{label:<36} val PR-AUC {r['val_observed']['pr_auc']:.4f} (vérité {r['val_truth']['pr_auc']:.4f}) | "
            f"test vérité {r['test_truth']['pr_auc']:.4f} | p99 {p99:.1f} ms | explication {explain_ms} ms")

    # ------------------------------------------------------------------ règle de choix
    eligible = {k: r for k, r in rows.items() if r["meets_realtime"] and r["meets_explain"]}
    best_val = max(r["val_observed"]["pr_auc"] for r in eligible.values())
    close = [k for k, r in eligible.items() if best_val - r["val_observed"]["pr_auc"] < TIE]
    chosen = min(close, key=lambda k: (len(rows[k]["components"]), -rows[k]["val_observed"]["pr_auc"]))
    log(f"\nCandidats éligibles (B3, B4) : {list(eligible)}")
    log(f"À moins de {TIE} du meilleur sur la validation : {close}")
    log(f"CHOIX : {chosen}")

    # ------------------------------------------------------------------ confirmation sur le test
    current = "Hybride actuel (XGB + LSTM + AE)"
    confirm = {}
    for k in dict.fromkeys([chosen, "Forêt aléatoire seule", "Forêt aléatoire + LSTM"]):
        if k == current:
            continue
        confirm[k] = {"vs_hybride_actuel_truth": paired_bootstrap(t_test, rows[k]["_test_scores"],
                                                                  rows[current]["_test_scores"], users_test, n_boot),
                      "vs_hybride_actuel_visa_truth": paired_bootstrap(
                          t_test[ch_test == "VISA_VIRTUAL"], rows[k]["_test_scores"][ch_test == "VISA_VIRTUAL"],
                          rows[current]["_test_scores"][ch_test == "VISA_VIRTUAL"], users_test[ch_test == "VISA_VIRTUAL"],
                          n_boot)}
        log(f"Test (vérité) {k} − hybride actuel : {confirm[k]['vs_hybride_actuel_truth']}")

    # calibration (la décision coût-sensible suppose p calibrée)
    calib = {}
    for k in dict.fromkeys([chosen, current]):
        if len(rows[k]["components"]) > 1:
            p = rows[k]["_test_scores"]
            calib[k] = {"brier_observed": round(float(brier_score_loss(y_test, p)), 5),
                        "brier_truth": round(float(brier_score_loss(t_test, p)), 5),
                        "mean_p": round(float(p.mean()), 5), "rate_observed": round(float(y_test.mean()), 5),
                        "rate_truth": round(float(t_test.mean()), 5)}

    # ------------------------------------------------------------------ B5 : fraude jamais vue
    loto = {}
    if not skip_loto:
        params = XGB_CONFIGS[best_xgb] or dict(scale_pos_weight=float((y[fit] == 0).sum() / (y[fit] == 1).sum()),
                                                max_depth=6, min_child_weight=3, learning_rate=0.05)
        ftype_obs = meta["fraud_type"].fillna("").to_numpy()
        legit_test = t_test == 0
        for T in ["SIM_SWAP", "ACCOUNT_TAKEOVER", "SOCIAL_ENGINEERING", "AGENT_FRAUD", "SMURFING",
                  "CARD_TESTING", "CNP_FRAUD", "TOPUP_DRAIN"]:
            keep_fit = fit[ftype_obs[fit] != T]
            keep_es = es[ftype_obs[es] != T]
            keep_val = val[ftype_obs[val] != T]
            log(f"Fraude inconnue : entraînement sans {T}...")
            m = train_xgb(X[keep_fit], y[keep_fit], X[keep_es], y[keep_es], params, seed)
            sx_val, sx_test = xgb_margin(m, X[keep_val]), xgb_margin(m, X[test])
            ae_val = ae_score(ens.ae, X[keep_val])
            meta_T = build_meta_learner().fit(np.column_stack([sx_val, ae_val]), y[keep_val])
            s_combo_val = meta_T.predict_proba(np.column_stack([sx_val, ae_val]))[:, 1]
            s_combo_test = meta_T.predict_proba(np.column_stack([sx_test, S_test_cur[:, 2]]))[:, 1]
            legit_val = truth[keep_val] == 0
            pos = typ_true[test] == T

            def recall_T(s_val, s_test):
                thr = np.quantile(s_val[legit_val], 0.99)       # 1 % de fausses alertes sur la validation
                return round(float((s_test[pos] >= thr).mean()), 3), round(float((s_test[legit_test] >= thr).mean()), 4)

            full_val, full_test = base[best_xgb]
            loto[T] = {
                "n_test": int(pos.sum()),
                "xgb_sans_T": recall_T(sx_val, sx_test),
                "xgb_sans_T_plus_ae": recall_T(s_combo_val, s_combo_test),
                "ae_seul": recall_T(ae_val, S_test_cur[:, 2]),
                "reference_xgb_avec_T": recall_T(full_val[np.isin(val, keep_val)], full_test),
            }
            log(f"    {T}: {loto[T]}")

    # ------------------------------------------------------------------ sauvegarde
    joblib.dump(xgb_models[best_xgb], CANDIDATES_DIR / f"{best_xgb}.pkl")
    if chosen in stacked_models:
        joblib.dump(stacked_models[chosen], CANDIDATES_DIR / "meta_choisi.pkl")
    report = {
        "besoins": {"latence_p99_ms": LATENCY_P99_MS, "explication_p99_ms": EXPLAIN_MS, "egalite": TIE},
        "meilleure_config_xgb": best_xgb,
        "xgb_iterations": {k: int(m.best_iteration) for k, m in xgb_models.items()},
        "couts_composants_ms": comps_cost,
        "candidats": {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in rows.items()},
        "eligibles": list(eligible), "a_egalite": close, "choix": chosen,
        "confirmation_test": confirm, "calibration": calib, "fraude_inconnue": loto,
        "duree_min": round((time.perf_counter() - t0) / 60, 1),
    }
    (REPORTS / "selection_modele.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                                   encoding="utf-8")
    log(f"\nRapport : {REPORTS / 'selection_modele.json'}  ({report['duree_min']} min)")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-loto", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.skip_loto, a.n_boot)
