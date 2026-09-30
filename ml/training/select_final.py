"""
Troisième manche de la sélection de modèle : protocole PRÉ-ENREGISTRÉ, décision sur la
VALIDATION uniquement, test lu une seule fois pour confirmer.

Pourquoi cette manche. Les deux premières (select_model.py, select_forest.py) ont utilisé le
test pour décider : rejet de « XGB + LSTM » sur un bootstrap du test, puis conception de la
2e manche parce que la forêt + LSTM était meilleure sur le test. Leurs écarts « test » ne sont
donc plus une confirmation indépendante : ils sont présentés comme EXPLORATOIRES dans le
mémoire. Cette manche fixe la règle AVANT de calculer quoi que ce soit (PREREGISTRATION
ci-dessous, recopiée telle quelle dans le rapport) et ne réentraîne aucune branche.

Règle (identique aux besoins B1-B6 de select_model.py, appliquée telle qu'écrite) :
    1. éligibles : latence de la CHAÎNE COMPLÈTE (prédiction + explication), mesurée ici dans
       les mêmes conditions pour tous (1 transaction, 1 cœur), p99 ≤ 20 ms (B3/B4) ;
    2. B2 sur la VALIDATION : la PR-AUC Visa ne doit pas être significativement inférieure à
       celle du modèle en production (bootstrap apparié par client, IC 95 %) ;
    3. B1 : meilleure PR-AUC de validation. Critère principal = étiquettes observées (ce que
       l'opérateur connaît). Variante déclarée « validation vérifiée » (vérité terrain de la
       validation, jamais celle du test) : calculée et rapportée, décisionnelle seulement si
       --critere verite ;
    4. B6 : parmi les candidats à moins de 0,005 du meilleur, le moins de composants gagne.

Comparaisons déclarées (bootstrap apparié par client, sur la validation PUIS sur le test ;
correction de Holm sur chaque famille) :
    C_lstm  : Forêt + LSTM        − Forêt seule          (apport du profil séquentiel)
    C_ae    : Forêt + LSTM + AE   − Forêt + LSTM         (apport de l'autoencodeur empilé)
    C_choix : modèle choisi       − modèle en production (si différents)

Autoencodeur hors méta-apprenant (« veille des anomalies ») : seuil = quantile 99,5 % du score
de l'autoencodeur sur les transactions de validation NON signalées ; sur le test, on mesure
la part de fraudes (vérité) signalées par la veille, et surtout celles que le modèle choisi
MANQUE à son seuil. Aucune friction client : signal pour les analystes seulement.

    python -m ml.training.select_final [--critere observe|verite] [--n-boot 1000]
Sorties : ml/reports/selection_finale.json, ml/reports/selection_finale.md
Ensuite : la commande `adopt_model` à lancer est affichée à la fin (rien n'est mis en
production automatiquement).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path("ml/reports")
LATENCY_P99_MS, TIE, ANOMALY_QUANTILE = 20.0, 0.005, 0.995
CURRENT = "Forêt + LSTM + AE"          # architecture en production au 28/09/2026

CANDIDATES = {                         # nom -> composants, du plus simple au plus complexe
    "Forêt seule": ["rf"],
    "XGB seul": ["xgb"],
    "LSTM seul": ["lstm"],
    "Forêt + LSTM": ["rf", "lstm"],
    "XGB + LSTM": ["xgb", "lstm"],
    "Forêt + LSTM + AE": ["rf", "lstm", "ae"],
    "XGB + LSTM + AE": ["xgb", "lstm", "ae"],
}
COMPARISONS = {
    "C_lstm": ("Forêt + LSTM", "Forêt seule"),
    "C_ae": ("Forêt + LSTM + AE", "Forêt + LSTM"),
}
PREREGISTRATION = {
    "decision_sur": "validation uniquement",
    "critere_principal_B1": "PR-AUC (average precision) de validation, étiquettes observées",
    "variante_declaree": "PR-AUC de validation, vérité terrain (scénario « échantillon vérifié »)",
    "eligibilite_B3_B4": f"chaîne complète prédiction + explication, p99 ≤ {LATENCY_P99_MS} ms, 1 cœur",
    "B2": "PR-AUC Visa (validation) pas significativement < modèle en production (IC 95 %, bootstrap client)",
    "B6_egalite": TIE,
    "comparaisons": {k: f"{a} − {b}" for k, (a, b) in COMPARISONS.items()} | {"C_choix": "choisi − production"},
    "correction_multiple": "Holm, par famille (validation / test × étiquettes)",
    "veille_anomalies": f"seuil = quantile {ANOMALY_QUANTILE} du score AE, validation non signalée",
    "test": "lu une seule fois, après la décision ; aucune décision n'en dépend",
}
SCOPES = (("global", None), ("mobile_money", "MOBILE_MONEY"), ("visa", "VISA_VIRTUAL"))


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------- statistiques (sans torch)
def client_groups(users: np.ndarray) -> list[np.ndarray]:
    codes, uniq = pd.factorize(users)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def paired_bootstrap(y, s_a, s_b, users, n_boot: int = 1000, seed: int = 0) -> dict:
    """Écart de PR-AUC (a − b) avec rééchantillonnage des CLIENTS ; p-valeur bilatérale."""
    from sklearn.metrics import average_precision_score as ap
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    groups = client_groups(np.asarray(users))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if 0 < y[idx].sum() < len(idx):
            diffs.append(ap(y[idx], s_a[idx]) - ap(y[idx], s_b[idx]))
    d = np.asarray(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
    return {"delta": round(float(ap(y, s_a) - ap(y, s_b)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_value": round(p, 4), "n_fraud": int(y.sum())}


def holm(pvalues: dict) -> dict:
    """p-valeurs ajustées de Holm-Bonferroni (contrôle du risque de 1re espèce de la famille)."""
    keys = sorted(pvalues, key=pvalues.get)
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = round(running, 4)
    return out


def apply_rule(rows: dict, criterion: str, tie: float = TIE) -> dict:
    """rows[nom] = {"n_components", "val": {"observe", "verite"}, "eligible", "b2_ok"}.
    Renvoie le choix et les étapes (fonction pure, testée dans ml/tests/test_models.py)."""
    eligible = [k for k, r in rows.items() if r["eligible"] and r["b2_ok"]]
    if not eligible:
        raise RuntimeError("aucun candidat ne respecte B2-B4")
    best = max(rows[k]["val"][criterion] for k in eligible)
    close = [k for k in eligible if best - rows[k]["val"][criterion] < tie]
    chosen = min(close, key=lambda k: (rows[k]["n_components"], -rows[k]["val"][criterion]))
    return {"eligibles": eligible, "meilleur_val": round(best, 4), "a_egalite": close, "choix": chosen}


def metrics(y, s, channel) -> dict:
    from sklearn.metrics import average_precision_score as ap
    from ml.training.evaluation import recall_at_fpr
    out = {"pr_auc": round(float(ap(y, s)), 4), "recall_at_1pct_fpr": round(recall_at_fpr(y, s, 0.01), 4)}
    for name, ch in SCOPES[1:]:
        m = channel == ch
        out[f"pr_auc_{name}"] = round(float(ap(y[m], s[m])), 4) if y[m].sum() else None
    return out


def compare_family(pairs: dict, scores: dict, y, users, channel, n_boot: int) -> dict:
    """pairs[nom] = (a, b) ; bootstrap global / MM / Visa puis Holm sur l'ensemble de la famille."""
    res = {}
    for name, (a, b) in pairs.items():
        for scope, ch in SCOPES:
            m = np.ones(len(y), bool) if ch is None else channel == ch
            res[f"{name}/{scope}"] = paired_bootstrap(y[m], scores[a][m], scores[b][m], users[m], n_boot)
    adj = holm({k: v["p_value"] for k, v in res.items()})
    for k, v in res.items():
        v["p_holm"] = adj[k]
        v["significatif_holm"] = adj[k] < 0.05
    return res


# ---------------------------------------------------------------------- latence de la chaîne complète
def time_ms(fn, n: int = 300) -> dict:
    for _ in range(20):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return {"p50": round(float(np.percentile(t, 50)), 3), "p99": round(float(np.percentile(t, 99)), 3)}


def chain_latency(comps: list[str], parts: dict) -> dict:
    """Chaîne réelle : chaque composant (arbres = prédiction + explication) puis méta-apprenant."""
    fns = [parts[c] for c in comps]

    def run():
        z = np.array([f() for f in fns], dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-float(z.sum())))

    return time_ms(run)


# ---------------------------------------------------------------------- programme principal
def main(criterion: str = "observe", n_boot: int = 1000, seed: int = 42) -> dict:
    import joblib
    import torch
    import xgboost as xgb

    from ml.models.autoencoder_branch import ae_score
    from ml.models.flat_forest import FlatForest
    from ml.models.hybrid_ensemble import HybridEnsemble, rf_logit
    from ml.models.lstm_attention_branch import gather_sequences, lstm_logit
    from ml.models.meta_learner import build_meta_learner
    from ml.models.xgboost_branch import train_xgboost_branch, xgb_logit
    from ml.preprocessing.train_test_split_normalize import ARTIFACTS_DIR, load_processed
    from ml.training.evaluation import best_f1_threshold
    from sklearn.model_selection import cross_val_predict

    t0 = time.perf_counter()
    crit_key = {"observe": "observe", "verite": "verite"}[criterion]
    data = load_processed()
    X, y, seq_idx, meta = data.X, data.y, data.seq_idx, data.meta
    fit, es, val, test = data.rows("train"), data.rows("early_stop"), data.rows("val"), data.rows("test")
    truth = meta["is_fraud_true"].to_numpy().astype(int)
    channel = meta["channel"].to_numpy()
    users = meta["user_id"].to_numpy()
    typ_true = meta["fraud_type_true"].fillna("").to_numpy()

    prod = HybridEnsemble.load(ARTIFACTS_DIR)
    if prod.tree_kind != "random_forest" or prod.ae is None:
        raise SystemExit(f"modèle en production inattendu ({prod.branches}) : ce protocole suppose "
                         f"forêt + LSTM + AE (état du 28/09/2026)")
    xgb_path = ARTIFACTS_DIR / "xgb_branch.pkl"
    if xgb_path.exists():
        xgb_model = joblib.load(xgb_path)
        xgb_origin = "ml/artifacts/xgb_branch.pkl (branche de l'ancien hybride)"
    else:
        log("xgb_branch.pkl absent : réentraînement de la branche XGBoost (configuration d'origine)...")
        xgb_model = train_xgboost_branch(X[fit], y[fit], X[es], y[es], 600, seed)
        xgb_origin = "réentraînée (configuration d'origine, graine 42)"

    log("Scores des branches (validation, test)...")
    base = {}
    for part, rows in (("val", val), ("test", test)):
        base[("rf", part)] = rf_logit(prod.tree, X[rows])
        base[("xgb", part)] = xgb_logit(xgb_model, X[rows])
        base[("lstm", part)] = lstm_logit(prod.lstm, X, seq_idx, rows)
        base[("ae", part)] = ae_score(prod.ae, X[rows])

    # ------------------------------------------------------------ latence (même protocole pour tous)
    torch.set_num_threads(1)
    flat = FlatForest(prod.tree)
    booster = xgb_model.get_booster()
    booster.set_param({"nthread": 1})
    best_it = getattr(xgb_model, "best_iteration", None)
    it_range = (0, best_it + 1) if best_it is not None else (0, 0)
    x1 = X[test[:1]].astype(np.float32)
    seq, mask = gather_sequences(X, seq_idx, test[:1])
    x1t = torch.from_numpy(x1)

    def part_rf():
        flat.explain_one(x1[0])                              # explication Saabas (inclut la prédiction)
        return 0.0

    def part_xgb():
        booster.inplace_predict(x1, iteration_range=it_range, predict_type="margin")
        booster.predict(xgb.DMatrix(x1), pred_contribs=True, iteration_range=it_range)   # TreeSHAP
        return 0.0

    def part_lstm():
        with torch.no_grad():
            prod.lstm(seq, mask, return_attention=True)
        return 0.0

    def part_ae():
        with torch.no_grad():
            prod.ae.reconstruction_error(x1t)
        return 0.0

    parts = {"rf": part_rf, "xgb": part_xgb, "lstm": part_lstm, "ae": part_ae}
    latency = {name: chain_latency(comps, parts) for name, comps in CANDIDATES.items()}
    torch.set_num_threads(4)
    log("Latence p99 (chaîne complète, 1 cœur) : " + ", ".join(f"{k} {v['p99']} ms" for k, v in latency.items()))

    # ------------------------------------------------------------ scores des candidats
    scores_val, scores_test, rows = {}, {}, {}
    y_val, t_val = y[val], truth[val]
    for name, comps in CANDIDATES.items():
        Sv = np.column_stack([base[(c, "val")] for c in comps])
        St = np.column_stack([base[(c, "test")] for c in comps])
        if len(comps) > 1:
            m = build_meta_learner()
            # validation : prédictions hors pli (évaluation honnête du choix)
            scores_val[name] = cross_val_predict(m, Sv, y_val, cv=5, method="predict_proba")[:, 1]
            m.fit(Sv, y_val)
            scores_test[name] = m.predict_proba(St)[:, 1]
        else:
            scores_val[name], scores_test[name] = Sv[:, 0], St[:, 0]
        rows[name] = {"components": comps, "n_components": len(comps),
                      "val": {"observe": metrics(y_val, scores_val[name], channel[val])["pr_auc"],
                              "verite": metrics(t_val, scores_val[name], channel[val])["pr_auc"]},
                      "val_detail": {"observe": metrics(y_val, scores_val[name], channel[val]),
                                     "verite": metrics(t_val, scores_val[name], channel[val])},
                      "latency_ms": latency[name], "eligible": latency[name]["p99"] <= LATENCY_P99_MS}

    # ------------------------------------------------------------ B2 sur la validation
    visa_val = channel[val] == "VISA_VIRTUAL"
    for name, r in rows.items():
        if name == CURRENT:
            r["b2"], r["b2_ok"] = None, True
            continue
        r["b2"] = paired_bootstrap(y_val[visa_val], scores_val[name][visa_val], scores_val[CURRENT][visa_val],
                                   users[val][visa_val], n_boot)
        r["b2_ok"] = not (r["b2"]["ci95"][1] < 0)

    # ------------------------------------------------------------ DÉCISION (validation seule)
    decision = apply_rule(rows, crit_key)
    decision_variante = apply_rule(rows, "verite" if crit_key == "observe" else "observe")
    chosen = decision["choix"]
    log(f"\nÉligibles (B2-B4) : {decision['eligibles']}")
    log(f"À moins de {TIE} du meilleur ({crit_key}) : {decision['a_egalite']}")
    log(f"CHOIX (validation, critère {crit_key}) : {chosen}")
    log(f"Pour information, critère {'verite' if crit_key == 'observe' else 'observe'} : {decision_variante['choix']}")

    pairs = dict(COMPARISONS)
    if chosen != CURRENT:
        pairs["C_choix"] = (chosen, CURRENT)
    log("Comparaisons déclarées sur la validation...")
    comp_val = {lab: compare_family(pairs, scores_val, yy, users[val], channel[val], n_boot)
                for lab, yy in (("observe", y_val), ("verite", t_val))}

    # ------------------------------------------------------------ TEST : lecture unique
    log("Lecture unique du test (aucune décision n'en dépend)...")
    y_test, t_test = y[test], truth[test]
    test_metrics = {name: {"observe": metrics(y_test, scores_test[name], channel[test]),
                           "verite": metrics(t_test, scores_test[name], channel[test])} for name in CANDIDATES}
    comp_test = {lab: compare_family(pairs, scores_test, yy, users[test], channel[test], n_boot)
                 for lab, yy in (("observe", y_test), ("verite", t_test))}

    # ------------------------------------------------------------ veille des anomalies (AE séparé)
    ae_val, ae_test = base[("ae", "val")], base[("ae", "test")]
    anomaly_thr = float(np.quantile(ae_val[y_val == 0], ANOMALY_QUANTILE))
    flag = ae_test >= anomaly_thr
    comps = CANDIDATES[chosen]
    if len(comps) > 1:
        m = build_meta_learner().fit(np.column_stack([base[(c, "val")] for c in comps]), y_val)
        p_val_fit = m.predict_proba(np.column_stack([base[(c, "val")] for c in comps]))[:, 1]
    else:
        p_val_fit = scores_val[chosen]
    thr_model = best_f1_threshold(y_val, p_val_fit)
    alert = scores_test[chosen] >= thr_model
    fraud, legit = t_test == 1, t_test == 0
    per_type = {}
    for T in sorted(set(typ_true[test][fraud]) - {""}):
        mT = typ_true[test] == T
        missed = mT & ~alert
        per_type[T] = {"n": int(mT.sum()), "modele": round(float(alert[mT].mean()), 3),
                       "veille": round(float(flag[mT].mean()), 3),
                       "veille_parmi_manquees": round(float(flag[missed].mean()), 3) if missed.any() else None}
    anomaly = {"seuil_score_ae": round(anomaly_thr, 5), "quantile": ANOMALY_QUANTILE,
               "seuil_modele_choisi": round(thr_model, 5),
               "taux_legitimes_signales": round(float(flag[legit].mean()), 5),
               "fraudes_signalees": round(float(flag[fraud].mean()), 4),
               "fraudes_manquees_par_le_modele": int((fraud & ~alert).sum()),
               "dont_signalees_par_la_veille": int((fraud & ~alert & flag).sum()),
               "par_typologie": per_type}

    report = {"preenregistrement": PREREGISTRATION, "critere_utilise": crit_key,
              "branche_xgboost": xgb_origin, "production_au_depart": CURRENT,
              "candidats": {k: {kk: vv for kk, vv in r.items()} for k, r in rows.items()},
              "decision": decision, "decision_variante": decision_variante,
              "comparaisons_validation": comp_val, "test_lecture_unique": test_metrics,
              "comparaisons_test": comp_test, "veille_anomalies": anomaly,
              "duree_min": round((time.perf_counter() - t0) / 60, 1)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "selection_finale.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                                   encoding="utf-8")
    (REPORTS / "selection_finale.md").write_text(render_markdown(report), encoding="utf-8")
    log(f"\nRapports : {REPORTS / 'selection_finale.json'}, {REPORTS / 'selection_finale.md'} "
        f"({report['duree_min']} min)")
    log("Mise en production (après relecture du rapport) :\n    " + adopt_command(chosen))
    return report


def adopt_command(chosen: str) -> str:
    names = {"rf": "random_forest", "xgb": "xgboost", "lstm": "lstm_attention", "ae": "autoencoder"}
    comps = [names[c] for c in CANDIDATES[chosen]]
    if comps[0] not in ("random_forest", "xgboost"):
        return (f"(« {chosen} » n'a pas de branche arbres, requise par le service de scoring "
                f"(explication par variable, B4) : à discuter avant toute mise en production)")
    extra = "" if "autoencoder" in comps else " --anomaly-detector"
    return f"python -m ml.training.adopt_model --branches {' '.join(comps)}{extra}"


def _fmt_cmp(v: dict) -> str:
    star = " *" if v["significatif_holm"] else ""
    return f"{v['delta']:+.4f} [{v['ci95'][0]:+.4f} ; {v['ci95'][1]:+.4f}] p_Holm={v['p_holm']:.3f}{star}"


def render_markdown(r: dict) -> str:
    L = ["# Sélection finale du modèle (protocole pré-enregistré)", "",
         f"Critère décisionnel : **{r['critere_utilise']}** (validation). Test lu une seule fois.", "",
         "## Candidats (validation)", "",
         "| Candidat | PR-AUC val. observée | PR-AUC val. vérité | Visa val. (obs.) | p99 chaîne complète | B2 | Éligible |",
         "|---|---|---|---|---|---|---|"]
    for k, c in r["candidats"].items():
        b2 = "réf." if c["b2"] is None else ("ok" if c["b2_ok"] else f"échec {c['b2']['delta']:+.3f}")
        visa = c["val_detail"]["observe"]["pr_auc_visa"]
        L.append(f"| {k} | {c['val']['observe']:.4f} | {c['val']['verite']:.4f} | {visa} | "
                 f"{c['latency_ms']['p99']} ms | {b2} | {'oui' if c['eligible'] and c['b2_ok'] else 'non'} |")
    d, dv = r["decision"], r["decision_variante"]
    L += ["", f"**Choix : {d['choix']}** (à égalité : {', '.join(d['a_egalite'])}).",
          f"Variante (autre critère) : {dv['choix']}.", "",
          "## Comparaisons déclarées", "", "`*` = significatif après correction de Holm.", ""]
    for part, comp in (("Validation", r["comparaisons_validation"]), ("Test (lecture unique)", r["comparaisons_test"])):
        L += [f"### {part}", "", "| Comparaison / périmètre | Étiquettes observées | Vérité terrain |", "|---|---|---|"]
        for key in comp["observe"]:
            L.append(f"| {key} | {_fmt_cmp(comp['observe'][key])} | {_fmt_cmp(comp['verite'][key])} |")
        L.append("")
    L += ["## Test (lecture unique)", "", "| Candidat | PR-AUC obs. | vérité | MM vérité | Visa vérité |", "|---|---|---|---|---|"]
    for k, m in r["test_lecture_unique"].items():
        L.append(f"| {k} | {m['observe']['pr_auc']:.4f} | {m['verite']['pr_auc']:.4f} | "
                 f"{m['verite']['pr_auc_mobile_money']} | {m['verite']['pr_auc_visa']} |")
    a = r["veille_anomalies"]
    L += ["", "## Veille des anomalies (autoencodeur hors méta-apprenant)", "",
          f"Seuil : quantile {a['quantile']} des transactions de validation non signalées. "
          f"Légitimes signalés : {a['taux_legitimes_signales']:.3%}. Fraudes signalées : {a['fraudes_signalees']:.1%}.",
          f"Fraudes manquées par le modèle choisi : {a['fraudes_manquees_par_le_modele']}, "
          f"dont {a['dont_signalees_par_la_veille']} signalées par la veille.", "",
          "| Typologie | n | Modèle | Veille | Veille parmi les manquées |", "|---|---|---|---|---|"]
    for T, v in a["par_typologie"].items():
        L.append(f"| {T} | {v['n']} | {v['modele']} | {v['veille']} | {v['veille_parmi_manquees']} |")
    L += ["", "Commande de mise en production proposée :", "", f"    {adopt_command(d['choix'])}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--critere", choices=["observe", "verite"], default="observe")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.critere, a.n_boot)
