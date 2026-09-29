"""
Comparaison statistique de deux entraînements sur le MÊME jeu de test (ex. sans / avec C1).

    - Écart de PR-AUC, global et par canal, avec intervalle de confiance à 95 % par
      bootstrap apparié PAR CLIENT : on rééchantillonne des clients, pas des transactions,
      car les transactions d'un même épisode de fraude ne sont pas indépendantes.
    - Écart de rappel par typologie, au seuil que chaque modèle a choisi sur la validation.
    - Test de McNemar sur les décisions (hypothèse d'indépendance : indicatif seulement).

Usage :
    python -m ml.training.compare_runs --a ml/reports/sans_C1 --b ml/reports --label-a "sans C1" --label-b "avec C1"
    python -m ml.training.compare_runs --a ml/reports/profil_silo --b ml/reports/profil_unifie \
        --label-a silo --label-b unifié --segments card_history          # expérience C2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score


def load_run(d: Path, model: str) -> tuple[pd.DataFrame, float]:
    preds = pd.read_csv(d / "test_predictions.csv")
    res = json.loads((d / "phase3_results.json").read_text(encoding="utf-8"))
    return preds, float(res["results"][model]["threshold"])


CARD_HISTORY_BUCKETS = [(0, 4, "carte_0-4_tx"), (5, 19, "carte_5-19_tx"), (20, 10 ** 9, "carte_20+_tx")]


def card_history_segments(raw_tx_path: Path) -> pd.Series:
    """Pour chaque transaction carte : nombre de transactions carte ANTÉRIEURES du titulaire,
    regroupé en tranches (cartes neuves / peu utilisées / habituelles). Hypothèse C2 : le profil
    unifié aide surtout les cartes sans historique."""
    tx = pd.read_csv(raw_tx_path, usecols=["transaction_id", "timestamp", "user_id", "channel"], low_memory=False)
    card = tx[tx.channel == "VISA_VIRTUAL"].sort_values(["timestamp", "transaction_id"], kind="mergesort")
    prior = card.groupby("user_id").cumcount()
    labels = pd.Series(index=card.index, dtype=object)
    for lo, hi, name in CARD_HISTORY_BUCKETS:
        labels[(prior >= lo) & (prior <= hi)] = name
    return pd.Series(labels.to_numpy(), index=card["transaction_id"].to_numpy())


def recall(y, hit, mask=None) -> float:
    m = (y == 1) if mask is None else (y == 1) & mask
    return float(hit[m].mean()) if m.any() else float("nan")


def compare(dir_a: Path, dir_b: Path, model: str = "hybrid", n_boot: int = 1000, seed: int = 0,
            segments: pd.Series | None = None, truth: bool = False) -> dict:
    """segments : étiquette de segment par transaction_id (ex. card_history_segments) ;
    PR-AUC et rappel sont alors aussi calculés par segment.
    truth : évaluer sur la vérité terrain (fraudes jamais signalées incluses) plutôt que sur
    les étiquettes connues de l'opérateur."""
    a, thr_a = load_run(dir_a, model)
    b, thr_b = load_run(dir_b, model)
    label_col, type_col = ("is_fraud_true", "fraud_type_true") if truth else ("is_fraud", "fraud_type")
    if label_col not in a.columns:
        raise ValueError(f"colonne {label_col} absente : relancer le prétraitement puis l'entraînement")
    a = a.assign(is_fraud=a[label_col], fraud_type=a[type_col])
    df = a[["transaction_id", "user_id", "channel", "is_fraud", "fraud_type", f"score_{model}"]].merge(
        b[["transaction_id", f"score_{model}"]], on="transaction_id", suffixes=("_a", "_b"))
    if len(df) != len(a) or len(df) != len(b):
        raise ValueError("les deux runs n'ont pas le même jeu de test")
    y = df["is_fraud"].to_numpy()
    sa, sb = df[f"score_{model}_a"].to_numpy(), df[f"score_{model}_b"].to_numpy()
    hit_a, hit_b = sa >= thr_a, sb >= thr_b
    channel, ftype = df["channel"].to_numpy(), df["fraud_type"].fillna("").to_numpy()
    seg = (df["transaction_id"].map(segments).fillna("").to_numpy() if segments is not None
           else np.full(len(df), "", dtype=object))
    seg_names = sorted(s for s in set(seg) if s)

    def metrics(idx) -> dict:
        yy, out = y[idx], {}
        groups = [("global", slice(None)), ("MOBILE_MONEY", channel[idx] == "MOBILE_MONEY"),
                  ("VISA_VIRTUAL", channel[idx] == "VISA_VIRTUAL")]
        groups += [(s, seg[idx] == s) for s in seg_names]
        for name, m in groups:
            ys = yy[m]
            if 0 < ys.sum() < len(ys):
                out[f"pr_auc_{name}"] = (average_precision_score(ys, sa[idx][m]),
                                         average_precision_score(ys, sb[idx][m]))
        for s in seg_names:
            out[f"recall_{s}"] = (recall(yy, hit_a[idx], seg[idx] == s), recall(yy, hit_b[idx], seg[idx] == s))
        out["recall_global"] = (recall(yy, hit_a[idx]), recall(yy, hit_b[idx]))
        for t in sorted(set(ftype[y == 1])):
            out[f"recall_{t}"] = (recall(yy, hit_a[idx], ftype[idx] == t), recall(yy, hit_b[idx], ftype[idx] == t))
        return out

    point = metrics(np.arange(len(df)))
    rng = np.random.default_rng(seed)
    groups = list(df.groupby("user_id").indices.values())   # lignes de test de chaque client
    diffs: dict[str, list] = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        for k, (va, vb) in metrics(idx).items():
            diffs[k].append(vb - va)

    def n_fraud_for(k: str) -> int | None:
        name = k.split("_", 1)[1] if k.startswith(("recall_", "pr_auc_")) else ""
        if name in seg_names:
            return int(((y == 1) & (seg == name)).sum())
        if k.startswith("recall_") and name in set(ftype):
            return int(((y == 1) & (ftype == name)).sum())
        return None

    table = {}
    for k, (va, vb) in point.items():
        d = np.array([x for x in diffs[k] if x == x])
        lo, hi = (np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan))
        n_fraud = n_fraud_for(k)
        table[k] = {"a": round(va, 4), "b": round(vb, 4), "delta": round(vb - va, 4),
                    "ci95": [round(float(lo), 4), round(float(hi), 4)],
                    "significant": bool(lo > 0 or hi < 0), **({"n_fraud": n_fraud} if n_fraud else {})}

    # McNemar sur les erreurs de décision (b01 : A juste / B faux ; b10 : A faux / B juste)
    ok_a, ok_b = hit_a == (y == 1), hit_b == (y == 1)
    b01, b10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p = binomtest(b10, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"model": model, "labels_used": label_col, "n_test": int(len(df)), "n_boot": n_boot, "metrics": table,
            "mcnemar": {"a_right_b_wrong": b01, "a_wrong_b_right": b10, "p_value": float(p)}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="dossier de rapports du run de référence")
    p.add_argument("--b", required=True, help="dossier de rapports du run comparé")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--model", default="hybrid")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=None, help="fichier JSON de sortie (défaut : <b>/comparison.json)")
    p.add_argument("--segments", choices=["card_history"], default=None,
                   help="card_history : résultats par ancienneté d'usage de la carte (C2)")
    p.add_argument("--raw-tx", default="ml/data/raw/transactions.csv")
    p.add_argument("--truth", action="store_true",
                   help="évaluer sur la vérité terrain (fraudes non signalées incluses)")
    a = p.parse_args()

    segments = card_history_segments(Path(a.raw_tx)) if a.segments == "card_history" else None
    res = compare(Path(a.a), Path(a.b), a.model, a.n_boot, segments=segments, truth=a.truth)
    res["labels"] = {"a": a.label_a, "b": a.label_b}
    out = Path(a.out) if a.out else Path(a.b) / "comparison.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{a.label_b} vs {a.label_a} — modèle « {a.model} », {res['n_test']:,} transactions de test, "
          f"bootstrap par client ×{a.n_boot} — étiquettes : "
          f"{'VÉRITÉ TERRAIN' if a.truth else 'connues de l opérateur'}")
    print(f"{'mesure':<30}{a.label_a:>10}{a.label_b:>10}{'écart':>9}   IC 95 %           ")
    for k, m in res["metrics"].items():
        star = "  *" if m["significant"] else ""
        n = f"  (n={m['n_fraud']})" if "n_fraud" in m else ""
        print(f"{k:<30}{m['a']:>10.4f}{m['b']:>10.4f}{m['delta']:>+9.4f}   "
              f"[{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]{star}{n}")
    mc = res["mcnemar"]
    print(f"\nMcNemar : {mc['a_wrong_b_right']} erreurs corrigées, {mc['a_right_b_wrong']} erreurs introduites, "
          f"p = {mc['p_value']:.3g}")
    print("* : l'intervalle de confiance exclut 0 (écart significatif au seuil de 5 %)")
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()
