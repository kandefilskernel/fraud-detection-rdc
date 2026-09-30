"""Contrôle qualité des données importées, AVANT tout entraînement.

Erreurs bloquantes : l'entraînement est refusé (sauf --force, déconseillé).
Avertissements : l'entraînement est possible mais le rapport dit ce qui est neutralisé ou supposé.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.onboarding.importer import BALANCE_UNKNOWN_USD
from ml.preprocessing.train_test_split_normalize import SPLIT_NAMES, temporal_split

MIN_TX, GOOD_TX = 10_000, 100_000
MIN_DAYS, GOOD_DAYS = 30, 90
MIN_FRAUDS_EVAL, GOOD_FRAUDS_EVAL = 20, 100      # fraudes en validation ET en test
MAX_DROP_BLOCKING, MAX_DROP_WARNING = 0.10, 0.01
MAX_FRAUD_RATE, MIN_FRAUD_RATE = 0.20, 0.0001


def assess(tx: pd.DataFrame, users: pd.DataFrame, logs: dict) -> dict:
    blocking, warnings = [], []
    n = len(tx)
    ts = pd.to_datetime(tx["timestamp"])
    days = (ts.max() - ts.min()).total_seconds() / 86400 if n else 0.0
    y = tx["is_fraud"].astype(int).to_numpy() if n else np.array([])
    frauds = int(y.sum())

    # ---- volume et période
    if n < MIN_TX:
        blocking.append(f"{n:,} transactions : minimum {MIN_TX:,} pour entraîner et évaluer un modèle")
    elif n < GOOD_TX:
        warnings.append(f"{n:,} transactions : résultats fragiles en dessous de {GOOD_TX:,}")
    if days < MIN_DAYS:
        blocking.append(f"période de {days:.0f} jours : minimum {MIN_DAYS} (le split est temporel)")
    elif days < GOOD_DAYS:
        warnings.append(f"période de {days:.0f} jours : {GOOD_DAYS} jours ou plus recommandés "
                        "(habitudes mensuelles, fins de mois)")
    if n and ts.max() > pd.Timestamp.now() + pd.Timedelta(days=1):
        warnings.append("des transactions sont datées dans le futur : vérifier le fuseau et le format de date")

    # ---- lignes écartées à l'import
    read = sum(l.get("rows_read", 0) for l in logs.get("imports", {}).values())
    dropped = sum(l.get("rows_dropped", 0) for l in logs.get("imports", {}).values())
    if read:
        rate = dropped / read
        detail = "; ".join(f"{src} : {', '.join(f'{k} ({v:,})' for k, v in l.get('dropped', {}).items())}"
                           for src, l in logs["imports"].items() if l.get("dropped"))
        if rate > MAX_DROP_BLOCKING:
            blocking.append(f"{rate:.1%} des lignes écartées à l'import ({detail}) : corriger le fichier "
                            "de correspondance")
        elif rate > MAX_DROP_WARNING:
            warnings.append(f"{rate:.1%} des lignes écartées à l'import ({detail})")
    for src, l in logs.get("imports", {}).items():
        for field, vals in l.get("unknown_values", {}).items():
            warnings.append(f"{src} : valeurs de « {field} » non reconnues {vals} (compléter values.{field})")
        if not l.get("pseudonymized"):
            blocking.append(f"{src} : identifiants NON pseudonymisés. Définir PSEUDONYMIZATION_KEY "
                            "(obligatoire pour des données réelles)")
        for what, k in l.get("defaults", {}).items():
            if "signalement" in what:
                warnings.append(f"{src} : {k:,} fraudes sans date de signalement -> {what}")

    # ---- étiquettes
    split, bounds = temporal_split(ts) if n else (np.array([]), {})
    per_split = {name: {"n": int((split == i).sum()), "frauds": int(y[split == i].sum())}
                 for i, name in enumerate(SPLIT_NAMES)} if n else {}
    if frauds == 0:
        blocking.append("aucune fraude étiquetée : fournir la colonne de fraude ou le fichier de signalements "
                        "(plaintes, fraudes confirmées, contestations)")
    else:
        fr = frauds / n
        if fr > MAX_FRAUD_RATE:
            blocking.append(f"taux de fraude de {fr:.1%} : invraisemblable, vérifier labels.positive_values")
        elif fr < MIN_FRAUD_RATE:
            warnings.append(f"taux de fraude de {fr:.4%} : très faible, les signalements sont-ils complets ?")
        for name in ("val", "test"):
            k = per_split[name]["frauds"]
            if k < MIN_FRAUDS_EVAL:
                blocking.append(f"seulement {k} fraudes dans la tranche « {name} » (fin de période) : minimum "
                                f"{MIN_FRAUDS_EVAL} pour une évaluation crédible. Allonger la période ou "
                                "compléter les signalements récents")
            elif k < GOOD_FRAUDS_EVAL:
                warnings.append(f"{k} fraudes dans la tranche « {name} » : intervalle de confiance large "
                                f"(recommandé : {GOOD_FRAUDS_EVAL}+)")
        late = pd.to_datetime(tx.loc[tx["is_fraud"] == 1, "fraud_reported_at"]) > ts.max()
        if late.mean() > 0.3:
            warnings.append("plus de 30 % des fraudes sont signalées après la fin de l'export : les dernières "
                            "semaines sont sous-étiquetées (prévoir un délai avant d'extraire)")

    # ---- familles de variables neutralisées faute de données
    fam = {}
    def share(mask) -> float:
        return float(mask.mean()) if n else 0.0
    fam["appareil"] = share(tx["device_id"].astype(str).str.startswith("UNK-")
                            | tx["device_id"].astype(str).str.len().eq(0))
    fam["canal d'accès (app / USSD / agent)"] = share(tx["access_channel"] == "UNKNOWN")
    fam["solde"] = share(tx["balance_before_usd"] >= BALANCE_UNKNOWN_USD)
    fam["province de la transaction"] = logs.get("build", {}).get("province_filled_from_home", 0) / n if n else 0
    p2p = tx["tx_type"].isin(["P2P_SEND", "P2P_RECEIVE"])
    fam["contrepartie des transferts"] = float(tx.loc[p2p, "counterparty_id"].isna().mean()) if p2p.any() else 0.0
    cash = tx["tx_type"].isin(["CASH_IN", "CASH_OUT"])
    fam["agent des dépôts / retraits"] = float(tx.loc[cash, "agent_id"].isna().mean()) if cash.any() else 0.0
    for name, missing in fam.items():
        if missing >= 0.99:
            warnings.append(f"« {name} » absent : ces variables sont neutralisées (le modèle ne pourra pas "
                            "s'en servir)")
        elif missing >= 0.30:
            warnings.append(f"« {name} » manquant pour {missing:.0%} des transactions")
    kyc = logs.get("build", {}).get("users", {})
    if kyc.get("total"):
        cov = kyc.get("with_kyc", 0) / kyc["total"]
        if cov < 0.5:
            warnings.append(f"référentiel KYC pour {cov:.0%} des clients seulement : profil prudent par défaut "
                            "pour les autres (niveau KYC, plafond, revenu, ancienneté)")

    stats = {
        "transactions": n, "clients": int(tx["user_id"].nunique()) if n else 0,
        "periode": {"debut": str(ts.min()) if n else None, "fin": str(ts.max()) if n else None,
                    "jours": round(days, 1)},
        "fraudes": frauds, "taux_fraude": round(frauds / n, 6) if n else 0,
        "par_operateur": tx["operator"].fillna("VISA").value_counts().to_dict() if n else {},
        "par_canal": tx["channel"].value_counts().to_dict() if n else {},
        "par_type": tx["tx_type"].value_counts().to_dict() if n else {},
        "fraudes_par_type": tx.loc[tx["is_fraud"] == 1, "fraud_type"].value_counts().to_dict() if n else {},
        "montant_usd": {q: round(float(tx["amount_usd"].quantile(v)), 2)
                        for q, v in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.0))} if n else {},
        "taux_echecs": round(float((tx["status"] == "FAILED").mean()), 4) if n else 0,
        "donnees_manquantes": {k: round(v, 4) for k, v in fam.items()},
        "decoupage_temporel": {"bornes": bounds, "tranches": per_split},
        "sources": logs.get("build", {}).get("sources", []),
    }
    return {"ready": not blocking, "blocking": blocking, "warnings": warnings, "stats": stats}


def write_report(result: dict, workspace: Path) -> Path:
    s = result["stats"]
    lines = [f"# Rapport qualité des données — {workspace.name}", "",
             ("**Verdict : prêt pour l'entraînement.**" if result["ready"]
              else "**Verdict : NON prêt pour l'entraînement** (erreurs bloquantes ci-dessous)."), ""]
    if result["blocking"]:
        lines += ["## Erreurs bloquantes", ""] + [f"- {b}" for b in result["blocking"]] + [""]
    if result["warnings"]:
        lines += ["## Avertissements", ""] + [f"- {w}" for w in result["warnings"]] + [""]
    per = s["decoupage_temporel"]["tranches"]
    lines += ["## Données", "",
              f"- Sources : {', '.join(s['sources'])}",
              f"- {s['transactions']:,} transactions, {s['clients']:,} clients",
              f"- Période : {s['periode']['debut']} → {s['periode']['fin']} ({s['periode']['jours']} jours)",
              f"- Fraudes étiquetées : {s['fraudes']:,} (taux {s['taux_fraude']:.4%})",
              f"- Échecs : {s['taux_echecs']:.2%}",
              f"- Montants (USD) : médiane {s['montant_usd'].get('p50')}, p95 {s['montant_usd'].get('p95')}, "
              f"p99 {s['montant_usd'].get('p99')}, max {s['montant_usd'].get('max')}", "",
              "## Découpage temporel prévu", "", "| Tranche | Transactions | Fraudes |", "|---|---|---|"]
    lines += [f"| {k} | {v['n']:,} | {v['frauds']:,} |" for k, v in per.items()]
    lines += ["", "## Répartition", "", "| Type d'opération | Transactions |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in s["par_type"].items()]
    if s["fraudes_par_type"]:
        lines += ["", "| Typologie de fraude | Nombre |", "|---|---|"]
        lines += [f"| {k} | {v:,} |" for k, v in s["fraudes_par_type"].items()]
    lines += ["", "## Données manquantes (part des transactions concernées)", "", "| Famille | Manquant |",
              "|---|---|"] + [f"| {k} | {v:.0%} |" for k, v in s["donnees_manquantes"].items()]
    path = workspace / "rapport_qualite.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (workspace / "rapport_qualite.json").write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                                               default=str), encoding="utf-8")
    return path
