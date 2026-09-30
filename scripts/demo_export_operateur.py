"""Fabrique un FAUX export « comme celui d'un opérateur » à partir des données synthétiques, pour
tester et démontrer l'intégration des données réelles (ml.onboarding) sans données réelles.

Le fichier imite les défauts d'un vrai export : séparateur « ; », virgule décimale, dates en
UTC au format jour/mois/année, numéros au format +243 81 234 5678, libellés en français,
provinces avec accents, colonnes personnelles inutiles (nom, carte d'électeur) qui ne doivent
JAMAIS être copiées, fraudes dans un fichier séparé avec leur date de plainte.

    python scripts/demo_export_operateur.py --operator VODACOM --users 600
    -> data/exports_operateurs/demo_vodacom/{transactions.csv, fraudes.csv, kyc.csv}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TYPE_FR = {"P2P_SEND": "Transfert envoyé", "P2P_RECEIVE": "Transfert reçu", "CASH_IN": "Dépôt",
           "CASH_OUT": "Retrait", "MERCHANT_PAYMENT": "Paiement marchand", "AIRTIME": "Achat crédit",
           "BILL_PAYMENT": "Paiement facture"}
CANAL_FR = {"APP": "Application", "USSD": "*1222#", "AGENT": "Agent"}
ACCENTS = {"Kasai-Oriental": "Kasaï-Oriental", "Kasai-Central": "Kasaï-Central", "Kasai": "Kasaï"}


def msisdn(v) -> str | None:
    if v is None or v != v:
        return None
    s = str(v).split(".")[0]
    return f"+{s[:3]} {s[3:5]} {s[5:8]} {s[8:]}" if s.isdigit() and len(s) == 12 else s


def fr_number(x: pd.Series) -> pd.Series:
    return x.map(lambda v: "" if v != v else f"{v:.2f}".replace(".", ","))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--operator", default="VODACOM", choices=["VODACOM", "AIRTEL", "ORANGE"])
    p.add_argument("--users", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    raw = ROOT / "ml/data/raw"
    tx = pd.read_csv(raw / "transactions.csv", low_memory=False, dtype={"wallet_id": str, "counterparty_id": str})
    users = pd.read_csv(raw / "users.csv", dtype={"wallet_id": str})
    users = users[users["operator"] == a.operator]
    users = users.sample(min(a.users, len(users)), random_state=a.seed)
    tx = tx[tx["user_id"].isin(users["user_id"]) & (tx["channel"] == "MOBILE_MONEY")].copy()
    wallet = users.set_index("user_id")["wallet_id"]
    ts = pd.to_datetime(tx["timestamp"]) - pd.Timedelta(hours=1)          # Kinshasa (UTC+1) -> UTC
    rng = np.random.default_rng(a.seed)
    out = pd.DataFrame({
        "ID_TRANSACTION": tx["transaction_id"],
        "DATE_HEURE": ts.dt.strftime("%d/%m/%Y %H:%M:%S"),
        "MSISDN_CLIENT": tx["user_id"].map(wallet).map(msisdn),
        "NOM_CLIENT": "Client " + tx["user_id"].str[-4:],                      # donnée personnelle inutile
        "NUMERO_CARTE_ELECTEUR": [f"CE{rng.integers(10**9, 10**10)}" for _ in range(len(tx))],
        "TYPE_OPERATION": tx["tx_type"].map(TYPE_FR),
        "MONTANT": fr_number(tx["amount"]),
        "DEVISE": tx["currency"].map({"CDF": "FC", "USD": "USD"}),
        "STATUT": np.where(tx["status"] == "SUCCESS", "Réussi", "Echoué"),
        "MSISDN_CONTREPARTIE": tx["counterparty_id"].map(msisdn),
        "CODE_AGENT": tx["agent_id"], "CODE_MARCHAND": tx["merchant_id"],
        "IMEI": tx["device_id"], "CANAL": tx["access_channel"].map(CANAL_FR),
        "PROVINCE": tx["location_province"].map(lambda v: ACCENTS.get(v, v)),
        "SOLDE_AVANT": fr_number(tx["balance_before_usd"] * np.where(tx["currency"] == "CDF", 2850.0, 1.0)),
    })
    # ce que l'opérateur SAIT : fraudes signalées seulement, avec la date de la plainte
    rep = tx[tx["is_fraud"].eq(1) & tx["fraud_reported_at"].notna()]
    frauds = pd.DataFrame({"ID_TRANSACTION": rep["transaction_id"],
                           "DATE_PLAINTE": (pd.to_datetime(rep["fraud_reported_at"]) - pd.Timedelta(hours=1))
                           .dt.strftime("%d/%m/%Y %H:%M:%S"),
                           "TYPOLOGIE": rep["fraud_type"].str.replace("_", " ").str.lower()})
    period = pd.Timestamp("2025-06-01")
    kyc = pd.DataFrame({"MSISDN": users["wallet_id"].map(msisdn), "NOM": "Client",
                        "PROVINCE_RESIDENCE": users["province"].map(lambda v: ACCENTS.get(v, v)),
                        "NIVEAU_KYC": users["kyc_level"],
                        "PLAFOND_TRANSACTION": fr_number(users["kyc_tx_limit_usd"] * 2850.0),
                        "DATE_OUVERTURE": (period - pd.to_timedelta(users["account_age_days"], unit="D"))
                        .dt.strftime("%d/%m/%Y")})
    dest = Path(a.out or ROOT / f"data/exports_operateurs/demo_{a.operator.lower()}")
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "transactions.csv", sep=";", index=False, encoding="utf-8")
    frauds.to_csv(dest / "fraudes.csv", sep=";", index=False, encoding="utf-8")
    kyc.to_csv(dest / "kyc.csv", sep=";", index=False, encoding="utf-8")
    print(f"[OK] {len(out):,} transactions, {len(frauds):,} fraudes signalées, {len(kyc):,} clients -> {dest}")


if __name__ == "__main__":
    main()
