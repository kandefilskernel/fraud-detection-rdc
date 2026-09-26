"""
Simule le travail des analystes et les plaintes des clients (démonstration de la boucle
d'apprentissage continu), via l'API du back-office — comme le feraient de vrais utilisateurs.

    - chaque dossier ouvert reçoit le verdict correspondant à la vérité terrain
      (FRAUDE_CONFIRMEE ou FAUX_POSITIF) : rôle de l'analyste après son enquête ;
    - une partie des fraudes APPROUVÉES par le modèle fait l'objet d'une plainte client
      (--complaint-rate, 70 % par défaut) : c'est ainsi qu'arrivent les faux négatifs.

La vérité terrain vient du jeu synthétique (colonne is_fraud) ; elle n'est JAMAIS
transmise au modèle autrement que par ces verdicts, exactement comme en production.

Usage : python scripts/simulate_feedback.py --complaint-rate 0.7
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from seed_database import DEMO_USERS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8003")
    p.add_argument("--complaint-rate", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rng = random.Random(a.seed)

    truth = pd.read_csv(ROOT / "ml/data/raw/transactions.csv", usecols=["transaction_id", "is_fraud", "fraud_type"])
    truth = truth.set_index("transaction_id")
    email, _, _, password = DEMO_USERS[0]
    c = httpx.Client(base_url=a.url, timeout=30)
    token = c.post("/auth/login", data={"username": email, "password": password}).json()["access_token"]
    c.headers["Authorization"] = f"Bearer {token}"

    # 1) verdicts sur les dossiers ouverts
    verdicts = {"FRAUDE_CONFIRMEE": 0, "FAUX_POSITIF": 0}
    while True:
        items = c.get("/cases", params={"status": "OUVERT", "size": 200}).json()["items"]
        if not items:
            break
        for case in items:
            is_fraud = int(truth.loc[case["transaction_id"], "is_fraud"]) if case["transaction_id"] in truth.index else 0
            status = "FRAUDE_CONFIRMEE" if is_fraud else "FAUX_POSITIF"
            note = ("Client injoignable puis opération non reconnue ; compte sécurisé." if is_fraud
                    else "Client joint : opération confirmée par le titulaire.")
            c.patch(f"/cases/{case['id']}", json={"status": "EN_COURS"})
            r = c.patch(f"/cases/{case['id']}", json={"status": status, "resolution_note": note})
            if r.status_code == 200:
                verdicts[status] += 1
    print(f"verdicts analystes : {verdicts}")

    # 2) plaintes clients pour les fraudes laissées passer
    # on collecte d'abord la liste (étiqueter pendant la pagination décalerait les pages)
    approved, page = [], 1
    while True:
        res = c.get("/transactions", params={"action": "APPROVE", "size": 500, "page": page, "labeled": False}).json()
        if not res["items"]:
            break
        approved += [t["transaction_id"] for t in res["items"]]
        page += 1
    missed_ids = [t for t in approved if t in truth.index and int(truth.loc[t, "is_fraud"]) == 1]
    complaints = 0
    for tid in missed_ids:
        if rng.random() < a.complaint_rate:
            r = c.post(f"/cases/complaint/{tid}",
                       json={"note": f"Le client ne reconnaît pas l'opération ({truth.loc[tid, 'fraud_type']})."})
            complaints += r.status_code == 201
    missed = len(missed_ids)
    print(f"fraudes approuvées par le modèle : {missed} ; plaintes enregistrées : {complaints}")


if __name__ == "__main__":
    main()
