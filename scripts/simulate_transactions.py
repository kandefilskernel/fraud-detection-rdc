"""
Simulateur de trafic opérateurs (démo temps réel).

Rejoue les transactions de la période de TEST (jamais vues à l'entraînement) dans l'ordre
chronologique. Chaque transaction est convertie dans le format PROPRIÉTAIRE de son
opérateur (Vodacom, Airtel, Orange, Visa) puis envoyée à l'integration-layer, comme le
feraient les systèmes des opérateurs.

Le statut réel (succès / solde insuffisant) n'est PAS envoyé : comme en production, la
transaction est scorée avant exécution.

Usage :
    python scripts/simulate_transactions.py --rate 20            # 20 transactions / seconde
    python scripts/simulate_transactions.py --rate 50 --n 2000
    python scripts/simulate_transactions.py --only-fraud-episodes --rate 5   # démo jury
    python scripts/simulate_transactions.py --duplicate-rate 0.05  # 5 % de renvois (idempotence)
    python scripts/simulate_transactions.py --url https://localhost:8443/ingest --mtls-dir infra/nginx/certs

Chaque message est SIGNÉ (HMAC, en-têtes X-Timestamp / X-Signature) comme le ferait le
système d'un opérateur ; clés et secrets : OPERATOR_API_KEYS / OPERATOR_HMAC_SECRETS, lus dans .env.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "integration-layer"))

from app.adapters.airtel_adapter import AirtelAdapter  # noqa: E402
from app.adapters.orange_adapter import OrangeAdapter  # noqa: E402
from app.adapters.visa_virtual_adapter import VisaVirtualAdapter  # noqa: E402
from app.adapters.vodacom_adapter import VodacomAdapter  # noqa: E402
from shared.schemas.unified_transaction import canonical_id  # noqa: E402
from shared.security.request_signing import parse_secrets, signed_headers  # noqa: E402
from shared.env_file import load_env  # noqa: E402

load_env()   # mêmes clés et secrets que la plateforme (.env)

ADAPTERS = {"vodacom": VodacomAdapter(), "airtel": AirtelAdapter(), "orange": OrangeAdapter(),
            "visa": VisaVirtualAdapter()}
_DEV_KEYS = "vodacom:dev-vodacom-key,airtel:dev-airtel-key,orange:dev-orange-key,visa:dev-visa-key"
_DEV_HMAC = "vodacom:dev-vodacom-hmac,airtel:dev-airtel-hmac,orange:dev-orange-hmac,visa:dev-visa-hmac"
KEYS = parse_secrets(os.getenv("OPERATOR_API_KEYS", _DEV_KEYS))
HMAC_SECRETS = parse_secrets(os.getenv("OPERATOR_HMAC_SECRETS", _DEV_HMAC))
COLORS = {"APPROVE": "\033[32m", "VERIFY": "\033[33m", "BLOCK": "\033[31m"}


def load(n: int | None, skip: int, only_fraud: bool) -> list[dict]:
    prep = json.loads((ROOT / "ml/artifacts/preprocessing.json").read_text(encoding="utf-8"))
    tx = pd.read_csv(ROOT / "ml/data/raw/transactions.csv")
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx = tx.sort_values(["timestamp", "transaction_id"], kind="mergesort")
    tx = tx[tx["timestamp"] >= pd.Timestamp(prep["split_bounds"]["test_from"])].iloc[skip:]
    if only_fraud:
        # épisodes de fraude + un peu de trafic normal autour, pour une démo lisible
        users = set(tx.loc[tx["is_fraud"] == 1, "user_id"].head(40))
        tx = tx[tx["user_id"].isin(users)]
    if n:
        tx = tx.head(n)
    tx = tx.astype(object).where(tx.notna(), None)
    rows = []
    for r in tx.to_dict("records"):
        for k in ("user_id", "wallet_id", "card_id", "counterparty_id", "merchant_id", "agent_id", "device_id"):
            r[k] = canonical_id(r[k])
        r["timestamp"] = pd.Timestamp(r["timestamp"]).to_pydatetime()
        r["status"] = None  # inconnu au moment du scoring
        rows.append(r)
    return rows


def provider_of(r: dict) -> str:
    return "visa" if r["channel"] == "VISA_VIRTUAL" else r["operator"].lower()


def signed_request(prov: str, r: dict) -> tuple[str, bytes, dict]:
    """Chemin, corps et en-têtes d'un message opérateur signé (le chemin signé est /v1/...,
    sans le préfixe /ingest de la passerelle)."""
    path = f"/v1/transactions/{prov}"
    body = json.dumps(ADAPTERS[prov].from_unified(r), default=str).encode()
    headers = {"X-API-Key": KEYS[prov], "Content-Type": "application/json",
               **signed_headers(HMAC_SECRETS[prov], "POST", path, body)}
    return path, body, headers


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8002")
    p.add_argument("--rate", type=float, default=20, help="transactions par seconde")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--only-fraud-episodes", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--duplicate-rate", type=float, default=0.0,
                   help="part des transactions renvoyées une 2e fois (coupure réseau simulée)")
    p.add_argument("--mtls-dir", default=None,
                   help="dossier des certificats clients (<opérateur>.crt/.key + operators-ca.crt)")
    a = p.parse_args()

    rows = load(a.n, a.skip, a.only_fraud_episodes)
    print(f"{len(rows)} transactions à rejouer à {a.rate}/s vers {a.url}\n")
    stats, lat, tp, fn, fp = Counter(), [], 0, 0, 0
    sem = asyncio.Semaphore(64)
    if a.mtls_dir:   # un client HTTP par opérateur : chacun présente SON certificat
        d = Path(a.mtls_dir)
        clients = {prov: httpx.AsyncClient(base_url=a.url, timeout=5, verify=str(d / "operators-ca.crt"),
                                           cert=(str(d / f"{prov}.crt"), str(d / f"{prov}.key")))
                   for prov in ADAPTERS}
    else:
        shared_client = httpx.AsyncClient(base_url=a.url, timeout=5)
        clients = {prov: shared_client for prov in ADAPTERS}
    rng = random.Random(0)
    try:
        async def send(r):
            nonlocal tp, fn, fp
            prov = provider_of(r)
            path, body, headers = signed_request(prov, r)
            async with sem:
                try:
                    resp = await clients[prov].post(path, content=body, headers=headers)
                    try:
                        d = resp.json()
                    except ValueError:
                        d = {"detail": resp.text[:200]}
                except httpx.HTTPError as e:
                    stats["erreur"] += 1
                    print(f"erreur réseau ({type(e).__name__}) sur {r['transaction_id']} : {e or 'délai dépassé'}")
                    return
            if resp.status_code != 200:
                stats["erreur"] += 1
                print(f"erreur HTTP {resp.status_code} sur {r['transaction_id']} : {d}")
                return
            if d.get("idempotent_replay"):
                # 2e exemplaire d'une transaction (renvoi ou original arrivé après son renvoi) :
                # décision d'origine renvoyée, profil du client non recompté
                stats["renvoi_deja_score"] += 1
                return
            action = d["action"]
            stats[action] += 1
            lat.append(d.get("end_to_end_ms", 0))
            caught = action != "APPROVE"
            if r["is_fraud"] == 1:
                tp += caught
                fn += not caught
            else:
                fp += action == "BLOCK"
            if not a.quiet and (action != "APPROVE" or r["is_fraud"] == 1):
                truth = f"FRAUDE {r['fraud_type']}" if r["is_fraud"] == 1 else "légitime"
                prob = d.get("fraud_probability")
                p_txt = f"{prob:.3f}" if prob is not None else "  n/a"
                print(f"{COLORS.get(action, '')}{action:7s}\033[0m p={p_txt} "
                      f"{d['risk_level']:8s} {prov:7s} {r['tx_type']:16s} {r['amount_usd']:>9.2f} USD "
                      f"| réalité : {truth} | {d['end_to_end_ms']:.0f} ms")

        t0 = time.perf_counter()
        tasks = []
        for i, r in enumerate(rows):
            target = t0 + i / a.rate
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send(r)))
            if a.duplicate_rate and rng.random() < a.duplicate_rate:
                tasks.append(asyncio.create_task(send(r)))
        await asyncio.gather(*tasks)
        dt = time.perf_counter() - t0
    finally:
        for c in set(clients.values()):
            await c.aclose()

    lat_s = sorted(lat)
    pct = lambda q: lat_s[min(len(lat_s) - 1, int(q * len(lat_s)))] if lat_s else 0  # noqa: E731
    print(f"\n{len(rows)} transactions en {dt:.1f} s ({len(rows) / dt:.1f}/s) : {dict(stats)}")
    print(f"latence bout-en-bout : p50={pct(0.5):.0f} ms  p95={pct(0.95):.0f} ms  p99={pct(0.99):.0f} ms")
    if tp + fn:
        print(f"fraudes interceptées (VERIFY ou BLOCK) : {tp}/{tp + fn} ; clients honnêtes bloqués : {fp}")


if __name__ == "__main__":
    asyncio.run(main())
