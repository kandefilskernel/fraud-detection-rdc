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
"""
from __future__ import annotations

import argparse
import asyncio
import json
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

ADAPTERS = {"vodacom": VodacomAdapter(), "airtel": AirtelAdapter(), "orange": OrangeAdapter(),
            "visa": VisaVirtualAdapter()}
KEYS = {"vodacom": "dev-vodacom-key", "airtel": "dev-airtel-key", "orange": "dev-orange-key",
        "visa": "dev-visa-key"}
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


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8002")
    p.add_argument("--rate", type=float, default=20, help="transactions par seconde")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--only-fraud-episodes", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    rows = load(a.n, a.skip, a.only_fraud_episodes)
    print(f"{len(rows)} transactions à rejouer à {a.rate}/s vers {a.url}\n")
    stats, lat, tp, fn, fp = Counter(), [], 0, 0, 0
    sem = asyncio.Semaphore(64)
    async with httpx.AsyncClient(base_url=a.url, timeout=5) as client:
        async def send(r):
            nonlocal tp, fn, fp
            prov = provider_of(r)
            async with sem:
                try:
                    resp = await client.post(f"/v1/transactions/{prov}", json=ADAPTERS[prov].from_unified(r),
                                             headers={"X-API-Key": KEYS[prov]})
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
                print(f"{COLORS.get(action, '')}{action:7s}\033[0m p={d['fraud_probability']:.3f} "
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
        await asyncio.gather(*tasks)
        dt = time.perf_counter() - t0

    lat_s = sorted(lat)
    pct = lambda q: lat_s[min(len(lat_s) - 1, int(q * len(lat_s)))] if lat_s else 0  # noqa: E731
    print(f"\n{len(rows)} transactions en {dt:.1f} s ({len(rows) / dt:.1f}/s) : {dict(stats)}")
    print(f"latence bout-en-bout : p50={pct(0.5):.0f} ms  p95={pct(0.95):.0f} ms  p99={pct(0.99):.0f} ms")
    if tp + fn:
        print(f"fraudes interceptées (VERIFY ou BLOCK) : {tp}/{tp + fn} ; clients honnêtes bloqués : {fp}")


if __name__ == "__main__":
    asyncio.run(main())
