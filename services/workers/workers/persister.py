"""prediction-logger : enregistre chaque transaction scorée (variables, score, décision,
explication, version du modèle) dans TimescaleDB. C'est la mémoire du système : tableau de
bord, dossiers, et jeu d'apprentissage futur une fois les étiquettes des analystes posées."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.dialects.postgresql import insert

from shared.database.models import ScoredTransaction
from shared.database.session import make_session_factory
from shared.kafka_config.topics import TOPIC_TRANSACTIONS_SCORED
from workers.common import run

engine, _ = make_session_factory()


def to_row(e: dict) -> dict:
    t, d = e["transaction"], e["decision"]
    return {
        "transaction_id": t["transaction_id"], "tx_time": datetime.fromisoformat(t["timestamp"]),
        "scored_at": datetime.fromisoformat(e["scored_at"]), "user_id": t["user_id"],
        "channel": t["channel"], "operator": t.get("operator"), "tx_type": t["tx_type"],
        "access_channel": t["access_channel"], "amount": t["amount"], "currency": t["currency"],
        "amount_usd": t["amount_usd"], "province": t["location_province"], "device_id": t["device_id"],
        "counterparty_id": t.get("counterparty_id"), "agent_id": t.get("agent_id"),
        "merchant_id": t.get("merchant_id"), "fraud_probability": d["fraud_probability"],
        "risk_level": d["risk_level"], "action": d["action"], "reason": d["reason"][:255],
        "rules_triggered": d["rules_triggered"], "model_version": d["model_version"],
        "latency_ms": d["latency_ms"], "degraded": d["degraded"],
        "explanation": e["explanation"], "features": e["features"],
    }


def handle(batch: list[dict]) -> None:
    rows = {r["transaction_id"]: r for r in map(to_row, batch)}  # dédoublonnage dans le lot
    stmt = insert(ScoredTransaction).values(list(rows.values())).on_conflict_do_nothing()
    with engine.begin() as c:
        c.execute(stmt)


if __name__ == "__main__":
    run("persister", [TOPIC_TRANSACTIONS_SCORED], handle, batch_size=1000)
