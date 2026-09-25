"""auditor : unique écrivain du journal d'audit chaîné (hachage SHA-256 de l'entrée
précédente). Consomme les actions du back-office, les ouvertures de dossiers et les
verdicts des analystes. Un seul consommateur et une partition par topic : ordre total."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text

from shared.database.audit_chain import GENESIS, entry_hash
from shared.database.models import AuditLog
from shared.database.session import make_session_factory
from shared.kafka_config.topics import TOPIC_ANALYST_FEEDBACK, TOPIC_AUDIT_LOGS
from workers.common import run

engine, Session = make_session_factory()


def normalize(e: dict) -> dict:
    if e["topic"] == TOPIC_ANALYST_FEEDBACK:
        return {"event_id": e.get("event_id") or f"fb-{e['case_id']}-{e['label']}", "ts": e["ts"],
                "actor": e["analyst"], "action": "VERDICT_ANALYSTE", "entity": "transaction",
                "entity_id": e["transaction_id"], "details": {"label": e["label"], "case_id": e["case_id"]}}
    return {"event_id": e.get("event_id") or uuid.uuid4().hex, **{k: e.get(k) for k in
            ("ts", "actor", "action", "entity", "entity_id")}, "details": e.get("details") or {}}


def handle(batch: list[dict]) -> None:
    with Session() as db:
        # verrou applicatif : même si deux auditeurs tournaient, la chaîne reste linéaire
        db.execute(text("SELECT pg_advisory_xact_lock(424242)"))
        last = db.scalar(select(AuditLog.hash).order_by(AuditLog.id.desc()).limit(1)) or GENESIS
        for raw in batch:
            e = normalize(raw)
            details = {**e["details"], "event_id": e["event_id"]}
            seen = db.scalar(select(AuditLog.id).where(AuditLog.details["event_id"].astext == e["event_id"]))
            if seen:
                continue
            ts = datetime.fromisoformat(e["ts"]) if e.get("ts") else datetime.now(timezone.utc)
            ts = ts.astimezone(timezone.utc)  # PostgreSQL relit en UTC : même valeur hachée
            h = entry_hash(last, ts, e["actor"], e["action"], e["entity"], e["entity_id"], details)
            db.add(AuditLog(ts=ts, actor=e["actor"], action=e["action"], entity=e["entity"],
                            entity_id=e["entity_id"], details=details, prev_hash=last, hash=h))
            db.flush()
            last = h
        db.commit()


if __name__ == "__main__":
    run("auditor", [TOPIC_AUDIT_LOGS, TOPIC_ANALYST_FEEDBACK], handle, batch_size=200)
