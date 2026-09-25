"""alerter : pour chaque alerte (VERIFY / BLOCK), ouvre un dossier et notifie.

    - dossier (table cases), prioritaire selon le niveau de risque ;
    - SMS au client pour une VÉRIFICATION (confirmation de l'opération) ou un BLOCAGE ;
    - e-mail à l'équipe fraude pour les alertes CRITIQUES ;
    - webhook vers l'opérateur si une URL est configurée.

En local, les e-mails arrivent dans Mailpit (http://localhost:8025) et les SMS sont
journalisés ; en production, SMS_GATEWAY_URL pointe vers la passerelle de l'opérateur.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage

import httpx
from confluent_kafka import Producer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from shared.database.models import Case, Notification
from shared.database.session import make_session_factory
from shared.kafka_config.topics import TOPIC_AUDIT_LOGS, TOPIC_FRAUD_ALERTS
from workers.common import BOOTSTRAP, run

log = logging.getLogger("alerter")
engine, Session = make_session_factory()

SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "1025"))
FRAUD_TEAM_EMAIL = os.getenv("FRAUD_TEAM_EMAIL", "equipe-fraude@fraud-rdc.local")
SMS_GATEWAY_URL = os.getenv("SMS_GATEWAY_URL", "")          # vide = SMS simulés (journalisés)
OPERATOR_WEBHOOKS = json.loads(os.getenv("OPERATOR_WEBHOOKS", "{}"))  # {"VODACOM": "https://..."}
producer = Producer({"bootstrap.servers": BOOTSTRAP})
http = httpx.Client(timeout=3)


def sms_text(e: dict) -> str | None:
    t, d = e["transaction"], e["decision"]
    amount = f"{t['amount']:,.0f} {t['currency']}".replace(",", " ")
    if d["action"] == "VERIFY":
        return (f"Operation de {amount} en attente. Si c'est bien vous, confirmez avec votre PIN "
                f"sur le menu *1#. Sinon, ne faites rien et appelez le service client.")
    if d["action"] == "BLOCK":
        return (f"Operation de {amount} bloquee par securite. Si vous en etes l'auteur, "
                f"contactez le service client. Ne communiquez jamais votre PIN.")
    return None


def send_sms(wallet: str | None, text: str) -> tuple[str, str]:
    if not wallet:
        return "ECHEC", "pas de numéro de portefeuille"
    if not SMS_GATEWAY_URL:
        log.info("SMS simulé -> %s : %s", wallet, text)
        return "ENVOYE", "simulé (aucune passerelle configurée)"
    try:
        http.post(SMS_GATEWAY_URL, json={"to": wallet, "message": text}).raise_for_status()
        return "ENVOYE", "passerelle"
    except httpx.HTTPError as err:
        return "ECHEC", str(err)[:500]


def send_email(case_id: int, e: dict) -> tuple[str, str]:
    t, d = e["transaction"], e["decision"]
    top = ", ".join(f["feature"] for f in e["explanation"].get("top_features", [])[:4]) or "n/d"
    msg = EmailMessage()
    msg["Subject"] = f"[{d['risk_level']}] Dossier #{case_id} — {t['tx_type']} {t['amount_usd']:.2f} USD ({d['action']})"
    msg["From"] = "alertes@fraud-rdc.local"
    msg["To"] = FRAUD_TEAM_EMAIL
    msg.set_content(
        f"Transaction {t['transaction_id']} du client {t['user_id']} ({t['channel']}, {t.get('operator')})\n"
        f"Montant : {t['amount']} {t['currency']} ({t['amount_usd']:.2f} USD) — {t['location_province']}\n"
        f"Probabilité de fraude : {d['fraud_probability']:.3f} — décision : {d['action']} ({d['reason']})\n"
        f"Règles déclenchées : {', '.join(d['rules_triggered']) or 'aucune'}\n"
        f"Principaux facteurs : {top}\n\nTraiter le dossier dans le tableau de bord.")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=5) as s:
            s.send_message(msg)
        return "ENVOYE", FRAUD_TEAM_EMAIL
    except OSError as err:
        return "ECHEC", str(err)[:500]


def handle(batch: list[dict]) -> None:
    with Session() as db:
        for e in batch:
            t, d = e["transaction"], e["decision"]
            case_id = db.scalar(insert(Case).values(
                transaction_id=t["transaction_id"], tx_time=datetime.fromisoformat(t["timestamp"]),
                user_id=t["user_id"], channel=t["channel"], tx_type=t["tx_type"], amount_usd=t["amount_usd"],
                fraud_probability=d["fraud_probability"], risk_level=d["risk_level"], action=d["action"],
            ).on_conflict_do_nothing(index_elements=["transaction_id"]).returning(Case.id))
            if case_id is None:          # message rejoué : dossier déjà ouvert
                continue
            notes = []
            text = sms_text(e)
            if text:
                status, detail = send_sms(t.get("wallet_id"), text)
                notes.append(Notification(case_id=case_id, channel="SMS", recipient=t.get("wallet_id") or "?",
                                          status=status, detail=detail))
            if d["risk_level"] == "CRITIQUE":
                status, detail = send_email(case_id, e)
                notes.append(Notification(case_id=case_id, channel="EMAIL", recipient=FRAUD_TEAM_EMAIL,
                                          status=status, detail=detail))
            hook = OPERATOR_WEBHOOKS.get(t.get("operator") or "")
            if hook:
                try:
                    http.post(hook, json={"transaction_id": t["transaction_id"], "action": d["action"],
                                          "case_id": case_id}).raise_for_status()
                    notes.append(Notification(case_id=case_id, channel="WEBHOOK", recipient=hook, status="ENVOYE"))
                except httpx.HTTPError as err:
                    notes.append(Notification(case_id=case_id, channel="WEBHOOK", recipient=hook,
                                              status="ECHEC", detail=str(err)[:500]))
            db.add_all(notes)
            producer.produce(TOPIC_AUDIT_LOGS, json.dumps({
                "event_id": uuid.uuid4().hex, "ts": datetime.now(timezone.utc).isoformat(), "actor": "system:alerter",
                "action": "DOSSIER_OUVERT", "entity": "case", "entity_id": str(case_id),
                "details": {"transaction_id": t["transaction_id"], "risk_level": d["risk_level"],
                            "action": d["action"], "notifications": [n.channel for n in notes]},
            }).encode(), b"case")
        db.commit()
    producer.flush(5)


if __name__ == "__main__":
    run("alerter", [TOPIC_FRAUD_ALERTS], handle, batch_size=100)
