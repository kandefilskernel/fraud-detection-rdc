"""Événements émis par le back-office : actions à auditer, verdicts des analystes."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from shared.kafka_config.topics import TOPIC_ANALYST_FEEDBACK, TOPIC_AUDIT_LOGS, TOPIC_FRAUD_CONFIRMED

log = logging.getLogger("backoffice.events")


class Publisher:
    def __init__(self, bootstrap: str, enabled: bool):
        self.producer = None
        if enabled:
            try:
                from confluent_kafka import Producer
                self.producer = Producer({"bootstrap.servers": bootstrap, "acks": "all",
                                          "enable.idempotence": True})
            except Exception as e:  # noqa: BLE001
                log.warning("Kafka indisponible : %s", e)

    def _send(self, topic: str, key: str, payload: dict) -> None:
        if self.producer is None:
            log.info("[%s] %s", topic, payload)
            return
        self.producer.produce(topic, json.dumps(payload, default=str).encode(), key.encode())
        self.producer.poll(0)

    def audit(self, actor: str, action: str, entity: str, entity_id: str | None, details: dict) -> None:
        self._send(TOPIC_AUDIT_LOGS, entity, {
            "event_id": uuid.uuid4().hex, "ts": datetime.now(timezone.utc).isoformat(), "actor": actor, "action": action,
            "entity": entity, "entity_id": entity_id, "details": details})

    def feedback(self, transaction_id: str, label: int, analyst: str, case_id: int) -> None:
        self._send(TOPIC_ANALYST_FEEDBACK, transaction_id, {
            "event_id": uuid.uuid4().hex, "ts": datetime.now(timezone.utc).isoformat(), "transaction_id": transaction_id,
            "label": label, "analyst": analyst, "case_id": case_id})

    def fraud_confirmed(self, tx, source: str) -> None:
        """Fraude confirmée : ses entités (appareil, portefeuille, agent, marchand, compte)
        alimentent le profil de réputation du scoring (topic fraud.confirmed)."""
        self._send(TOPIC_FRAUD_CONFIRMED, tx.transaction_id, {
            "event_id": uuid.uuid4().hex, "ts": datetime.now(timezone.utc).isoformat(), "source": source,
            "transaction_id": tx.transaction_id, "tx_time": tx.tx_time.isoformat(), "tx_type": tx.tx_type,
            "user_id": tx.user_id, "device_id": tx.device_id, "counterparty_id": tx.counterparty_id,
            "agent_id": tx.agent_id, "merchant_id": tx.merchant_id})

    def flush(self) -> None:
        if self.producer is not None:
            self.producer.flush(5)


publisher: Publisher | None = None


def get_publisher() -> Publisher:
    return publisher
