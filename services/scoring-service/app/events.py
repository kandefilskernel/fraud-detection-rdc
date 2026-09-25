"""Publication Kafka APRÈS la décision : jamais sur le chemin critique de la réponse.
Si le broker est indisponible, la décision est quand même rendue (et un compteur monte)."""
from __future__ import annotations

import json
import logging

from shared.kafka_config.topics import TOPIC_FRAUD_ALERTS, TOPIC_TRANSACTIONS_SCORED

log = logging.getLogger("scoring.events")


class EventPublisher:
    def __init__(self, bootstrap: str, enabled: bool = True):
        self.producer = None
        self.failures = 0
        if not enabled:
            return
        try:
            from confluent_kafka import Producer
            self.producer = Producer({
                "bootstrap.servers": bootstrap,
                "linger.ms": 5,                 # regroupe les messages : débit élevé
                "acks": "1",
                "message.timeout.ms": 10000,
                "queue.buffering.max.messages": 200000,
            })
        except Exception as e:  # noqa: BLE001
            log.warning("Kafka indisponible, événements désactivés : %s", e)

    def _cb(self, err, _msg):
        if err is not None:
            self.failures += 1

    def publish(self, event: dict) -> None:
        if self.producer is None:
            return
        try:
            key = event["transaction"]["user_id"].encode()
            payload = json.dumps(event, default=str).encode()
            self.producer.produce(TOPIC_TRANSACTIONS_SCORED, payload, key, on_delivery=self._cb)
            if event["decision"]["action"] != "APPROVE":
                self.producer.produce(TOPIC_FRAUD_ALERTS, payload, key, on_delivery=self._cb)
            self.producer.poll(0)
        except BufferError:
            self.failures += 1

    def flush(self, timeout: float = 5.0) -> None:
        if self.producer is not None:
            self.producer.flush(timeout)
