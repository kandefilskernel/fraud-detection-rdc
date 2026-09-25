"""Outils communs aux consommateurs Kafka : création des topics, boucle de consommation
« au moins une fois » (commit des offsets APRÈS l'écriture en base), métriques."""
from __future__ import annotations

import json
import logging
import os
import signal
import time

from confluent_kafka import Consumer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
from prometheus_client import Counter, Histogram, start_http_server

from shared.kafka_config.topics import (TOPIC_ANALYST_FEEDBACK, TOPIC_AUDIT_LOGS, TOPIC_FRAUD_ALERTS,
                                        TOPIC_TRANSACTIONS_SCORED)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
# audit : 1 partition = ordre total, indispensable au chaînage par hachage
TOPIC_PARTITIONS = {TOPIC_TRANSACTIONS_SCORED: 3, TOPIC_FRAUD_ALERTS: 3,
                    TOPIC_AUDIT_LOGS: 1, TOPIC_ANALYST_FEEDBACK: 1}

PROCESSED = Counter("worker_messages_total", "Messages traités", ["worker", "outcome"])
BATCH_SECONDS = Histogram("worker_batch_seconds", "Durée de traitement d'un lot", ["worker"])


def ensure_topics() -> None:
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    for _ in range(30):
        try:
            existing = set(admin.list_topics(timeout=5).topics)
            break
        except KafkaException:
            time.sleep(2)
    else:
        raise RuntimeError("Kafka injoignable")
    new = [NewTopic(t, num_partitions=p, replication_factor=1)
           for t, p in TOPIC_PARTITIONS.items() if t not in existing]
    if new:
        for t, f in admin.create_topics(new).items():
            try:
                f.result()
            except KafkaException as e:
                if "TOPIC_ALREADY_EXISTS" not in str(e):
                    raise


def run(worker: str, topics: list[str], handle_batch, batch_size: int = 500, metrics_port: int = 9100):
    """handle_batch(list[dict]) écrit en base ; en cas d'exception le lot est rejoué."""
    log = logging.getLogger(worker)
    ensure_topics()
    start_http_server(metrics_port)
    consumer = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": f"fraud-{worker}",
                         "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    consumer.subscribe(topics)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    log.info("démarré : topics=%s", topics)
    try:
        while not stop["flag"]:
            msgs = consumer.consume(num_messages=batch_size, timeout=1.0)
            if not msgs:
                continue
            payloads = []
            for m in msgs:
                if m.error():
                    log.warning("erreur Kafka : %s", m.error())
                    continue
                try:
                    payloads.append({"topic": m.topic(), **json.loads(m.value())})
                except json.JSONDecodeError:
                    PROCESSED.labels(worker, "invalid").inc()
            if not payloads:
                continue
            with BATCH_SECONDS.labels(worker).time():
                while True:
                    try:
                        handle_batch(payloads)
                        break
                    except Exception:  # noqa: BLE001 — base indisponible : on réessaie
                        log.exception("échec du lot, nouvel essai dans 2 s")
                        PROCESSED.labels(worker, "retry").inc()
                        time.sleep(2)
                        if stop["flag"]:
                            return
            consumer.commit(asynchronous=False)
            PROCESSED.labels(worker, "ok").inc(len(payloads))
    finally:
        consumer.close()
