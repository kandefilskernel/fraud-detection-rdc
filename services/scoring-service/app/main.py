"""
scoring-service : décision de fraude synchrone, en temps réel.

    POST /v1/score   transaction (schéma pivot) -> probabilité, niveau de risque, action,
                     explication. Le profil comportemental est lu ET mis à jour dans Redis.
    POST /v1/scam-reports  SMS d'arnaque signalé par un client (texte déjà masqué par la couche
                     d'intégration) -> classification NLP ; si c'est une arnaque, les numéros
                     désignés sont marqués (règle BENEFICIAIRE_SIGNALE_PAR_SMS).
    GET  /health     état du service (modèle chargé, Redis joignable)
    GET  /metrics    métriques Prometheus (latence, décisions, erreurs)
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import redis
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from prometheus_client import Counter, Gauge, Histogram

from shared.utils.metrics import metrics_response
from app.config import settings
from app.events import EventPublisher
from app.schemas import ScoreResponse
from ml.serving.decision import DecisionPolicy
from ml.serving.feature_store import PREFIX, RedisFeatureStore
from ml.serving.idempotency import IdempotencyConflict, IdempotencyInProgress, IdempotencyStore, fingerprint
from pydantic import BaseModel, Field
from shared.kafka_config.topics import TOPIC_FRAUD_CONFIRMED
from ml.serving.scorer import HybridScorer
from ml.nlp.scam_sms import ScamSmsClassifier
from shared.schemas.unified_transaction import UnifiedTransaction
from shared.logging.logger_config import configure_logging

configure_logging("scoring-service")

LATENCY = Histogram("scoring_latency_seconds", "Latence de /v1/score",
                    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1))
DECISIONS = Counter("scoring_decisions_total", "Décisions rendues", ["action", "channel"])
RISK = Counter("scoring_risk_levels_total", "Niveaux de risque", ["risk_level"])
DEGRADED = Counter("scoring_degraded_total", "Scores rendus en mode dégradé")
ERRORS = Counter("scoring_errors_total", "Erreurs de scoring", ["kind"])
PROBA = Histogram("scoring_fraud_probability", "Distribution des probabilités (dérive)",
                  buckets=(0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99))
KAFKA_FAIL = Gauge("scoring_kafka_publish_failures", "Échecs cumulés de publication Kafka",
                   multiprocess_mode="livesum")

state: dict = {}
log = logging.getLogger("scoring")
MODEL_VERSION = Gauge("scoring_model_info", "Version du modèle servie (1 = active)", ["version"],
                      multiprocess_mode="livemax")
RELOADS = Counter("scoring_model_reloads_total", "Rechargements à chaud du modèle")
REPLAYS = Counter("scoring_idempotent_replays_total", "Transactions renvoyées par un opérateur (déjà scorées)")
REPUTATION_UPDATES = Counter("scoring_reputation_reports_total", "Fraudes confirmées intégrées au profil de réputation")
SMS_REPORTS = Counter("scoring_scam_sms_reports_total", "SMS signalés par les clients", ["category", "outcome"])


def _watch_model(stop: threading.Event, interval: float = 30.0) -> None:
    """Recharge le modèle à chaud quand le réentraînement promeut un nouveau champion
    (metadata.json est écrit en dernier par ml/retraining/retrain.py)."""
    meta_path = Path(settings.ARTIFACTS_DIR) / "metadata.json"
    while not stop.wait(interval):
        try:
            version = json.loads(meta_path.read_text(encoding="utf-8")).get("trained_at")
            current = state["scorer"].model_version
            if version and version != current:
                new = HybridScorer(settings.ARTIFACTS_DIR)      # chargé à côté, puis échange atomique
                state["scorer"], state["policy"] = new, DecisionPolicy(threshold=new.threshold)
                MODEL_VERSION.labels(current).set(0)
                MODEL_VERSION.labels(new.model_version).set(1)
                RELOADS.inc()
                log.warning("nouveau modèle chargé à chaud : %s -> %s", current, new.model_version)
        except Exception:  # noqa: BLE001 — fichiers en cours d'écriture : on réessaie plus tard
            log.exception("rechargement du modèle impossible pour l'instant")


def report_event(d: dict) -> dict:
    """Événement « fraude confirmée » -> format de RedisFeatureStore.apply_report. L'horodatage
    est celui de la transaction frauduleuse, en heure locale naïve (même convention que ts)."""
    tx_time = datetime.fromisoformat(str(d["tx_time"])).replace(tzinfo=None)
    return {"transaction_id": d["transaction_id"], "ts": (tx_time - datetime(1970, 1, 1)).total_seconds(),
            "tx_type": d.get("tx_type"), "user_id": d.get("user_id"), "device_id": d.get("device_id"),
            "counterparty_id": d.get("counterparty_id"), "agent_id": d.get("agent_id"),
            "merchant_id": d.get("merchant_id")}


def _consume_confirmed_frauds(stop: threading.Event) -> None:
    """Profil de réputation : chaque fraude confirmée (analyste, plainte, retour d'opérateur,
    publiée par le back-office) marque ses entités dans Redis pour les transactions suivantes."""
    try:
        from confluent_kafka import Consumer
        consumer = Consumer({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
                             "group.id": "scoring-reputation", "auto.offset.reset": "earliest"})
        consumer.subscribe([TOPIC_FRAUD_CONFIRMED])
    except Exception:  # noqa: BLE001
        log.exception("consommateur de réputation indisponible")
        return
    while not stop.is_set():
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        try:
            state["store"].apply_report(report_event(json.loads(msg.value())))
            REPUTATION_UPDATES.inc()
        except Exception:  # noqa: BLE001 — un message invalide ne doit pas arrêter le consommateur
            log.exception("signalement de fraude invalide ignoré")
    consumer.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    r = redis.Redis.from_url(settings.REDIS_URL)
    meta_raw = r.get(f"{PREFIX}:meta")
    if meta_raw is None:
        raise RuntimeError("Feature store vide : lancer python -m ml.serving.seed_feature_store")
    meta = json.loads(meta_raw)
    scorer = HybridScorer(settings.ARTIFACTS_DIR)
    state.update(
        redis=r,
        scorer=scorer,
        store=RedisFeatureStore(r, pd.Timestamp(meta["period_start"]).timestamp(), seq_len=scorer.seq_len),
        policy=DecisionPolicy(threshold=scorer.threshold),
        events=EventPublisher(settings.KAFKA_BOOTSTRAP_SERVERS, settings.KAFKA_ENABLED),
        api_keys={k.strip() for k in settings.SCORING_API_KEYS.split(",") if k.strip()},
        idempotency=IdempotencyStore(r, ttl_s=settings.IDEMPOTENCY_TTL_S),
    )
    MODEL_VERSION.labels(scorer.model_version).set(1)
    try:   # classifieur des SMS signalés (facultatif : sans lui, /v1/scam-reports répond 503)
        state["sms"] = ScamSmsClassifier.load(Path(settings.ARTIFACTS_DIR) / "nlp")
    except FileNotFoundError:
        state["sms"] = None
        log.warning("classifieur de SMS absent (ml/artifacts/nlp) : signalements SMS désactivés")
    stop = threading.Event()
    threading.Thread(target=_watch_model, args=(stop,), daemon=True).start()
    if settings.KAFKA_ENABLED:
        threading.Thread(target=_consume_confirmed_frauds, args=(stop,), daemon=True).start()
    yield
    stop.set()
    state["events"].flush()


app = FastAPI(title="Scoring de fraude — Mobile Money RDC & Visa virtuelle", version="1.0.0",
              lifespan=lifespan)


def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key not in state["api_keys"]:
        raise HTTPException(status_code=401, detail="clé d'API invalide")


def _score(tx: UnifiedTransaction) -> dict:
    scorer: HybridScorer = state["scorer"]
    t0 = time.perf_counter()
    # En temps réel le statut est inconnu (transaction pas encore exécutée) : le profil
    # enregistre la transaction comme réussie, ou échouée si elle est bloquée.
    tx_in = tx.to_feature_input(tx.status)
    if tx.tx_type.value == "P2P_SEND" and tx.counterparty_id:
        # signalements SMS du bénéficiaire (règle BENEFICIAIRE_SIGNALE_PAR_SMS, hors modèle)
        tx_in["cp_scam_reports_30d"] = state["store"].scam_report_count(tx_in["counterparty_id"], tx_in["ts"])

    def decide(feats, x, history):
        res = scorer.score(x, history, explain="auto", top_k=settings.EXPLAIN_TOP_K)
        dec = state["policy"].decide(res.probability, tx.amount_usd, tx.channel.value, feats, tx_in)
        return (res, dec), ("FAILED" if dec["action"] == "BLOCK" else "SUCCESS")

    out = state["store"].compute(tx_in, scorer.scale, decide)
    res, decision = out["decided"]
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "transaction_id": tx.transaction_id,
        "fraud_probability": round(res.probability, 6),
        "is_fraud_predicted": res.probability >= scorer.threshold,
        "threshold": round(scorer.threshold, 6),
        **decision,
        "known_user": out["known_user"],
        "explanation": {
            "top_features": res.top_features,
            "branch_scores": res.branch_scores,
            "branch_contributions": res.branch_contributions,
            "attention_on_history": res.attention,
            "attribution": res.attribution,
            "anomaly": res.anomaly,
        },
        "degraded": res.degraded,
        "degraded_branches": res.degraded_branches,
        "model_version": scorer.model_version,
        "latency_ms": round(latency_ms, 2),
        "features": {k: round(float(v), 5) for k, v in out["features"].items()},
    }


@app.post("/v1/score", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
async def score(tx: UnifiedTransaction):
    # Idempotence : un renvoi de l'opérateur reçoit la décision d'origine sans recompter la
    # transaction dans le profil du client (ni la republier vers la base et les alertes).
    idem: IdempotencyStore = state["idempotency"]
    idem_key = idem.key(tx.channel.value, tx.operator, tx.transaction_id)
    fp = fingerprint(tx.model_dump(mode="json", exclude={"status"}))
    try:
        previous = await run_in_threadpool(idem.begin, idem_key, fp)
    except IdempotencyConflict as e:
        ERRORS.labels("idempotency_conflict").inc()
        raise HTTPException(status_code=409, detail=str(e))
    except IdempotencyInProgress as e:
        raise HTTPException(status_code=409, detail=str(e))
    except redis.RedisError:
        ERRORS.labels("redis").inc()
        raise HTTPException(status_code=503, detail="feature store indisponible")
    if previous is not None:
        REPLAYS.inc()
        return {**previous, "idempotent_replay": True}

    with LATENCY.time():
        try:
            result = await run_in_threadpool(_score, tx)
        except TimeoutError:
            idem.abort(idem_key)
            ERRORS.labels("lock_timeout").inc()
            raise HTTPException(status_code=503, detail="profil client occupé, réessayer")
        except redis.RedisError:
            idem.abort(idem_key)
            ERRORS.labels("redis").inc()
            raise HTTPException(status_code=503, detail="feature store indisponible")
        except Exception:  # noqa: BLE001 — jamais de 500 muet : l'appelant applique sa politique de repli
            idem.abort(idem_key)
            ERRORS.labels("internal").inc()
            log.exception("échec du scoring", extra={"transaction_id": tx.transaction_id})
            raise HTTPException(status_code=500, detail="erreur interne du scoring")
    idem.complete(idem_key, fp, result)
    DECISIONS.labels(result["action"], tx.channel.value).inc()
    RISK.labels(result["risk_level"]).inc()
    PROBA.observe(result["fraud_probability"])
    if result["degraded"]:
        DEGRADED.inc()
    if result["action"] != "APPROVE":
        log.warning("alerte %s", result["action"], extra={
            "transaction_id": tx.transaction_id, "user_id": tx.user_id, "action": result["action"],
            "latency_ms": result["latency_ms"]})

    event = {
        "event_id": uuid.uuid4().hex,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "transaction": tx.model_dump(mode="json"),
        "decision": {k: result[k] for k in ("action", "risk_level", "reason", "rules_triggered",
                                            "fraud_probability", "is_fraud_predicted", "threshold",
                                            "model_version", "latency_ms", "degraded",
                                            "verification_method")},
        "explanation": result["explanation"],
        "features": result["features"],
    }
    state["events"].publish(event)
    KAFKA_FAIL.set(state["events"].failures)
    return result


class FraudConfirmed(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=64)
    tx_time: datetime
    tx_type: str
    user_id: str
    device_id: str | None = None
    counterparty_id: str | None = None
    agent_id: str | None = None
    merchant_id: str | None = None


@app.post("/v1/reputation/report", dependencies=[Depends(require_api_key)])
def reputation_report(body: FraudConfirmed):
    """Voie directe (tests, reprise) ; en fonctionnement normal : topic Kafka fraud.confirmed."""
    n = state["store"].apply_report(report_event(body.model_dump()))
    REPUTATION_UPDATES.inc()
    return {"transaction_id": body.transaction_id, "entities_updated": n}


class ScamReport(BaseModel):
    report_id: str = Field(..., min_length=1, max_length=64)
    received_at: datetime
    masked_text: str = Field(..., min_length=1, max_length=1000)    # numéros déjà remplacés par <NUMERO>
    sender_is_number: bool | None = None
    numbers: list[str] = Field(default_factory=list, max_length=10)  # numéros désignés (pseudonymisés si besoin)


@app.post("/v1/scam-reports", dependencies=[Depends(require_api_key)])
def scam_report(body: ScamReport):
    clf: ScamSmsClassifier | None = state.get("sms")
    if clf is None:
        raise HTTPException(503, "classifieur de SMS non disponible")
    res = clf.classify(body.masked_text, body.sender_is_number)
    flagged: list[str] = []
    # un message envoyé sous le nom d'un opérateur ne sert jamais à marquer un numéro
    if res["is_scam"] and body.sender_is_number is not False:
        ts = (body.received_at.replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds()
        for n in dict.fromkeys(body.numbers):
            state["store"].add_scam_report(n, body.report_id, ts)
            flagged.append(n)
    SMS_REPORTS.labels(res["category"], "numeros_marques" if flagged else "non_marque").inc()
    return {"report_id": body.report_id, **res, "flagged_numbers": flagged, "model_version": clf.version}


@app.get("/health")
def health():
    try:
        state["redis"].ping()
        redis_ok = True
    except redis.RedisError:
        redis_ok = False
    return {"status": "ok" if redis_ok else "degraded", "service": settings.SERVICE_NAME,
            "model_version": state["scorer"].model_version, "redis": redis_ok,
            "kafka": state["events"].producer is not None}


@app.get("/metrics")
def metrics():
    return metrics_response()
