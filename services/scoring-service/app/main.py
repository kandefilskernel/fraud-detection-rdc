"""
scoring-service : décision de fraude synchrone, en temps réel.

    POST /v1/score   transaction (schéma pivot) -> probabilité, niveau de risque, action,
                     explication. Le profil comportemental est lu ET mis à jour dans Redis.
    GET  /health     état du service (modèle chargé, Redis joignable)
    GET  /metrics    métriques Prometheus (latence, décisions, erreurs)
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
import redis
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app.config import settings
from app.events import EventPublisher
from app.schemas import ScoreResponse
from ml.serving.decision import DecisionPolicy
from ml.serving.feature_store import PREFIX, RedisFeatureStore
from ml.serving.scorer import HybridScorer
from shared.schemas.unified_transaction import UnifiedTransaction

LATENCY = Histogram("scoring_latency_seconds", "Latence de /v1/score",
                    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1))
DECISIONS = Counter("scoring_decisions_total", "Décisions rendues", ["action", "channel"])
RISK = Counter("scoring_risk_levels_total", "Niveaux de risque", ["risk_level"])
DEGRADED = Counter("scoring_degraded_total", "Scores rendus en mode dégradé")
ERRORS = Counter("scoring_errors_total", "Erreurs de scoring", ["kind"])
PROBA = Histogram("scoring_fraud_probability", "Distribution des probabilités (dérive)",
                  buckets=(0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99))
KAFKA_FAIL = Gauge("scoring_kafka_publish_failures", "Échecs cumulés de publication Kafka")

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    r = redis.Redis.from_url(settings.REDIS_URL)
    meta_raw = r.get(f"{PREFIX}:meta")
    if meta_raw is None:
        raise RuntimeError("Feature store vide : lancer python -m ml.serving.seed_feature_store")
    import json
    meta = json.loads(meta_raw)
    scorer = HybridScorer(settings.ARTIFACTS_DIR)
    state.update(
        redis=r,
        scorer=scorer,
        store=RedisFeatureStore(r, pd.Timestamp(meta["period_start"]).timestamp(), seq_len=scorer.seq_len),
        policy=DecisionPolicy(threshold=scorer.threshold),
        events=EventPublisher(settings.KAFKA_BOOTSTRAP_SERVERS, settings.KAFKA_ENABLED),
        api_keys={k.strip() for k in settings.SCORING_API_KEYS.split(",") if k.strip()},
    )
    yield
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

    def decide(feats, x, history):
        res = scorer.score(x, history, explain=True, top_k=settings.EXPLAIN_TOP_K)
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
        },
        "degraded": res.degraded,
        "degraded_branches": res.degraded_branches,
        "model_version": scorer.model_version,
        "latency_ms": round(latency_ms, 2),
        "features": {k: round(float(v), 5) for k, v in out["features"].items()},
    }


@app.post("/v1/score", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
async def score(tx: UnifiedTransaction):
    with LATENCY.time():
        try:
            result = await run_in_threadpool(_score, tx)
        except TimeoutError:
            ERRORS.labels("lock_timeout").inc()
            raise HTTPException(status_code=503, detail="profil client occupé, réessayer")
        except redis.RedisError:
            ERRORS.labels("redis").inc()
            raise HTTPException(status_code=503, detail="feature store indisponible")
    DECISIONS.labels(result["action"], tx.channel.value).inc()
    RISK.labels(result["risk_level"]).inc()
    PROBA.observe(result["fraud_probability"])
    if result["degraded"]:
        DEGRADED.inc()

    event = {
        "event_id": uuid.uuid4().hex,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "transaction": tx.model_dump(mode="json"),
        "decision": {k: result[k] for k in ("action", "risk_level", "reason", "rules_triggered",
                                            "fraud_probability", "is_fraud_predicted", "threshold",
                                            "model_version", "latency_ms", "degraded")},
        "explanation": result["explanation"],
        "features": result["features"],
    }
    state["events"].publish(event)
    KAFKA_FAIL.set(state["events"].failures)
    return result


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
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
