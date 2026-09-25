"""
integration-layer : point d'entrée des opérateurs.

    POST /v1/transactions/{provider}   provider ∈ vodacom | airtel | orange | visa
        1. authentifie l'opérateur (X-API-Key propre à chaque opérateur) ;
        2. convertit le message propriétaire vers le schéma pivot (Adapter) ;
        3. demande la décision au scoring-service (appel synchrone) ;
        4. renvoie la décision à l'opérateur, qui exécute, fait vérifier ou refuse.

Si le scoring est indisponible, la politique de repli est configurable (FAIL_OPEN) :
laisser passer (continuité de service) ou exiger une vérification (prudence).
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.adapters.airtel_adapter import AirtelAdapter
from app.adapters.base_adapter import AdapterError
from app.adapters.orange_adapter import OrangeAdapter
from app.adapters.vodacom_adapter import VodacomAdapter
from app.adapters.visa_virtual_adapter import VisaVirtualAdapter


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    SCORING_URL: str = "http://localhost:8001"
    SCORING_API_KEY: str = "dev-scoring-key"
    SCORING_TIMEOUT_S: float = 0.8
    USD_CDF_RATE: float = 2850.0
    # clés d'API opérateur -> fournisseur, ex. "vodacom:dev-vodacom-key,airtel:dev-airtel-key"
    OPERATOR_API_KEYS: str = ("vodacom:dev-vodacom-key,airtel:dev-airtel-key,"
                              "orange:dev-orange-key,visa:dev-visa-key")
    FAIL_OPEN: bool = False


settings = Settings()
ADAPTERS = {a.provider: a for a in (VodacomAdapter(settings.USD_CDF_RATE), AirtelAdapter(settings.USD_CDF_RATE),
                                    OrangeAdapter(settings.USD_CDF_RATE), VisaVirtualAdapter(settings.USD_CDF_RATE))}
KEYS = dict(reversed(pair.split(":", 1)) for pair in settings.OPERATOR_API_KEYS.split(",") if ":" in pair)

REQUESTS = Counter("integration_requests_total", "Messages opérateurs reçus", ["provider", "outcome"])
LATENCY = Histogram("integration_latency_seconds", "Latence bout-en-bout (adaptateur + scoring)",
                    ["provider"], buckets=(0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1))

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["http"] = httpx.AsyncClient(base_url=settings.SCORING_URL, timeout=settings.SCORING_TIMEOUT_S,
                                      limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))
    yield
    await state["http"].aclose()


app = FastAPI(title="Integration layer — opérateurs Mobile Money & Visa", version="1.0.0", lifespan=lifespan)


@app.post("/v1/transactions/{provider}")
async def ingest(provider: str, request: Request, x_api_key: str = Header(..., alias="X-API-Key")):
    t0 = time.perf_counter()
    adapter = ADAPTERS.get(provider)
    if adapter is None:
        raise HTTPException(404, f"opérateur inconnu : {provider}")
    if KEYS.get(x_api_key) != provider:
        REQUESTS.labels(provider, "unauthorized").inc()
        raise HTTPException(401, "clé d'API opérateur invalide")
    try:
        tx = adapter.to_unified(await request.json())
    except (AdapterError, ValidationError, ValueError) as e:
        REQUESTS.labels(provider, "invalid").inc()
        raise HTTPException(422, f"message {provider} invalide : {e}")

    try:
        r = await state["http"].post("/v1/score", content=tx.model_dump_json(),
                                     headers={"X-API-Key": settings.SCORING_API_KEY,
                                              "Content-Type": "application/json"})
        r.raise_for_status()
        result = r.json()
        REQUESTS.labels(provider, result["action"]).inc()
    except (httpx.HTTPError, KeyError):
        REQUESTS.labels(provider, "scoring_unavailable").inc()
        result = {"transaction_id": tx.transaction_id,
                  "action": "APPROVE" if settings.FAIL_OPEN else "VERIFY",
                  "risk_level": "INCONNU", "reason": "scoring indisponible (politique de repli)",
                  "fraud_probability": None}
    LATENCY.labels(provider).observe(time.perf_counter() - t0)
    result["provider"] = provider
    result["end_to_end_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


@app.get("/health")
async def health():
    try:
        r = await state["http"].get("/health")
        scoring = r.json().get("status")
    except httpx.HTTPError:
        scoring = "unreachable"
    return {"status": "ok", "service": "integration-layer", "scoring": scoring,
            "providers": sorted(ADAPTERS)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
