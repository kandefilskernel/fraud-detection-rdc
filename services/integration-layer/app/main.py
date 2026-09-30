"""
integration-layer : point d'entrée des opérateurs.

    POST /v1/transactions/{provider}   provider ∈ vodacom | airtel | orange | visa
        1. sécurité : clé d'API, IP autorisée, certificat mTLS, signature HMAC (app/security.py) ;
        2. convertit le message propriétaire vers le schéma pivot (Adapter) ;
        3. pseudonymise les identifiants personnels si la plateforme est mutualisée ;
        4. demande la décision au scoring-service (appel synchrone, idempotent) ;
        5. renvoie la décision à l'opérateur, qui exécute, fait vérifier ou refuse
           (pour Visa : code réponse ISO 8583 pour le processeur de l'émetteur).

    POST /v1/feedback/{provider}
        Retour de l'opérateur sur une transaction : fraude confirmée, plainte client,
        contestation (chargeback) ou opération légitime. Transmis au back-office, il devient
        l'étiquette utilisée par le réentraînement (même sécurité que les transactions).

Si le scoring est indisponible, la politique de repli est configurable (FAIL_OPEN) :
laisser passer (continuité de service) ou exiger une vérification (prudence).

Modes de déploiement (DEPLOYMENT_MODE), selon le cadre légal retenu :
    demo              tous les opérateurs, identifiants en clair (environnement local)
    operator_instance l'opérateur héberge sa propre instance : seul OPERATOR_ID est accepté
    shared_platform   plateforme mutualisée entre opérateurs : pseudonymisation OBLIGATOIRE
                      et signature des messages obligatoire (refus de démarrer sinon)
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.nlp.phone_numbers import extract_numbers, is_phone_number, mask_numbers, normalize
from shared.privacy.pseudonymize import Pseudonymizer
from shared.schemas.unified_transaction import UnifiedTransaction
from shared.security.request_signing import parse_secrets
from shared.utils.metrics import metrics_response
from app.adapters.airtel_adapter import AirtelAdapter
from app.adapters.base_adapter import AdapterError, BaseAdapter
from app.adapters.orange_adapter import OrangeAdapter
from app.adapters.vodacom_adapter import VodacomAdapter
from app.adapters.visa_virtual_adapter import VisaVirtualAdapter
from app.security import OperatorSecurity, SecurityRejection, parse_allowlist, parse_networks
from shared.logging.logger_config import configure_logging

configure_logging("integration-layer")
log = logging.getLogger("integration")

DEPLOYMENT_MODES = ("demo", "operator_instance", "shared_platform")


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
    # --- sécurité des messages (secrets HMAC distincts des clés d'API)
    OPERATOR_HMAC_SECRETS: str = ("vodacom:dev-vodacom-hmac,airtel:dev-airtel-hmac,"
                                  "orange:dev-orange-hmac,visa:dev-visa-hmac")
    REQUIRE_SIGNATURE: bool = True
    SIGNATURE_WINDOW_S: int = 300
    # ex. "vodacom:10.10.0.0/16|10.20.0.5,airtel:10.30.0.0/16" ; vide = pas de restriction
    OPERATOR_IP_ALLOWLIST: str = ""
    TRUSTED_PROXIES: str = "127.0.0.1/32,172.16.0.0/12"      # nginx (réseau Docker)
    REQUIRE_MTLS: bool = False                              # activé par docker-compose.mtls.yml
    # --- cadre légal : mode de déploiement et données personnelles
    DEPLOYMENT_MODE: str = "demo"
    OPERATOR_ID: str = ""
    PSEUDONYMIZE_IDS: bool = False
    PSEUDONYMIZATION_KEY: str = ""
    # --- retour des opérateurs -> back-office (API interne, jamais exposée par nginx)
    BACKOFFICE_URL: str = "http://localhost:8003"
    INTERNAL_API_KEY: str = "dev-internal-key"
    # --- pilote silencieux : opérateurs (ex. "vodacom,airtel" ou "all") pour lesquels la décision
    # est calculée et enregistrée mais JAMAIS appliquée (réponse APPROVE, aucun SMS client)
    SHADOW_MODE_OPERATORS: str = ""


@dataclass
class Runtime:
    settings: Settings
    adapters: dict[str, BaseAdapter]
    allowed: set[str]
    security: OperatorSecurity
    pseudonymizer: Pseudonymizer | None
    shadow: set[str] = field(default_factory=set)


def configure(s: Settings) -> Runtime:
    """Construit la configuration d'exécution et refuse les combinaisons dangereuses."""
    adapters = {a.provider: a for a in (VodacomAdapter(s.USD_CDF_RATE), AirtelAdapter(s.USD_CDF_RATE),
                                        OrangeAdapter(s.USD_CDF_RATE), VisaVirtualAdapter(s.USD_CDF_RATE))}
    if s.DEPLOYMENT_MODE not in DEPLOYMENT_MODES:
        raise RuntimeError(f"DEPLOYMENT_MODE doit valoir {DEPLOYMENT_MODES}")
    allowed = set(adapters)
    if s.DEPLOYMENT_MODE == "operator_instance":
        if s.OPERATOR_ID not in adapters:
            raise RuntimeError(f"operator_instance : OPERATOR_ID doit être l'un de {sorted(adapters)}")
        allowed = {s.OPERATOR_ID}
    if s.DEPLOYMENT_MODE == "shared_platform":
        if not s.PSEUDONYMIZE_IDS:
            raise RuntimeError("plateforme mutualisée : PSEUDONYMIZE_IDS=true obligatoire (protection des données)")
        if not s.REQUIRE_SIGNATURE:
            raise RuntimeError("plateforme mutualisée : REQUIRE_SIGNATURE=true obligatoire")
    pseudonymizer = Pseudonymizer(s.PSEUDONYMIZATION_KEY) if s.PSEUDONYMIZE_IDS else None
    security = OperatorSecurity(
        api_keys=dict(reversed(pair.split(":", 1)) for pair in s.OPERATOR_API_KEYS.split(",") if ":" in pair),
        hmac_secrets=parse_secrets(s.OPERATOR_HMAC_SECRETS), require_signature=s.REQUIRE_SIGNATURE,
        signature_window_s=s.SIGNATURE_WINDOW_S, ip_allowlist=parse_allowlist(s.OPERATOR_IP_ALLOWLIST),
        trusted_proxies=parse_networks(s.TRUSTED_PROXIES), require_mtls=s.REQUIRE_MTLS)
    if s.DEPLOYMENT_MODE == "demo":
        log.warning("mode démo : identifiants en clair, tous les opérateurs acceptés (ne pas utiliser en production)")
    shadow = parse_shadow(s.SHADOW_MODE_OPERATORS, set(adapters))
    if shadow:
        log.warning("PILOTE SILENCIEUX pour %s : décisions notées, jamais appliquées", sorted(shadow))
    return Runtime(s, adapters, allowed, security, pseudonymizer, shadow)


def parse_shadow(spec: str, providers: set[str]) -> set[str]:
    """ "vodacom, airtel" -> {"vodacom", "airtel"} ; "all" -> tous ; nom inconnu -> erreur au démarrage."""
    names = {p.strip().lower() for p in (spec or "").split(",") if p.strip()}
    if "all" in names:
        return set(providers)
    unknown = names - providers
    if unknown:
        raise RuntimeError(f"SHADOW_MODE_OPERATORS : opérateurs inconnus {sorted(unknown)} ({sorted(providers)})")
    return names


def apply_shadow(provider: str, result: dict) -> dict:
    """Pilote silencieux : l'opérateur reçoit toujours APPROVE ; la vraie décision reste dans
    shadow_action (et dans la base, pour mesurer précision et rappel sur les vraies données)."""
    if provider in RT.shadow:
        result["shadow_mode"] = True
        result["shadow_action"] = result.get("action")
        result["action"] = "APPROVE"
    return result


settings = Settings()
RT = configure(settings)

REQUESTS = Counter("integration_requests_total", "Messages opérateurs reçus", ["provider", "outcome"])
FEEDBACK = Counter("integration_feedback_total", "Retours opérateurs reçus", ["provider", "outcome"])
SMS = Counter("integration_scam_sms_total", "SMS d'arnaque signalés par les clients", ["provider", "outcome"])
LATENCY = Histogram("integration_latency_seconds", "Latence bout-en-bout (adaptateur + scoring)",
                    ["provider"], buckets=(0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1))

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["http"] = httpx.AsyncClient(base_url=RT.settings.SCORING_URL, timeout=RT.settings.SCORING_TIMEOUT_S,
                                      limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))
    state["backoffice"] = httpx.AsyncClient(base_url=RT.settings.BACKOFFICE_URL, timeout=5.0)
    yield
    await state["http"].aclose()
    await state["backoffice"].aclose()


app = FastAPI(title="Integration layer — opérateurs Mobile Money & Visa", version="1.1.0", lifespan=lifespan)


async def _authenticated_body(provider: str, request: Request, counter: Counter) -> bytes:
    if provider not in RT.adapters or provider not in RT.allowed:
        counter.labels(provider, "unknown_provider").inc()
        raise HTTPException(404, f"opérateur inconnu : {provider}")
    body = await request.body()
    try:
        RT.security.check(provider, request, body)
    except SecurityRejection as e:
        counter.labels(provider, e.outcome).inc()
        raise HTTPException(e.status, e.detail) from None
    return body


@app.post("/v1/transactions/{provider}")
async def ingest(provider: str, request: Request):
    t0 = time.perf_counter()
    body = await _authenticated_body(provider, request, REQUESTS)
    try:
        tx = RT.adapters[provider].to_unified(json.loads(body))
    except (AdapterError, ValidationError, ValueError) as e:
        REQUESTS.labels(provider, "invalid").inc()
        raise HTTPException(422, f"message {provider} invalide : {e}") from None
    if RT.pseudonymizer is not None:
        tx = UnifiedTransaction(**RT.pseudonymizer.transaction(tx.model_dump()))

    try:
        r = await state["http"].post("/v1/score", content=tx.model_dump_json(),
                                     headers={"X-API-Key": RT.settings.SCORING_API_KEY,
                                              "Content-Type": "application/json"})
        if r.status_code == 409:   # même identifiant, contenu différent, ou envoi simultané
            REQUESTS.labels(provider, "conflict").inc()
            raise HTTPException(409, r.json().get("detail", "transaction en conflit"))
        r.raise_for_status()
        result = r.json()
        REQUESTS.labels(provider, "replay" if result.get("idempotent_replay") else result["action"]).inc()
    except (httpx.HTTPError, KeyError, ValueError):
        REQUESTS.labels(provider, "scoring_unavailable").inc()
        result = {"transaction_id": tx.transaction_id,
                  "action": "APPROVE" if RT.settings.FAIL_OPEN else "VERIFY",
                  "risk_level": "INCONNU", "reason": "scoring indisponible (politique de repli)",
                  "fraud_probability": None}
    LATENCY.labels(provider).observe(time.perf_counter() - t0)
    result["provider"] = provider
    apply_shadow(provider, result)
    if provider == "visa":
        result["iso8583_response_code"] = VisaVirtualAdapter.authorization_response(result["action"])
    result["end_to_end_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


class OperatorFeedback(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=64)
    # FRAUD_CONFIRMED : fraude avérée (enquête de l'opérateur) ; CUSTOMER_COMPLAINT : le client ne
    # reconnaît pas l'opération ; CHARGEBACK : contestation carte ; LEGITIMATE : le client a
    # confirmé être à l'origine de l'opération (ex. après une vérification)
    outcome: Literal["FRAUD_CONFIRMED", "CUSTOMER_COMPLAINT", "CHARGEBACK", "LEGITIMATE"]
    reported_at: datetime | None = None
    reference: str | None = Field(None, max_length=64)     # n° de dossier chez l'opérateur
    comment: str | None = Field(None, max_length=1000)


@app.post("/v1/feedback/{provider}")
async def feedback(provider: str, request: Request):
    body = await _authenticated_body(provider, request, FEEDBACK)
    try:
        fb = OperatorFeedback.model_validate_json(body)
    except ValidationError as e:
        FEEDBACK.labels(provider, "invalid").inc()
        raise HTTPException(422, f"retour {provider} invalide : {e.errors()}") from None
    try:
        r = await state["backoffice"].post("/internal/operator-feedback",
                                           json={**fb.model_dump(mode="json"), "provider": provider},
                                           headers={"X-Internal-Key": RT.settings.INTERNAL_API_KEY})
    except httpx.HTTPError:
        FEEDBACK.labels(provider, "backoffice_unavailable").inc()
        raise HTTPException(503, "back-office indisponible : renvoyer le retour plus tard") from None
    FEEDBACK.labels(provider, fb.outcome if r.status_code < 300 else f"http_{r.status_code}").inc()
    try:
        content = r.json()
    except ValueError:
        content = {"detail": r.text[:200]}
    return JSONResponse(content, status_code=r.status_code)


class ScamSmsReport(BaseModel):
    """SMS transféré par un client au numéro court de signalement de son opérateur."""
    report_id: str = Field(..., min_length=1, max_length=64)
    received_at: datetime
    text: str = Field(..., min_length=1, max_length=1000)
    sender: str | None = Field(None, max_length=40)      # numéro ou nom d'expéditeur du SMS signalé
    reporter: str | None = Field(None, max_length=20)    # numéro du client qui signale


@app.post("/v1/scam-reports/{provider}")
async def scam_report(provider: str, request: Request):
    """Minimisation : les numéros sont extraits ici, pseudonymisés si la plateforme l'exige, et
    remplacés par <NUMERO> dans le texte. Le scoring ne voit jamais un numéro en clair."""
    body = await _authenticated_body(provider, request, SMS)
    try:
        rep = ScamSmsReport.model_validate_json(body)
    except ValidationError as e:
        SMS.labels(provider, "invalid").inc()
        raise HTTPException(422, f"signalement {provider} invalide : {e.errors()}") from None
    numbers = extract_numbers(rep.text)
    sender = normalize(rep.sender)
    if sender and sender not in numbers:
        numbers.insert(0, sender)                     # l'escroc écrit souvent depuis son propre numéro
    reporter = normalize(rep.reporter)
    numbers = [n for n in numbers if n != reporter][:10]   # jamais le client qui signale
    if RT.pseudonymizer is not None:
        numbers = [RT.pseudonymizer(n) for n in numbers]
    payload = {"report_id": f"{provider}:{rep.report_id}", "received_at": rep.received_at.isoformat(),
               "masked_text": mask_numbers(rep.text),
               "sender_is_number": is_phone_number(rep.sender) if rep.sender else None, "numbers": numbers}
    try:
        r = await state["http"].post("/v1/scam-reports", json=payload, headers={"X-API-Key": RT.settings.SCORING_API_KEY})
    except httpx.HTTPError:
        SMS.labels(provider, "scoring_unavailable").inc()
        raise HTTPException(503, "service de scoring indisponible : renvoyer le signalement plus tard") from None
    content = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text[:200]}
    SMS.labels(provider, content.get("category", f"http_{r.status_code}") if r.status_code < 300 else f"http_{r.status_code}").inc()
    return JSONResponse(content, status_code=r.status_code)


@app.get("/health")
async def health():
    try:
        r = await state["http"].get("/health")
        scoring = r.json().get("status")
    except httpx.HTTPError:
        scoring = "unreachable"
    return {"status": "ok", "service": "integration-layer", "scoring": scoring,
            "providers": sorted(RT.allowed), "deployment_mode": RT.settings.DEPLOYMENT_MODE,
            "signature_required": RT.settings.REQUIRE_SIGNATURE, "mtls_required": RT.settings.REQUIRE_MTLS,
            "pseudonymization": RT.pseudonymizer is not None}


@app.get("/metrics")
def metrics():
    return metrics_response()
