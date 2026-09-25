"""
backoffice-api : API des analystes, superviseurs et administrateurs.

    /auth      connexion JWT, profil
    /users     gestion des comptes (ADMIN)
    /transactions  transactions scorées, détail + explication SHAP
    /cases     dossiers d'alerte, verdicts (-> étiquettes pour le réentraînement)
    /reports   KPIs temps réel, séries temporelles, ventilations, performance observée
    /audit     journal inaltérable + vérification de la chaîne de hachage
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import text

from app import events
from app.config import settings
from app.db import SessionLocal, engine
from app.routers import audit, auth, cases, reports, transactions


@asynccontextmanager
async def lifespan(_app: FastAPI):
    events.publisher = events.Publisher(settings.KAFKA_BOOTSTRAP_SERVERS, settings.KAFKA_ENABLED)
    with SessionLocal() as db:
        auth.ensure_bootstrap_admin(db, settings.BOOTSTRAP_ADMIN_EMAIL, settings.BOOTSTRAP_ADMIN_PASSWORD)
    yield
    events.publisher.flush()


app = FastAPI(title="Back-office — détection de fraude RDC", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app)

for r in (auth.router, transactions.router, cases.router, reports.router, audit.router):
    app.include_router(r)


@app.get("/health")
def health():
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "service": settings.SERVICE_NAME, "database": db_ok}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
