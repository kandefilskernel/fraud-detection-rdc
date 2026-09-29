"""Assistant d'enquête : cas similaires, procédures et note d'instruction pour une alerte.

Avis préparatoire à la demande de l'analyste : n'influence jamais la décision temps réel
et n'écrit rien dans le dossier. Chaque consultation est tracée dans le journal d'audit
avec les sources montrées, pour pouvoir relire plus tard sur quoi l'analyste s'est appuyé."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assistant import synthesis
from app.assistant.knowledge import get_knowledge
from app.assistant.retrieval import build_context
from app.config import settings
from app.db import get_db
from app.events import Publisher, get_publisher
from app.security import get_current_user
from shared.database.models import BackofficeUser, ScoredTransaction

router = APIRouter(tags=["assistant d'enquête"])

DISCLAIMER = ("Avis préparatoire généré automatiquement à partir de cas passés et des procédures internes. "
              "Il ne constitue pas une décision : l'analyste vérifie et décide.")
_cache: dict[tuple, tuple[float, dict]] = {}
_lock = threading.Lock()


@router.get("/assistant/status")
def status(_=Depends(get_current_user)):
    kn = get_knowledge()
    return {"llm_enabled": synthesis.llm_configured(),
            "model": settings.ASSISTANT_MODEL if synthesis.llm_configured() else None,
            "archive_cases": len(kn.cases.cases), "procedure_sections": len(kn.procedures)}


def _public_case(c: dict) -> dict:
    keep = ("ref", "id", "source", "date", "hour", "tx_type", "amount_usd", "province", "outcome", "typology",
            "similarity", "facts", "mitigating", "report_delay_days")
    out = {k: c.get(k) for k in keep if c.get(k) is not None}
    if c.get("source") == "VERDICT_PLATEFORME":      # l'analyste peut ouvrir la transaction
        out["transaction_id"] = c["source_id"]
    return out


@router.post("/transactions/{transaction_id}/assistant")
def investigate(transaction_id: str, mode: str = Query("auto", pattern="^(auto|extractif)$"),
                refresh: bool = False, db: Session = Depends(get_db),
                user: BackofficeUser = Depends(get_current_user), pub: Publisher = Depends(get_publisher)):
    tx = db.scalar(select(ScoredTransaction).where(ScoredTransaction.transaction_id == transaction_id))
    if tx is None:
        raise HTTPException(404, "transaction introuvable")
    key = (transaction_id, tx.label, mode)
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
    if hit and not refresh and now - hit[0] < settings.ASSISTANT_CACHE_S:
        return {**hit[1], "cached": True}

    ctx = build_context(db, tx, get_knowledge())
    note = synthesis.synthesize(ctx, mode)
    result = {
        "transaction_id": transaction_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": note.mode, "model": note.model, "fallback_used": note.fallback_used,
        "truncated": note.truncated, "notice": note.notice, "disclaimer": DISCLAIMER,
        "synthesis": note.markdown,
        "typology_hypotheses": ctx["typology_hypotheses"],
        "precedents": ctx["precedents"],
        "facts": ctx["facts"], "mitigating": ctx["mitigating"],
        "entity_links": ctx["entity_links"],
        "similar_cases": [_public_case(c) for c in ctx["cases"]],
        "procedures": ctx["procedures"],
        "cached": False,
    }
    with _lock:
        _cache[key] = (now, result)
        for k in [k for k, (t, _) in _cache.items() if now - t > settings.ASSISTANT_CACHE_S]:
            _cache.pop(k, None)
    pub.audit(user.email, "ASSISTANT_ENQUETE", "transaction", transaction_id, {
        "mode": note.mode, "model": note.model, "fallback": note.fallback_used, "usage": note.usage,
        "hypothese": ctx["typology_hypotheses"][0]["typology"] if ctx["typology_hypotheses"] else None,
        "sources": [c["ref"] + ":" + (c.get("id") or c.get("source_id", "")) for c in ctx["cases"]]
                   + [p["ref"] + ":" + p["id"] for p in ctx["procedures"]],
    })
    return result
