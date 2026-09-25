"""Gestion des dossiers : les analystes confirment ou infirment les alertes.
Chaque verdict devient une étiquette (label) qui alimente le réentraînement."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db import get_db
from app.events import Publisher, get_publisher
from app.security import ROLE_RANK, get_current_user
from shared.database.models import BackofficeUser, Case, ScoredTransaction

router = APIRouter(prefix="/cases", tags=["dossiers"])

FINAL = {"FRAUDE_CONFIRMEE": 1, "FAUX_POSITIF": 0}
TRANSITIONS = {"OUVERT": {"EN_COURS", *FINAL}, "EN_COURS": {"OUVERT", *FINAL}}


class CaseOut(BaseModel):
    id: int
    transaction_id: str
    tx_time: datetime
    user_id: str
    channel: str
    tx_type: str
    amount_usd: float
    fraud_probability: float
    risk_level: str
    action: str
    status: str
    assigned_to: int | None
    resolution_note: str | None
    resolved_by: int | None
    resolved_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CaseUpdate(BaseModel):
    status: str | None = Field(None, pattern="^(OUVERT|EN_COURS|FRAUDE_CONFIRMEE|FAUX_POSITIF)$")
    assigned_to: int | None = None
    resolution_note: str | None = Field(None, max_length=2000)


@router.get("")
def list_cases(db: Session = Depends(get_db), _=Depends(get_current_user),
               status: str | None = None, risk_level: str | None = None, assigned_to: int | None = None,
               page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=200)):
    q = select(Case)
    if status:
        q = q.where(Case.status == status)
    if risk_level:
        q = q.where(Case.risk_level == risk_level)
    if assigned_to is not None:
        q = q.where(Case.assigned_to == assigned_to)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    # priorité : risque critique d'abord, puis montant
    order = func.array_position(["CRITIQUE", "ELEVE", "MOYEN", "FAIBLE"], Case.risk_level)
    rows = db.scalars(q.order_by(Case.status != "OUVERT", order, Case.amount_usd.desc())
                      .offset((page - 1) * size).limit(size)).all()
    counts = dict(db.execute(select(Case.status, func.count()).group_by(Case.status)).all())
    return {"total": total, "counts": counts, "items": [CaseOut.model_validate(r).model_dump() for r in rows]}


@router.get("/{case_id}", response_model=CaseOut)
def get_case(case_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    c = db.get(Case, case_id)
    if c is None:
        raise HTTPException(404, "dossier introuvable")
    return c


@router.patch("/{case_id}", response_model=CaseOut)
def update_case(case_id: int, body: CaseUpdate, db: Session = Depends(get_db),
                user: BackofficeUser = Depends(get_current_user), pub: Publisher = Depends(get_publisher)):
    c = db.get(Case, case_id, with_for_update=True)
    if c is None:
        raise HTTPException(404, "dossier introuvable")
    changes: dict = {}

    if body.assigned_to is not None and body.assigned_to != c.assigned_to:
        # un analyste peut se l'attribuer ; réaffecter à autrui demande SUPERVISEUR
        if body.assigned_to != user.id and ROLE_RANK[user.role] < ROLE_RANK["SUPERVISEUR"]:
            raise HTTPException(403, "réaffectation réservée aux superviseurs")
        if db.get(BackofficeUser, body.assigned_to) is None:
            raise HTTPException(422, "utilisateur inconnu")
        changes["assigned_to"] = [c.assigned_to, body.assigned_to]
        c.assigned_to = body.assigned_to

    if body.status and body.status != c.status:
        if body.status not in TRANSITIONS.get(c.status, set()):
            raise HTTPException(409, f"transition {c.status} -> {body.status} interdite")
        if body.status in FINAL and not (body.resolution_note or c.resolution_note):
            raise HTTPException(422, "une note de résolution est obligatoire")
        changes["status"] = [c.status, body.status]
        c.status = body.status
        if body.status in FINAL:
            c.resolved_by, c.resolved_at = user.id, datetime.now(timezone.utc)
            label = FINAL[body.status]
            db.execute(update(ScoredTransaction)
                       .where(ScoredTransaction.transaction_id == c.transaction_id,
                              ScoredTransaction.tx_time == c.tx_time)
                       .values(label=label, labeled_at=c.resolved_at))
            pub.feedback(c.transaction_id, label, user.email, c.id)
        elif c.assigned_to is None:
            c.assigned_to = user.id

    if body.resolution_note is not None:
        c.resolution_note = body.resolution_note
        changes["resolution_note"] = True

    db.commit()
    if changes:
        pub.audit(user.email, "DOSSIER_MODIFIE", "case", str(c.id), changes)
    return c
