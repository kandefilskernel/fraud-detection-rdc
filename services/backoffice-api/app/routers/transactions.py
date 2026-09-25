from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.security import get_current_user
from shared.database.models import ScoredTransaction

router = APIRouter(prefix="/transactions", tags=["transactions"])


class TxSummary(BaseModel):
    transaction_id: str
    tx_time: datetime
    scored_at: datetime
    user_id: str
    channel: str
    operator: str | None
    tx_type: str
    access_channel: str
    amount_usd: float
    province: str
    fraud_probability: float
    risk_level: str
    action: str
    reason: str
    latency_ms: float
    label: int | None

    model_config = {"from_attributes": True}


class TxDetail(TxSummary):
    amount: float
    currency: str
    device_id: str
    counterparty_id: str | None
    agent_id: str | None
    merchant_id: str | None
    rules_triggered: list
    model_version: str
    degraded: bool
    explanation: dict
    features: dict


@router.get("")
def list_transactions(
    db: Session = Depends(get_db), _=Depends(get_current_user),
    channel: str | None = None, action: str | None = None, risk_level: str | None = None,
    operator: str | None = None, user_id: str | None = None, min_probability: float | None = None,
    labeled: bool | None = None,
    page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=500),
):
    q = select(ScoredTransaction)
    filters = []
    if channel:
        filters.append(ScoredTransaction.channel == channel)
    if action:
        filters.append(ScoredTransaction.action == action)
    if risk_level:
        filters.append(ScoredTransaction.risk_level == risk_level)
    if operator:
        filters.append(ScoredTransaction.operator == operator)
    if user_id:
        filters.append(ScoredTransaction.user_id == user_id)
    if min_probability is not None:
        filters.append(ScoredTransaction.fraud_probability >= min_probability)
    if labeled is not None:
        filters.append(ScoredTransaction.label.is_not(None) if labeled else ScoredTransaction.label.is_(None))
    q = q.where(*filters)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(ScoredTransaction.scored_at.desc()).offset((page - 1) * size).limit(size)).all()
    return {"total": total, "page": page, "size": size,
            "items": [TxSummary.model_validate(r).model_dump() for r in rows]}


@router.get("/{transaction_id}", response_model=TxDetail)
def get_transaction(transaction_id: str, db: Session = Depends(get_db), _=Depends(get_current_user)):
    tx = db.scalar(select(ScoredTransaction).where(ScoredTransaction.transaction_id == transaction_id))
    if tx is None:
        raise HTTPException(404, "transaction introuvable")
    return tx


@router.get("/by-user/{user_id}/history")
def user_history(user_id: str, db: Session = Depends(get_db), _=Depends(get_current_user),
                 limit: int = Query(30, le=200)):
    rows = db.scalars(select(ScoredTransaction).where(ScoredTransaction.user_id == user_id)
                      .order_by(ScoredTransaction.tx_time.desc()).limit(limit)).all()
    return [TxSummary.model_validate(r).model_dump() for r in rows]
