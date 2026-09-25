from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.security import require_role
from shared.database.audit_chain import verify_chain
from shared.database.models import AuditLog

router = APIRouter(prefix="/audit", tags=["audit & conformité"])


@router.get("")
def list_audit(db: Session = Depends(get_db), _=Depends(require_role("SUPERVISEUR")),
               actor: str | None = None, action: str | None = None,
               page: int = Query(1, ge=1), size: int = Query(50, le=200)):
    q = select(AuditLog)
    if actor:
        q = q.where(AuditLog.actor == actor)
    if action:
        q = q.where(AuditLog.action == action)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(AuditLog.id.desc()).offset((page - 1) * size).limit(size)).all()
    return {"total": total, "items": [{c.name: getattr(r, c.name) for c in AuditLog.__table__.columns}
                                      for r in rows]}


@router.get("/verify")
def verify(db: Session = Depends(get_db), _=Depends(require_role("SUPERVISEUR"))):
    """Recalcule toute la chaîne de hachage : toute altération est détectée."""
    return verify_chain(db.scalars(select(AuditLog).order_by(AuditLog.id)).yield_per(5000))
