"""Indicateurs pour le tableau de bord. Les fenêtres « temps réel » utilisent scored_at
(horloge réelle du scoring) ; les analyses métier utilisent tx_time (heure de la transaction)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_db
from app.security import get_current_user, require_role

router = APIRouter(prefix="/reports", tags=["rapports"])


def _rows(db: Session, sql: str, **params) -> list[dict]:
    return [dict(r._mapping) for r in db.execute(text(sql), params)]


@router.get("/kpis")
def kpis(minutes: int = Query(60, ge=1, le=60 * 24 * 30), db: Session = Depends(get_db),
         _=Depends(get_current_user)):
    row = _rows(db, """
        SELECT count(*)                                              AS volume,
               count(*) FILTER (WHERE action = 'APPROVE')            AS approved,
               count(*) FILTER (WHERE action = 'VERIFY')             AS verified,
               count(*) FILTER (WHERE action = 'BLOCK')              AS blocked,
               coalesce(sum(amount_usd), 0)                          AS amount_usd,
               coalesce(sum(amount_usd) FILTER (WHERE action = 'BLOCK'), 0)  AS amount_blocked_usd,
               coalesce(sum(amount_usd) FILTER (WHERE action = 'VERIFY'), 0) AS amount_verified_usd,
               coalesce(avg(latency_ms), 0)                          AS latency_avg_ms,
               coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), 0) AS latency_p95_ms,
               count(*) FILTER (WHERE degraded)                      AS degraded
        FROM scored_transactions WHERE scored_at >= now() - make_interval(mins => :m)
    """, m=minutes)[0]
    cases = _rows(db, "SELECT status, count(*) AS n FROM cases GROUP BY status")
    row["alert_rate"] = (row["verified"] + row["blocked"]) / row["volume"] if row["volume"] else 0.0
    row["cases"] = {c["status"]: c["n"] for c in cases}
    row["window_minutes"] = minutes
    return row


@router.get("/timeseries")
def timeseries(minutes: int = Query(60, ge=5, le=60 * 24 * 7), bucket_seconds: int = Query(60, ge=5, le=3600),
               db: Session = Depends(get_db), _=Depends(get_current_user)):
    return _rows(db, """
        SELECT time_bucket(make_interval(secs => :b), scored_at) AS bucket,
               count(*) AS volume,
               count(*) FILTER (WHERE action = 'VERIFY') AS verified,
               count(*) FILTER (WHERE action = 'BLOCK')  AS blocked,
               avg(fraud_probability) AS avg_probability,
               avg(latency_ms) AS latency_avg_ms
        FROM scored_transactions WHERE scored_at >= now() - make_interval(mins => :m)
        GROUP BY bucket ORDER BY bucket
    """, b=bucket_seconds, m=minutes)


@router.get("/breakdown")
def breakdown(dimension: str = Query("channel", pattern="^(channel|operator|tx_type|province|access_channel)$"),
              minutes: int = Query(60 * 24, ge=1, le=60 * 24 * 30), db: Session = Depends(get_db),
              _=Depends(get_current_user)):
    # `dimension` est validé par le motif ci-dessus : insertion sûre dans la requête
    return _rows(db, f"""
        SELECT coalesce({dimension}, 'N/A') AS key, count(*) AS volume,
               count(*) FILTER (WHERE action <> 'APPROVE') AS alerts,
               count(*) FILTER (WHERE action = 'BLOCK') AS blocked,
               avg(fraud_probability) AS avg_probability,
               coalesce(sum(amount_usd) FILTER (WHERE action = 'BLOCK'), 0) AS amount_blocked_usd
        FROM scored_transactions WHERE scored_at >= now() - make_interval(mins => :m)
        GROUP BY 1 ORDER BY volume DESC
    """, m=minutes)


@router.get("/model-performance")
def model_performance(db: Session = Depends(get_db), _=Depends(require_role("SUPERVISEUR"))):
    """Précision observée grâce aux verdicts des analystes (dossiers clos)."""
    rows = _rows(db, """
        SELECT action, risk_level,
               count(*) FILTER (WHERE label = 1) AS confirmed_fraud,
               count(*) FILTER (WHERE label = 0) AS false_positive
        FROM scored_transactions WHERE label IS NOT NULL
        GROUP BY action, risk_level ORDER BY action, risk_level
    """)
    tp = sum(r["confirmed_fraud"] for r in rows)
    fp = sum(r["false_positive"] for r in rows)
    return {"labeled": tp + fp, "observed_precision": tp / (tp + fp) if tp + fp else None, "detail": rows}


@router.get("/top-risky-users")
def top_risky_users(minutes: int = Query(60 * 24, ge=1), limit: int = Query(10, le=100),
                    db: Session = Depends(get_db), _=Depends(get_current_user)):
    return _rows(db, """
        SELECT user_id, count(*) AS alerts, max(fraud_probability) AS max_probability,
               sum(amount_usd) AS amount_usd
        FROM scored_transactions
        WHERE action <> 'APPROVE' AND scored_at >= now() - make_interval(mins => :m)
        GROUP BY user_id ORDER BY alerts DESC, max_probability DESC LIMIT :l
    """, m=minutes, l=limit)
