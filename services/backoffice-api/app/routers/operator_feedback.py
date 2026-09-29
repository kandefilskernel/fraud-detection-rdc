"""Retour des opérateurs (API INTERNE, appelée par l'integration-layer uniquement).

Fraude confirmée, plainte client, contestation carte (chargeback) ou opération légitime :
chaque retour devient l'étiquette de la transaction (scored_transactions.label), exactement
comme le verdict d'un analyste ou une plainte saisie au back-office. Le réentraînement
champion / challenger (ml/retraining/retrain.py) l'utilise donc sans modification.

Règles :
    - un opérateur ne peut étiqueter que SES transactions (plateforme mutualisée) ;
    - une fraude signalée l'emporte : elle ferme le dossier en FRAUDE_CONFIRMEE ;
    - « légitime » ne remplace jamais une étiquette de fraude déjà posée (analyste, plainte) :
      conflit renvoyé (409), arbitrage humain ;
    - un retour répété ne change rien (idempotent) ;
    - chaque changement est tracé dans le journal d'audit chaîné.

Jamais exposée par nginx ni par le frontend ; protégée par une clé de service (X-Internal-Key).
"""
from __future__ import annotations

import hmac
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.events import Publisher, get_publisher
from shared.database.models import Case, ScoredTransaction

router = APIRouter(prefix="/internal", tags=["interne"], include_in_schema=False)

FRAUD_OUTCOMES = {"FRAUD_CONFIRMED", "CUSTOMER_COMPLAINT", "CHARGEBACK"}
OUTCOME_FR = {"FRAUD_CONFIRMED": "fraude confirmée", "CUSTOMER_COMPLAINT": "plainte du client",
              "CHARGEBACK": "contestation carte (chargeback)", "LEGITIMATE": "opération légitime"}
FINAL = {"FRAUDE_CONFIRMEE", "FAUX_POSITIF"}


class OperatorFeedbackIn(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=64)
    provider: Literal["vodacom", "airtel", "orange", "visa"]
    outcome: Literal["FRAUD_CONFIRMED", "CUSTOMER_COMPLAINT", "CHARGEBACK", "LEGITIMATE"]
    reported_at: datetime | None = None
    reference: str | None = Field(None, max_length=64)
    comment: str | None = Field(None, max_length=1000)


def require_internal_key(x_internal_key: str = Header("", alias="X-Internal-Key")) -> None:
    if not settings.INTERNAL_API_KEY or not hmac.compare_digest(x_internal_key, settings.INTERNAL_API_KEY):
        raise HTTPException(401, "clé de service interne invalide")


def _belongs_to(tx: ScoredTransaction, provider: str) -> bool:
    if provider == "visa":
        return tx.channel == "VISA_VIRTUAL"
    return tx.channel == "MOBILE_MONEY" and (tx.operator or "").lower() == provider


@router.post("/operator-feedback", dependencies=[Depends(require_internal_key)])
def operator_feedback(body: OperatorFeedbackIn, db: Session = Depends(get_db),
                      pub: Publisher = Depends(get_publisher)):
    tx = db.scalar(select(ScoredTransaction).where(ScoredTransaction.transaction_id == body.transaction_id))
    if tx is None:
        raise HTTPException(404, "transaction inconnue ou pas encore enregistrée : renvoyer le retour plus tard")
    if not _belongs_to(tx, body.provider):
        raise HTTPException(403, "cette transaction n'appartient pas à cet opérateur")

    actor = f"operateur:{body.provider}"
    note = f"Retour opérateur {body.provider} : {OUTCOME_FR[body.outcome]}"
    if body.reference:
        note += f" (réf. {body.reference})"
    if body.comment:
        note += f" — {body.comment}"
    now = datetime.now(timezone.utc)
    previous_label = tx.label
    case = db.scalar(select(Case).where(Case.transaction_id == tx.transaction_id))

    if body.outcome in FRAUD_OUTCOMES:
        label = 1
        changed = tx.label != 1 or case is None or case.status != "FRAUDE_CONFIRMEE"
        if changed:
            if case is None:
                case = Case(transaction_id=tx.transaction_id, tx_time=tx.tx_time, user_id=tx.user_id,
                            channel=tx.channel, tx_type=tx.tx_type, amount_usd=tx.amount_usd,
                            fraud_probability=tx.fraud_probability, risk_level=tx.risk_level, action=tx.action)
                db.add(case)
            case.status, case.resolution_note, case.resolved_at = "FRAUDE_CONFIRMEE", note, now
            tx.label, tx.labeled_at = 1, now
    else:
        label = 0
        if tx.label == 1:
            raise HTTPException(409, "conflit : transaction déjà étiquetée frauduleuse (analyste ou plainte) ; "
                                     "arbitrage manuel requis")
        changed = tx.label != 0
        if changed:
            tx.label, tx.labeled_at = 0, now
            if case is not None and case.status not in FINAL:
                case.status, case.resolution_note, case.resolved_at = "FAUX_POSITIF", note, now
    db.commit()

    if changed and label == 1:
        pub.fraud_confirmed(tx, actor)   # profil de réputation du scoring
    if changed:
        pub.audit(actor, "RETOUR_OPERATEUR", "transaction", tx.transaction_id,
                  {"outcome": body.outcome, "label": label, "previous_label": previous_label,
                   "reference": body.reference, "case_id": case.id if case else None,
                   "decision_modele": tx.action})
    return {"transaction_id": tx.transaction_id, "label": label, "changed": changed,
            "case_id": case.id if case else None, "case_status": case.status if case else None}
