"""
Recherche du contexte d'une transaction à instruire (partie « R » du RAG) :

    1. faits saillants et éléments à décharge (variables du modèle traduites en clair) ;
    2. décision du modèle : probabilité, règles, principales contributions SHAP ;
    3. cas passés les plus proches dans l'archive (fraudes confirmées, alertes classées) ;
    4. verdicts déjà rendus sur la plateforme pour des transactions semblables ;
    5. liens de la transaction avec des fraudes confirmées (même appareil, bénéficiaire,
       agent ou marchand) ;
    6. sections de procédures internes pertinentes.

Tout ce qui sort d'ici peut être montré à l'analyste ; ce qui part vers le modèle de langage
est reconstruit par synthesis.render_prompt, SANS identifiant.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.assistant.knowledge import Knowledge
from shared.database.models import ScoredTransaction
from shared.investigation.case_index import CLEARED, CONFIRMED, summarize_neighbors
from shared.investigation.signals import TX_TYPE_LABELS, TYPOLOGY_LABELS, describe, mitigating

K_ARCHIVE = 6
K_PLATFORM = 3
PLATFORM_POOL = 3000          # verdicts récents comparés à la transaction
ENTITY_WINDOW = timedelta(days=30)
# signaux reçus de l'opérateur ou du réseau carte, connus seulement par la règle qu'ils déclenchent
RULE_FACTS = {
    "SIM_RECENTE_OPERATION_SORTANTE": "Carte SIM changée il y a moins de 24 h (signal opérateur)",
    "BENEFICIAIRE_SIGNALE_PAR_SMS": "Bénéficiaire désigné comme escroc dans un SMS signalé par un client (30 derniers jours)",
    "SCORE_RESEAU_CARTE_ELEVE": "Score de risque du réseau Visa élevé",
    "SCORE_RESEAU_CARTE_CRITIQUE": "Score de risque du réseau Visa critique",
}
ENTITIES = (("device_id", "l'appareil"), ("counterparty_id", "le bénéficiaire"),
            ("agent_id", "l'agent"), ("merchant_id", "le marchand"))


def _platform_verdicts(db: Session, tx: ScoredTransaction, kn: Knowledge) -> list[dict]:
    """Transactions déjà tranchées par un analyste, les plus proches de celle-ci."""
    rows = db.execute(
        select(ScoredTransaction.transaction_id, ScoredTransaction.features, ScoredTransaction.label,
               ScoredTransaction.tx_type, ScoredTransaction.amount_usd, ScoredTransaction.tx_time,
               ScoredTransaction.province)
        .where(ScoredTransaction.label.is_not(None), ScoredTransaction.channel == tx.channel,
               ScoredTransaction.transaction_id != tx.transaction_id)
        .order_by(ScoredTransaction.labeled_at.desc()).limit(PLATFORM_POOL)).all()
    rows = [r for r in rows if r.features]
    if not rows:
        return []
    idx = kn.cases
    w = idx.weights
    q = idx.scale_features(tx.features or {}) * w
    m = np.stack([idx.scale_features(r.features) * w for r in rows])
    sims = (m @ q) / np.maximum(np.linalg.norm(m, axis=1) * np.linalg.norm(q), 1e-9)
    out = []
    for i in np.argsort(-sims)[:K_PLATFORM]:
        r = rows[i]
        if sims[i] < 0.5:
            break
        out.append({"source": "VERDICT_PLATEFORME", "source_id": r.transaction_id,
                    "date": r.tx_time.strftime("%Y-%m-%d"), "hour": r.tx_time.hour, "tx_type": r.tx_type,
                    "amount_usd": round(r.amount_usd, 2), "province": r.province,
                    "outcome": CONFIRMED if r.label == 1 else CLEARED, "typology": None,
                    "facts": describe(r.features, r.tx_type, max_facts=5),
                    "similarity": round(float(sims[i]), 3)})
    return out


def _entity_links(db: Session, tx: ScoredTransaction) -> list[dict]:
    """Entités de la transaction déjà vues dans des fraudes confirmées ou des alertes récentes
    (30 jours). Seuls les comptes sortent d'ici, jamais les identifiants."""
    since = tx.tx_time - ENTITY_WINDOW
    out = []
    for col, label in ENTITIES:
        value = getattr(tx, col)
        if not value:
            continue
        confirmed, alerts, users = db.execute(
            select(func.count().filter(ScoredTransaction.label == 1),
                   func.count().filter(ScoredTransaction.action != "APPROVE"),
                   func.count(func.distinct(ScoredTransaction.user_id)).filter(ScoredTransaction.user_id != tx.user_id))
            .where(getattr(ScoredTransaction, col) == value, ScoredTransaction.tx_time >= since,
                   ScoredTransaction.transaction_id != tx.transaction_id)).one()
        if confirmed or alerts or (col != "device_id" and users >= 5):
            out.append({"entity": label, "confirmed_frauds": confirmed, "alerts": alerts, "other_clients": users})
    return out


def _client_history(db: Session, tx: ScoredTransaction) -> dict:
    rows = db.execute(select(ScoredTransaction.action, ScoredTransaction.label)
                      .where(ScoredTransaction.user_id == tx.user_id,
                             ScoredTransaction.transaction_id != tx.transaction_id,
                             ScoredTransaction.tx_time >= tx.tx_time - timedelta(days=7),
                             ScoredTransaction.tx_time <= tx.tx_time)).all()
    return {"n_7d": len(rows), "alerts_7d": sum(r.action != "APPROVE" for r in rows),
            "confirmed_7d": sum(r.label == 1 for r in rows)}


def build_context(db: Session, tx: ScoredTransaction, kn: Knowledge) -> dict:
    feats = tx.features or {}
    facts = [RULE_FACTS[r] for r in (tx.rules_triggered or []) if r in RULE_FACTS] + describe(feats, tx.tx_type)
    archive = kn.cases.search(feats, tx.channel, k=K_ARCHIVE, exclude_ids={tx.transaction_id})
    platform = _platform_verdicts(db, tx, kn)
    precedents = summarize_neighbors(archive)
    typologies = [t["typology"] for t in precedents["typologies"]]

    # alias C1..Cn : l'analyste retrouve chaque cas cité dans la synthèse
    cases = []
    for c in archive + platform:
        cases.append({**c, "ref": f"C{len(cases) + 1}"})

    rules = list(tx.rules_triggered or [])
    query = " ".join([*facts, *rules, *(TYPOLOGY_LABELS.get(t, t) for t in typologies),
                      TX_TYPE_LABELS.get(tx.tx_type, tx.tx_type)])
    procedures = [{"ref": f"P{i + 1}", "id": p.ref, "title": p.title, "section": p.section, "text": p.text}
                  for i, p in enumerate(kn.search_procedures(query, typologies, rules))]

    top_shap = [{"feature": f["feature"], "shap": f["shap"], "value": f["value"]}
                for f in (tx.explanation or {}).get("top_features", [])[:6]]
    return {
        "transaction": {
            "channel": tx.channel, "operator": tx.operator, "tx_type": tx.tx_type,
            "tx_type_label": TX_TYPE_LABELS.get(tx.tx_type, tx.tx_type), "access_channel": tx.access_channel,
            "amount_usd": round(tx.amount_usd, 2), "province": tx.province, "hour": tx.tx_time.hour,
            "date": tx.tx_time.strftime("%Y-%m-%d"),
        },
        "decision": {
            "fraud_probability": round(tx.fraud_probability, 4), "risk_level": tx.risk_level,
            "action": tx.action, "reason": tx.reason, "rules": rules, "top_shap": top_shap,
            "label": tx.label,
        },
        "facts": facts,
        "mitigating": mitigating(feats, tx.tx_type),
        "precedents": precedents,
        "typology_hypotheses": [{**t, "label": TYPOLOGY_LABELS.get(t["typology"], t["typology"])}
                                for t in precedents["typologies"]],
        "cases": cases,
        "entity_links": _entity_links(db, tx),
        "client_history": _client_history(db, tx),
        "procedures": procedures,
    }
