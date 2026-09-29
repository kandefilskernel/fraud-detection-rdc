"""Retours des opérateurs (API interne) : ils deviennent des étiquettes de réentraînement."""
from datetime import datetime, timezone

from sqlalchemy import insert, select

from app.db import engine
from shared.database.models import Case, ScoredTransaction

KEY = {"X-Internal-Key": "test-internal-key"}


def seed_tx(tx_id: str, operator: str = "VODACOM", channel: str = "MOBILE_MONEY", action: str = "APPROVE") -> None:
    with engine.begin() as c:
        c.execute(insert(ScoredTransaction).values(
            transaction_id=tx_id, tx_time=datetime(2025, 11, 4, 14, 0), scored_at=datetime.now(timezone.utc),
            user_id="U42", channel=channel, operator=operator, tx_type="P2P_SEND", access_channel="APP",
            amount=80.0, currency="USD", amount_usd=80.0, province="Kinshasa", device_id="D42",
            fraud_probability=0.04, risk_level="FAIBLE", action=action, reason="test", rules_triggered=[],
            model_version="t", latency_ms=10.0, degraded=False, explanation={}, features={}))


def label_of(tx_id: str):
    with engine.connect() as c:
        return c.scalar(select(ScoredTransaction.label).where(ScoredTransaction.transaction_id == tx_id))


def feedback(client, tx_id, outcome, provider="vodacom", headers=KEY, **kw):
    return client.post("/internal/operator-feedback", headers=headers,
                       json={"transaction_id": tx_id, "provider": provider, "outcome": outcome, **kw})


def test_requires_internal_key(client):
    seed_tx("TX-FB-KEY")
    assert feedback(client, "TX-FB-KEY", "CHARGEBACK", headers={}).status_code == 401
    assert feedback(client, "TX-FB-KEY", "CHARGEBACK", headers={"X-Internal-Key": "faux"}).status_code == 401


def test_chargeback_on_missed_fraud_creates_closed_case_and_label(client):
    """Fraude laissée passer (APPROVE) puis contestée : c'est le faux négatif que le modèle doit apprendre."""
    seed_tx("TX-FB-1")
    r = feedback(client, "TX-FB-1", "CHARGEBACK", reference="VDC-2025-778", comment="client en agence")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == 1 and body["changed"] and body["case_status"] == "FRAUDE_CONFIRMEE"
    assert label_of("TX-FB-1") == 1
    with engine.connect() as c:
        note = c.scalar(select(Case.resolution_note).where(Case.transaction_id == "TX-FB-1"))
    assert "VDC-2025-778" in note and "chargeback" in note


def test_repeated_feedback_is_idempotent(client):
    seed_tx("TX-FB-2")
    assert feedback(client, "TX-FB-2", "FRAUD_CONFIRMED").json()["changed"] is True
    again = feedback(client, "TX-FB-2", "FRAUD_CONFIRMED").json()
    assert again["changed"] is False and again["label"] == 1


def test_legitimate_closes_open_case_as_false_positive(client):
    seed_tx("TX-FB-3", action="VERIFY")
    with engine.begin() as c:
        c.execute(insert(Case).values(transaction_id="TX-FB-3", tx_time=datetime(2025, 11, 4, 14, 0), user_id="U42",
                                      channel="MOBILE_MONEY", tx_type="P2P_SEND", amount_usd=80.0,
                                      fraud_probability=0.6, risk_level="ELEVE", action="VERIFY"))
    r = feedback(client, "TX-FB-3", "LEGITIMATE", comment="client a confirmé par USSD")
    assert r.json()["label"] == 0 and r.json()["case_status"] == "FAUX_POSITIF"
    assert label_of("TX-FB-3") == 0


def test_legitimate_never_overrides_a_fraud_label(client):
    seed_tx("TX-FB-4")
    feedback(client, "TX-FB-4", "CUSTOMER_COMPLAINT")
    assert feedback(client, "TX-FB-4", "LEGITIMATE").status_code == 409
    assert label_of("TX-FB-4") == 1


def test_operator_cannot_label_another_operators_transaction(client):
    seed_tx("TX-FB-5", operator="AIRTEL")
    assert feedback(client, "TX-FB-5", "FRAUD_CONFIRMED", provider="vodacom").status_code == 403
    seed_tx("TX-FB-6", operator=None, channel="VISA_VIRTUAL")
    assert feedback(client, "TX-FB-6", "CHARGEBACK", provider="visa").status_code == 200


def test_unknown_transaction_returns_404(client):
    assert feedback(client, "TX-INEXISTANTE", "FRAUD_CONFIRMED").status_code == 404
