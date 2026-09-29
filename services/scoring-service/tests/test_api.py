"""Tests de l'API /v1/score avec le vrai modèle et un Redis simulé (fakeredis)."""
import json
import os
import sys
from pathlib import Path

import fakeredis
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "scoring-service")]
os.environ.update(KAFKA_ENABLED="false", ARTIFACTS_DIR=str(ROOT / "ml" / "artifacts"),
                  SCORING_API_KEYS="test-key")

import redis  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

FAKE = fakeredis.FakeRedis()
FAKE.set("fs:meta", json.dumps({"period_start": "2025-06-01"}))
FAKE.set("fs:profile:U1", json.dumps({"province": "Kinshasa", "kyc_level": 2, "kyc_tx_limit_usd": 500.0,
                                      "account_age_days": 400, "monthly_income_usd": 300.0,
                                      "has_visa_virtual": 0}))


@pytest.fixture(scope="module")
def client():
    redis.Redis.from_url = classmethod(lambda cls, *a, **k: FAKE)
    from app.main import app
    with TestClient(app) as c:
        yield c


def tx(i, **kw):
    return {"transaction_id": f"T{i}", "timestamp": f"2025-11-01T10:{i:02d}:00", "user_id": "U1",
            "channel": "MOBILE_MONEY", "operator": "VODACOM", "tx_type": "P2P_SEND", "access_channel": "APP",
            "amount": 12.0, "currency": "USD", "amount_usd": 12.0, "balance_before_usd": 150.0,
            "counterparty_id": "243820000001", "device_id": "DEV1", "device_type": "smartphone",
            "ip_country": "CD", "location_province": "Kinshasa", **kw}


def test_requires_api_key(client):
    assert client.post("/v1/score", json=tx(0)).status_code == 422
    assert client.post("/v1/score", json=tx(0), headers={"X-API-Key": "mauvaise"}).status_code == 401


def test_score_returns_decision_and_explanation(client):
    r = client.post("/v1/score", json=tx(1), headers={"X-API-Key": "test-key"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert 0.0 <= d["fraud_probability"] <= 1.0
    assert d["action"] in ("APPROVE", "VERIFY", "BLOCK")
    # branches du modèle en production (metadata.json -> branch_order)
    branches = json.loads((ROOT / "ml" / "artifacts" / "metadata.json").read_text(encoding="utf-8"))["branch_order"]
    assert set(d["explanation"]["branch_scores"]) == set(branches)
    assert d["explanation"]["attribution"]
    from ml.features.feature_engineering import FEATURE_NAMES
    assert len(d["features"]) == len(FEATURE_NAMES) and d["known_user"] is True


def test_profile_is_updated_between_calls(client):
    h = {"X-API-Key": "test-key"}
    first = client.post("/v1/score", json=tx(2, device_id="NEWDEV"), headers=h).json()
    second = client.post("/v1/score", json=tx(3, device_id="NEWDEV"), headers=h).json()
    assert first["features"]["is_new_device"] == 1.0
    assert second["features"]["is_new_device"] == 0.0
    assert second["features"]["tx_count_1h"] > first["features"]["tx_count_1h"]


def test_account_drain_from_new_device_is_not_approved(client):
    h = {"X-API-Key": "test-key"}
    client.post("/v1/score", json=tx(4), headers=h)
    d = client.post("/v1/score", json=tx(5, tx_type="CASH_OUT", access_channel="AGENT", agent_id="A9",
                                         device_id="INCONNU", amount=148.0, amount_usd=148.0,
                                         counterparty_id=None, location_province="Lualaba",
                                         timestamp="2025-11-01T03:10:00"), headers=h).json()
    assert d["action"] != "APPROVE"


def test_invalid_transaction_is_rejected(client):
    r = client.post("/v1/score", json=tx(6, amount_usd=-5), headers={"X-API-Key": "test-key"})
    assert r.status_code == 422


def test_health_and_metrics(client):
    assert client.get("/health").json()["redis"] is True
    assert "scoring_latency_seconds" in client.get("/metrics").text


# ---------------------------------------------------------------------- idempotence
def tx_count_in_profile() -> int:
    return json.loads(FAKE.get("fs:user:U1"))["n"]


def test_resent_transaction_returns_same_decision_without_recounting(client):
    """Renvoi après coupure réseau : même décision, profil du client NON modifié."""
    h = {"X-API-Key": "test-key"}
    first = client.post("/v1/score", json=tx(10), headers=h).json()
    n_after_first = tx_count_in_profile()
    again = client.post("/v1/score", json=tx(10), headers=h)
    assert again.status_code == 200
    d = again.json()
    assert d["idempotent_replay"] is True and first["idempotent_replay"] is False
    assert d["fraud_probability"] == first["fraud_probability"] and d["action"] == first["action"]
    assert tx_count_in_profile() == n_after_first
    assert "scoring_idempotent_replays_total" in client.get("/metrics").text


def test_same_id_with_different_content_is_refused(client):
    h = {"X-API-Key": "test-key"}
    client.post("/v1/score", json=tx(11), headers=h)
    r = client.post("/v1/score", json=tx(11, amount=999.0, amount_usd=999.0), headers=h)
    assert r.status_code == 409


# ---------------------------------------------------------------------- réputation
def test_confirmed_fraud_marks_its_device_for_next_transactions(client):
    h = {"X-API-Key": "test-key"}
    before = client.post("/v1/score", json=tx(20, device_id="DEV-FRAUDE"), headers=h).json()
    assert before["features"]["rep_device_frauds"] == 0.0
    r = client.post("/v1/reputation/report", headers=h, json={
        "transaction_id": "T20", "tx_time": "2025-11-01T10:20:00", "tx_type": "P2P_SEND", "user_id": "U1",
        "device_id": "DEV-FRAUDE", "counterparty_id": "243820000001"})
    assert r.status_code == 200 and r.json()["entities_updated"] == 3   # appareil, portefeuille, compte
    after = client.post("/v1/score", json=tx(21, device_id="DEV-FRAUDE"), headers=h).json()
    assert after["features"]["rep_device_frauds"] > 0 and after["features"]["rep_cp_frauds"] > 0


# ---------------------------------------------------------------------- signal télécom
def test_recent_sim_swap_forces_out_of_band_verification(client):
    d = client.post("/v1/score", json=tx(12, sim_swap_at="2025-11-01T09:00:00"),   # SIM changée 3 h avant
                    headers={"X-API-Key": "test-key"}).json()
    assert "SIM_RECENTE_OPERATION_SORTANTE" in d["rules_triggered"]
    assert d["action"] in ("VERIFY", "BLOCK")
    if d["action"] == "VERIFY":
        assert d["verification_method"] == "HORS_SIM"   # un OTP par SMS arriverait chez le fraudeur


# ---------------------------------------------------------------------- signalements SMS (NLP)
def scam_report(report_id, numbers, sender_is_number=True, text=None):
    return {"report_id": report_id, "received_at": "2025-11-01T09:00:00", "sender_is_number": sender_is_number,
            "numbers": numbers, "masked_text": text or ("Bonjour, je vous ai envoyé 50 000 FC par erreur, c'était pour "
                                                         "ma mère malade. Svp renvoyez au <NUMERO>. Dieu vous bénisse")}


def test_scam_sms_report_flags_number_then_rule_fires(client):
    h = {"X-API-Key": "test-key"}
    r = client.post("/v1/scam-reports", json=scam_report("R1", ["243899990001"]), headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["is_scam"] and r.json()["flagged_numbers"] == ["243899990001"]
    d = client.post("/v1/score", json=tx(40, counterparty_id="243899990001"), headers=h).json()
    assert "BENEFICIAIRE_SIGNALE_PAR_SMS" in d["rules_triggered"] and d["action"] in ("VERIFY", "BLOCK")


def test_operator_sender_never_flags_a_number(client):
    r = client.post("/v1/scam-reports", headers={"X-API-Key": "test-key"},
                    json=scam_report("R2", ["243899990002"], sender_is_number=False))
    assert r.status_code == 200 and r.json()["flagged_numbers"] == []
