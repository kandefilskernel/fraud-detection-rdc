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
    assert set(d["explanation"]["branch_scores"]) == {"xgboost", "lstm_attention", "autoencoder"}
    assert len(d["features"]) == 58 and d["known_user"] is True


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
