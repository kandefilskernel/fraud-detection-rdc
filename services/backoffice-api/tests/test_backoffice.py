from datetime import datetime, timezone

import psycopg2
import pytest
from sqlalchemy import insert

from app.db import engine
from shared.database.models import Case, ScoredTransaction

from .conftest import PG, TEST_DB


def seed_alert(tx_id: str) -> int:
    tx_time = datetime(2025, 11, 3, 2, 15)
    with engine.begin() as c:
        c.execute(insert(ScoredTransaction).values(
            transaction_id=tx_id, tx_time=tx_time, scored_at=datetime.now(timezone.utc), user_id="U9",
            channel="MOBILE_MONEY", operator="AIRTEL", tx_type="CASH_OUT", access_channel="AGENT", amount=250.0,
            currency="USD", amount_usd=250.0, province="Kinshasa", device_id="D9", fraud_probability=0.93,
            risk_level="CRITIQUE", action="BLOCK", reason="test", rules_triggered=[], model_version="t",
            latency_ms=12.0, degraded=False, explanation={}, features={}))
        return c.execute(insert(Case).values(
            transaction_id=tx_id, tx_time=tx_time, user_id="U9", channel="MOBILE_MONEY", tx_type="CASH_OUT",
            amount_usd=250.0, fraud_probability=0.93, risk_level="CRITIQUE", action="BLOCK")
            .returning(Case.id)).scalar_one()


def test_login_rejects_bad_password(client):
    assert client.post("/auth/login", data={"username": "admin@test.local", "password": "faux"}).status_code == 401


def test_endpoints_require_token(client):
    assert client.get("/transactions").status_code == 401


def test_rbac_analyst_cannot_manage_users(client, analyst):
    r = client.post("/users", headers=analyst, json={"email": "x@exemple.cd", "full_name": "X",
                                                     "role": "ADMIN", "password": "Password-2026!"})
    assert r.status_code == 403


def test_rbac_analyst_cannot_read_audit(client, analyst):
    assert client.get("/audit", headers=analyst).status_code == 403


def test_case_workflow_sets_label(client, analyst):
    case_id = seed_alert("TX-CASE-1")
    assert client.patch(f"/cases/{case_id}", headers=analyst, json={"status": "EN_COURS"}).json()["status"] == "EN_COURS"
    # note obligatoire pour clore
    assert client.patch(f"/cases/{case_id}", headers=analyst, json={"status": "FAUX_POSITIF"}).status_code == 422
    r = client.patch(f"/cases/{case_id}", headers=analyst,
                     json={"status": "FAUX_POSITIF", "resolution_note": "Client en voyage, opération confirmée"})
    assert r.status_code == 200 and r.json()["resolved_by"] is not None
    detail = client.get("/transactions/TX-CASE-1", headers=analyst).json()
    assert detail["label"] == 0
    # un dossier clos ne peut plus changer d'état
    assert client.patch(f"/cases/{case_id}", headers=analyst, json={"status": "OUVERT"}).status_code == 409


def test_reports_are_available(client, admin):
    k = client.get("/reports/kpis?minutes=60", headers=admin).json()
    assert k["volume"] >= 1 and "alert_rate" in k
    assert client.get("/reports/breakdown?dimension=operator", headers=admin).status_code == 200
    assert client.get("/reports/breakdown?dimension=drop table", headers=admin).status_code == 422


def test_audit_log_is_append_only():
    conn = psycopg2.connect(dbname=TEST_DB, **PG)
    with conn, conn.cursor() as cur:
        cur.execute("INSERT INTO audit_log (ts, actor, action, entity, details, prev_hash, hash) "
                    "VALUES (now(), 'test', 'X', 'y', '{}', 'a', 'b')")
    with pytest.raises(psycopg2.Error):
        with conn, conn.cursor() as cur:
            cur.execute("UPDATE audit_log SET actor = 'pirate'")
    conn.close()
