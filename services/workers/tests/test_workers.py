import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "workers")]
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://u:p@localhost:1/none")  # moteur paresseux

from shared.database.audit_chain import GENESIS, entry_hash, verify_chain  # noqa: E402

EVENT = {
    "scored_at": "2026-09-25T10:00:00+00:00",
    "transaction": {"transaction_id": "TX1", "timestamp": "2025-11-02T21:15:07", "user_id": "U1",
                    "channel": "MOBILE_MONEY", "operator": "VODACOM", "tx_type": "CASH_OUT",
                    "access_channel": "AGENT", "amount": 285000.0, "currency": "CDF", "amount_usd": 100.0,
                    "location_province": "Kinshasa", "device_id": "D1", "agent_id": "A1", "wallet_id": "2438"},
    "decision": {"fraud_probability": 0.97, "risk_level": "CRITIQUE", "action": "BLOCK", "reason": "x",
                 "rules_triggered": [], "model_version": "v1", "latency_ms": 9.5, "degraded": False},
    "explanation": {"top_features": [{"feature": "is_new_device", "shap": 1.2, "value": 1.0}]},
    "features": {"is_new_device": 1.0},
}


def test_persister_maps_event_to_row():
    from workers.persister import to_row
    row = to_row(EVENT)
    assert row["tx_time"] == datetime(2025, 11, 2, 21, 15, 7)
    assert row["province"] == "Kinshasa" and row["action"] == "BLOCK" and row["agent_id"] == "A1"


def test_sms_texts_are_short_and_never_ask_for_pin():
    from workers.alerter import sms_text
    for action in ("VERIFY", "BLOCK"):
        txt = sms_text({**EVENT, "decision": {**EVENT["decision"], "action": action}})
        assert len(txt) <= 200 and "285 000 CDF" in txt
        assert "communiquez" in txt or "confirmez" in txt
    assert sms_text({**EVENT, "decision": {**EVENT["decision"], "action": "APPROVE"}}) is None


def _chain(n):
    rows, prev = [], GENESIS
    for i in range(n):
        ts = datetime(2026, 9, 25, 10, i, tzinfo=timezone.utc)
        h = entry_hash(prev, ts, "a", "LOGIN", "bo_user", str(i), {"k": i})
        rows.append(SimpleNamespace(id=i + 1, ts=ts, actor="a", action="LOGIN", entity="bo_user",
                                    entity_id=str(i), details={"k": i}, prev_hash=prev, hash=h))
        prev = h
    return rows


def test_hash_chain_detects_tampering():
    rows = _chain(5)
    assert verify_chain(rows)["valid"]
    rows[2].details = {"k": 999}            # modification d'une entrée passée
    res = verify_chain(rows)
    assert not res["valid"] and res["broken_at_id"] == 3


def test_hash_chain_detects_deletion():
    rows = _chain(5)
    del rows[1]
    assert verify_chain(rows)["broken_at_id"] == 3
