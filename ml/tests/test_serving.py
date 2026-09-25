"""Tests du chemin de production : feature store Redis, scorer, décision."""
from collections import deque
from datetime import datetime

import fakeredis
import numpy as np
import pandas as pd
import pytest

from ml.features.feature_engineering import BehavioralFeatureExtractor, _UserState
from ml.serving.decision import DecisionPolicy
from ml.serving.feature_store import RedisFeatureStore, user_state_from_json, user_state_to_json
from shared.schemas.unified_transaction import UnifiedTransaction, canonical_id

USERS = pd.DataFrame([{"user_id": "U1", "province": "Kinshasa", "kyc_level": 2, "kyc_tx_limit_usd": 500.0,
                       "account_age_days": 400, "monthly_income_usd": 300.0, "has_visa_virtual": 1}])


def make_tx(i: int, **kw) -> dict:
    base = dict(transaction_id=f"T{i}", timestamp=datetime(2025, 11, 1, 10, i % 60), user_id="U1",
                wallet_id="243810000001", channel="MOBILE_MONEY", operator="VODACOM", tx_type="P2P_SEND",
                access_channel="APP", amount=10.0 + i, currency="USD", amount_usd=10.0 + i,
                balance_before_usd=200.0, counterparty_id=f"2438200000{i % 3}", device_id="DEV1",
                device_type="smartphone", ip_country="CD", location_province="Kinshasa")
    base.update(kw)
    return UnifiedTransaction(**base).to_feature_input("SUCCESS")


def test_canonical_id_handles_pandas_floats():
    assert canonical_id(243917538864.0) == "243917538864"
    assert canonical_id("A12") == "A12"
    assert canonical_id(float("nan")) is None


def test_timestamp_with_timezone_is_converted_to_kinshasa_time():
    tx = UnifiedTransaction(**{**make_tx_kwargs(), "timestamp": "2025-11-01T09:30:00+00:00"})
    assert tx.timestamp == datetime(2025, 11, 1, 10, 30)


def make_tx_kwargs():
    return dict(transaction_id="T", timestamp=datetime(2025, 11, 1), user_id="U1", channel="MOBILE_MONEY",
                tx_type="CASH_OUT", access_channel="AGENT", amount=5.0, currency="USD", amount_usd=5.0,
                balance_before_usd=10.0, device_id="D", location_province="Kinshasa")


def test_user_state_json_roundtrip():
    st = _UserState(n=3, recent=deque([(1.0, 2.0, True)]), failures=deque([5.0]), devices={"D": 2},
                    provinces={"Kinshasa"}, counterparties={"C1"}, agents={"A1"}, merchants={"M1"})
    back = user_state_from_json(user_state_to_json(st))
    assert back == st


def test_redis_store_matches_in_memory_extractor():
    """Même séquence de transactions : variables identiques en mémoire et via Redis."""
    period_start = pd.Timestamp("2025-06-01")
    mem = BehavioralFeatureExtractor(USERS, period_start)
    store = RedisFeatureStore(fakeredis.FakeRedis(), period_start.timestamp())
    store.bulk_load(BehavioralFeatureExtractor(USERS, period_start), {})
    identity = lambda f: np.zeros(58, dtype=np.float32)  # noqa: E731
    for i in range(12):
        tx = make_tx(i, agent_id="A1" if i % 4 == 0 else None,
                     tx_type="CASH_OUT" if i % 4 == 0 else "P2P_SEND")
        expected = mem.process(tx)
        got = store.compute(tx, identity)["features"]
        assert got == pytest.approx(expected)


def test_blocked_transaction_is_recorded_as_failure():
    period_start = pd.Timestamp("2025-06-01")
    store = RedisFeatureStore(fakeredis.FakeRedis(), period_start.timestamp())
    store.bulk_load(BehavioralFeatureExtractor(USERS, period_start), {})
    identity = lambda f: np.zeros(58, dtype=np.float32)  # noqa: E731
    tx = {**make_tx(0), "status": None}
    store.compute(tx, identity, lambda f, x, h: ("BLOCK", "FAILED"))
    nxt = store.compute({**make_tx(1), "status": None}, identity, lambda f, x, h: ("OK", "SUCCESS"))
    assert nxt["features"]["failed_count_24h"] == 1.0


def test_unknown_user_gets_default_profile():
    store = RedisFeatureStore(fakeredis.FakeRedis(), pd.Timestamp("2025-06-01").timestamp())
    out = store.compute({**make_tx(0), "user_id": "INCONNU"}, lambda f: np.zeros(58, dtype=np.float32))
    assert out["known_user"] is False
    assert out["features"]["kyc_level"] == 1.0


# ---------------------------------------------------------------------- décision
FEATS = {"is_new_device": 0.0, "is_near_full_drain": 0.0, "amount_to_kyc_limit": 0.1, "is_new_province": 0.0}
TX = {"tx_type": "P2P_SEND"}


def test_low_probability_is_approved():
    d = DecisionPolicy(threshold=0.5).decide(0.001, 500, "MOBILE_MONEY", FEATS, TX)
    assert d["action"] == "APPROVE" and d["risk_level"] == "FAIBLE"


def test_very_high_probability_is_blocked_whatever_the_amount():
    d = DecisionPolicy(threshold=0.5).decide(0.99, 0.5, "MOBILE_MONEY", FEATS, TX)
    assert d["action"] == "BLOCK" and d["risk_level"] == "CRITIQUE"


def test_amount_changes_the_decision_at_equal_probability():
    policy = DecisionPolicy(threshold=0.5)
    small = policy.decide(0.3, 1.0, "MOBILE_MONEY", FEATS, TX)["action"]
    large = policy.decide(0.3, 900.0, "MOBILE_MONEY", FEATS, TX)["action"]
    assert small == "APPROVE" and large != "APPROVE"


def test_rules_can_only_strengthen_the_decision():
    feats = {**FEATS, "is_new_device": 1.0, "is_near_full_drain": 1.0}
    d = DecisionPolicy(threshold=0.5).decide(0.001, 50, "MOBILE_MONEY", feats, TX)
    assert d["action"] == "VERIFY"
    assert "NOUVEL_APPAREIL_VIDAGE_COMPTE" in d["rules_triggered"]
