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


NET_USERS = pd.DataFrame([{"user_id": f"U{i}", "wallet_id": f"24381000000{i}", "province": "Kinshasa",
                           "kyc_level": 2, "kyc_tx_limit_usd": 1500.0, "account_age_days": 100 * (i + 1),
                           "monthly_income_usd": 300.0, "has_visa_virtual": 0} for i in range(6)])


def network_stream(n: int = 80) -> list[dict]:
    """Plusieurs clients et opérateurs, portefeuilles partagés (mule, client, vendeur),
    réceptions puis renvois : exerce tout l'état C1 (profil des portefeuilles)."""
    rng = np.random.default_rng(3)
    wallets = ["243777000001", "243777000002", "243810000003", "243888000001"]
    ops = ["VODACOM", "AIRTEL", "ORANGE"]
    out = []
    for i in range(n):
        uid = f"U{rng.integers(6)}"
        kind = rng.choice(["P2P_SEND", "P2P_RECEIVE", "CASH_OUT"], p=[0.55, 0.3, 0.15])
        kw = dict(transaction_id=f"N{i}", timestamp=datetime(2025, 11, 1) + pd.Timedelta(minutes=7 * i),
                  user_id=uid, wallet_id=f"24381000000{uid[1]}", operator=ops[rng.integers(3)],
                  tx_type=kind, amount=float(rng.integers(1, 80)), amount_usd=float(rng.integers(1, 80)),
                  counterparty_id=wallets[rng.integers(4)] if kind != "CASH_OUT" else None,
                  access_channel="AGENT" if kind == "CASH_OUT" else "APP",
                  agent_id="A1" if kind == "CASH_OUT" else None, device_id=f"D{uid}")
        out.append(make_tx(i, **kw))
    return out


def test_redis_store_matches_extractor_on_shared_wallets():
    """C1 : profils de portefeuilles alimentés par plusieurs clients — parité Redis / mémoire."""
    period_start = pd.Timestamp("2025-06-01")
    mem = BehavioralFeatureExtractor(NET_USERS, period_start)
    store = RedisFeatureStore(fakeredis.FakeRedis(), period_start.timestamp())
    store.bulk_load(BehavioralFeatureExtractor(NET_USERS, period_start), {})
    zero = lambda f: np.zeros(4, dtype=np.float32)  # noqa: E731
    nonzero = 0
    for tx in network_stream():
        expected = mem.process(tx)
        assert store.compute(tx, zero)["features"] == pytest.approx(expected)
        nonzero += expected["cp_in_senders_7d"] > 0
    assert nonzero > 10  # le scénario exerce bien le profil des portefeuilles


def test_seeding_mid_stream_keeps_parity():
    """Chemin réel : rejeu hors ligne jusqu'à une coupure, amorçage de Redis, puis temps réel."""
    period_start = pd.Timestamp("2025-06-01")
    stream = network_stream()
    offline = BehavioralFeatureExtractor(NET_USERS, period_start)
    for tx in stream[:50]:
        offline.process(tx)
    store = RedisFeatureStore(fakeredis.FakeRedis(), period_start.timestamp())
    store.bulk_load(offline, {})
    zero = lambda f: np.zeros(4, dtype=np.float32)  # noqa: E731
    for tx in stream[50:]:
        got = store.compute(tx, zero)["features"]   # lit Redis AVANT que offline ne change
        assert got == pytest.approx(offline.process(tx))


def test_reputation_parity_with_reports_interleaved():
    """Signalements de fraude reçus en cours de route (topic fraud.confirmed) : mêmes variables
    de réputation en mémoire (report) et via Redis (apply_report)."""
    period_start = pd.Timestamp("2025-06-01")
    stream = network_stream()
    mem = BehavioralFeatureExtractor(NET_USERS, period_start)
    store = RedisFeatureStore(fakeredis.FakeRedis(), period_start.timestamp())
    store.bulk_load(BehavioralFeatureExtractor(NET_USERS, period_start), {})
    zero = lambda f: np.zeros(4, dtype=np.float32)  # noqa: E731
    seen_rep = 0
    for i, tx in enumerate(stream):
        if i >= 10 and i % 7 == 0:        # la transaction d'il y a 5 pas est signalée frauduleuse
            event = {**stream[i - 5], "transaction_id": f"N{i - 5}"}
            mem.report(event)
            store.apply_report(event)
        expected = mem.process(tx)
        assert store.compute(tx, zero)["features"] == pytest.approx(expected)
        seen_rep += expected["rep_user_frauds_30d"] > 0
    assert seen_rep > 5


def test_seeding_carries_reputation():
    period_start = pd.Timestamp("2025-06-01")
    stream = network_stream()
    offline = BehavioralFeatureExtractor(NET_USERS, period_start)
    for i, tx in enumerate(stream[:50]):
        offline.process(tx)
        if i % 9 == 0:
            offline.report({**tx, "transaction_id": f"N{i}"})
    store = RedisFeatureStore(fakeredis.FakeRedis(), period_start.timestamp())
    store.bulk_load(offline, {})
    zero = lambda f: np.zeros(4, dtype=np.float32)  # noqa: E731
    for tx in stream[50:]:
        assert store.compute(tx, zero)["features"] == pytest.approx(offline.process(tx))


def test_user_state_without_c1_fields_still_loads():
    """Profils écrits dans Redis avant C1 : lisibles (réceptions vides)."""
    import json
    raw = json.loads(user_state_to_json(_UserState(n=1)))
    raw.pop("inflows")
    assert len(user_state_from_json(json.dumps(raw)).inflows) == 0


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


def test_late_transaction_does_not_crash_feature_extraction():
    """Transaction arrivée en retard (horloges d'opérateurs, reprise) : pas d'erreur de domaine."""
    store = RedisFeatureStore(fakeredis.FakeRedis(), pd.Timestamp("2025-06-01").timestamp())
    store.bulk_load(BehavioralFeatureExtractor(USERS, pd.Timestamp("2025-06-01")), {})
    zero = lambda f: np.zeros(58, dtype=np.float32)  # noqa: E731
    topup = make_tx(30, channel="VISA_VIRTUAL", tx_type="CARD_TOPUP", counterparty_id=None)
    store.compute(topup, zero)
    late = make_tx(5, channel="VISA_VIRTUAL", tx_type="CARD_PURCHASE", counterparty_id=None,
                   merchant_id="M1", merchant_category="GROCERY", merchant_country="CD")
    feats = store.compute(late, zero)["features"]
    assert feats["log_mins_since_topup"] == 0.0 and feats["log_history_days"] >= 0.0
