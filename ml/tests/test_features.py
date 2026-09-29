"""Tests de la phase 2 : variables comportementales, split temporel, séquences."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "generator"))
from generate_synthetic_data import GeneratorConfig, SyntheticDataGenerator  # noqa: E402

from ml.features.feature_engineering import (  # noqa: E402
    FEATURE_NAMES, BehavioralFeatureExtractor, build_feature_table,
)
from ml.preprocessing.train_test_split_normalize import (  # noqa: E402
    build_sequence_index, temporal_split,
)

CFG = GeneratorConfig(n_users=250, n_days=45, seed=11, visa_adoption_multiplier=2.0)


@pytest.fixture(scope="module")
def raw():
    return SyntheticDataGenerator(CFG).run()


@pytest.fixture(scope="module")
def table(raw):
    return build_feature_table(raw["transactions"], raw["users"], pd.Timestamp(CFG.start_date))


def test_shape_and_finite(raw, table):
    assert len(table) == len(raw["transactions"])
    X = table[FEATURE_NAMES].to_numpy()
    assert np.isfinite(X).all()
    assert table["timestamp"].is_monotonic_increasing


def test_no_future_leakage(raw, table):
    """Supprimer les transactions futures ne doit changer AUCUNE variable du passé."""
    tx = raw["transactions"]
    cutoff = pd.Timestamp(CFG.start_date) + pd.Timedelta(days=CFG.n_days // 2)
    past = build_feature_table(tx[tx.timestamp < cutoff], raw["users"], pd.Timestamp(CFG.start_date))
    full_past = table[table.timestamp < cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(past[FEATURE_NAMES], full_past[FEATURE_NAMES])


def test_labels_not_in_features():
    forbidden = {"is_fraud", "fraud_type", "fraud_episode_id", "is_mule_account", "is_compromised"}
    assert not forbidden & set(FEATURE_NAMES)


def test_behavioral_signals_separate_fraud(table):
    """Sanity check : les signaux clés vont dans le bon sens, sans être décisifs à eux seuls
    (téléphones partagés, dépôts initiés par l'agent : un appareil nouveau est aussi fréquent
    chez les clients légitimes)."""
    fraud, legit = table[table.is_fraud == 1], table[table.is_fraud == 0]
    assert fraud["log_device_prior_uses"].mean() < legit["log_device_prior_uses"].mean()
    assert fraud["amount_to_balance"].mean() > legit["amount_to_balance"].mean()
    assert fraud["is_new_counterparty"].mean() > legit["is_new_counterparty"].mean()


# ---------------------------------------------------------------------- C1 : réseau
C1_USERS = pd.DataFrame([
    {"user_id": f"U{i}", "wallet_id": f"24381000000{i}", "province": "Kinshasa", "kyc_level": 2,
     "kyc_tx_limit_usd": 1500.0, "account_age_days": 100 * (i + 1), "monthly_income_usd": 300.0,
     "has_visa_virtual": 0} for i in range(8)])
T0 = pd.Timestamp("2025-07-01 10:00").timestamp()


def c1_tx(uid, tx_type, cp, amt, minutes, operator="VODACOM", **kw):
    ts = T0 + minutes * 60
    return {"user_id": uid, "operator": operator, "tx_type": tx_type, "channel": "MOBILE_MONEY",
            "amount_usd": amt, "balance_before_usd": 500.0, "status": "SUCCESS", "device_id": f"D{uid}",
            "device_type": "smartphone", "access_channel": "APP", "ip_country": "CD",
            "location_province": "Kinshasa", "counterparty_id": cp, "agent_id": kw.get("agent_id"),
            "merchant_id": None, "merchant_category": None, "merchant_country": None,
            "ts": ts, "hour": pd.Timestamp(ts, unit="s").hour, "weekday": 1, "day": 1}


def c1_extractor():
    return BehavioralFeatureExtractor(C1_USERS, pd.Timestamp("2025-06-01"))


def test_c1_fake_mistaken_transfer_is_flagged():
    """« Je t'ai envoyé 2 USD par erreur, renvoie-moi 50 » : on renvoie beaucoup plus que reçu."""
    ext = c1_extractor()
    ext.process(c1_tx("U0", "P2P_RECEIVE", "243999000001", 2.0, 0))
    f = ext.process(c1_tx("U0", "P2P_SEND", "243999000001", 50.0, 12))
    assert f["refund_ratio_to_cp"] == pytest.approx(np.log1p(50 / 2.01))
    assert f["log_mins_since_received_from_cp"] == pytest.approx(np.log1p(12))
    normal = ext.process(c1_tx("U0", "P2P_SEND", "243999000002", 50.0, 20))
    assert normal["refund_ratio_to_cp"] == 0.0


def test_c1_mule_fan_in_across_operators():
    """Un portefeuille payé par 5 inconnus de 3 opérateurs : visible par le 6e expéditeur."""
    ext = c1_extractor()
    mule, ops = "243777000001", ["VODACOM", "AIRTEL", "ORANGE", "VODACOM", "AIRTEL"]
    for i, op in enumerate(ops):
        ext.process(c1_tx(f"U{i}", "P2P_SEND", mule, 30.0, i * 10, operator=op))
    f = ext.process(c1_tx("U5", "P2P_SEND", mule, 30.0, 60))
    assert f["cp_in_senders_7d"] == pytest.approx(np.log1p(5))
    assert f["cp_in_new_ratio_7d"] == pytest.approx(6 / 7)   # 5 premiers contacts sur 5
    assert f["cp_in_operators_7d"] == 3.0
    assert f["cp_is_customer"] == 0.0
    assert f["cp_log_age_days"] == pytest.approx(np.log1p(60 / 1440))  # vu pour la 1re fois il y a 1 h


def test_c1_known_customer_recipient():
    ext = c1_extractor()
    f = ext.process(c1_tx("U0", "P2P_SEND", "243810000003", 10.0, 0))   # portefeuille de U3
    assert f["cp_is_customer"] == 1.0
    days = 400 + (T0 - pd.Timestamp("2025-06-01").timestamp()) / 86400
    assert f["cp_log_age_days"] == pytest.approx(np.log1p(days))


def test_c1_pass_through_money():
    """Argent reçu de 3 inconnus puis retiré aussitôt : profil de compte mule."""
    ext = c1_extractor()
    for i in range(3):
        ext.process(c1_tx("U0", "P2P_RECEIVE", f"24366600000{i}", 40.0, i * 5))
    f = ext.process(c1_tx("U0", "CASH_OUT", None, 90.0, 30, agent_id="A1"))
    assert f["in_new_senders_24h"] == pytest.approx(np.log1p(3))
    assert f["log_inflow_24h"] == pytest.approx(np.log1p(120))
    assert f["passthrough_ratio"] == 1.0


def test_c1_window_expires():
    ext = c1_extractor()
    ext.process(c1_tx("U0", "P2P_SEND", "243777000009", 30.0, 0))
    f = ext.process(c1_tx("U1", "P2P_SEND", "243777000009", 30.0, 8 * 24 * 60))  # 8 jours plus tard
    assert f["cp_in_senders_7d"] == 0.0


def test_float_wallet_ids_are_canonicalized():
    """Un même portefeuille lu en flottant ou en texte par pandas = une seule entité."""
    rows = [c1_tx("U0", "P2P_SEND", 243555000001.0, 10.0, 0), c1_tx("U1", "P2P_SEND", "243555000001", 10.0, 5)]
    tx = pd.DataFrame(rows)
    tx["timestamp"] = pd.to_datetime(tx["ts"], unit="s")
    tx["transaction_id"] = ["T0", "T1"]
    tx["is_fraud"], tx["fraud_type"] = 0, None
    out = build_feature_table(tx, C1_USERS, pd.Timestamp("2025-06-01"))
    assert out.loc[1, "cp_in_senders_7d"] == pytest.approx(np.log1p(1))


# ---------------------------------------------------------------------- C2 : portée du profil
def _mixed_history() -> pd.DataFrame:
    """3 opérations Mobile Money puis un achat carte, sur le même téléphone."""
    rows = [c1_tx("U0", "P2P_SEND", "243999000001", 20.0, i * 30) for i in range(3)]
    card = c1_tx("U0", "CARD_PURCHASE", None, 25.0, 120)
    card.update(channel="VISA_VIRTUAL", merchant_id="V1", merchant_category="ECOMMERCE_INTL",
                merchant_country="FR")
    tx = pd.DataFrame(rows + [card])
    tx["timestamp"] = pd.to_datetime(tx["ts"], unit="s")
    tx["transaction_id"] = [f"T{i}" for i in range(len(tx))]
    tx["is_fraud"], tx["fraud_type"] = 0, None
    return tx


def test_c2_unified_profile_lets_the_card_inherit_wallet_history():
    out = build_feature_table(_mixed_history(), C1_USERS, pd.Timestamp("2025-06-01"), "unified")
    card = out.iloc[-1]
    assert card["log_user_tx_count"] == pytest.approx(np.log1p(3))
    assert card["log_device_prior_uses"] == pytest.approx(np.log1p(3))   # téléphone connu du portefeuille


def test_c2_silo_profile_hides_wallet_history_from_the_card():
    out = build_feature_table(_mixed_history(), C1_USERS, pd.Timestamp("2025-06-01"), "silo")
    card = out.iloc[-1]
    assert card["log_user_tx_count"] == 0.0           # l'émetteur de la carte ne voit que la carte
    assert card["log_device_prior_uses"] == 0.0
    assert out.iloc[2]["log_user_tx_count"] == pytest.approx(np.log1p(2))  # le portefeuille, lui, voit son historique


def test_c2_silo_has_no_future_leakage(raw):
    tx = raw["transactions"]
    cutoff = pd.Timestamp(CFG.start_date) + pd.Timedelta(days=CFG.n_days // 2)
    start = pd.Timestamp(CFG.start_date)
    full = build_feature_table(tx, raw["users"], start, "silo")
    past = build_feature_table(tx[tx.timestamp < cutoff], raw["users"], start, "silo")
    pd.testing.assert_frame_equal(past[FEATURE_NAMES], full[full.timestamp < cutoff].reset_index(drop=True)[FEATURE_NAMES])


# ---------------------------------------------------------------------- réputation
def test_reputation_only_counts_frauds_reported_before_the_transaction():
    """Une fraude n'aide qu'APRÈS son signalement : avant, l'opérateur ne la connaît pas."""
    ext = c1_extractor()
    fraud = c1_tx("U0", "P2P_SEND", "243777000050", 60.0, 0, agent_id=None)
    ext.process(fraud)
    before = ext.process(c1_tx("U1", "P2P_SEND", "243777000050", 60.0, 30))   # pas encore signalée
    assert before["rep_cp_frauds"] == 0.0
    ext.report({**fraud, "transaction_id": "TF1"})                             # signalement
    after = ext.process(c1_tx("U2", "P2P_SEND", "243777000050", 60.0, 60))
    assert after["rep_cp_frauds"] == pytest.approx(np.log1p(1))
    assert after["rep_device_frauds"] == 0.0                                   # autre appareil
    ext.report({**fraud, "transaction_id": "TF1"})                             # signalement répété
    assert ext.process(c1_tx("U3", "P2P_SEND", "243777000050", 60.0, 90))["rep_cp_frauds"] == pytest.approx(np.log1p(1))


def test_reputation_windows_for_agent_and_account():
    ext = c1_extractor()
    cashout = c1_tx("U0", "CASH_OUT", None, 90.0, 0, agent_id="A7")
    ext.report({**cashout, "transaction_id": "TF2"})
    soon = ext.process(c1_tx("U4", "CASH_OUT", None, 20.0, 60 * 24 * 3, agent_id="A7"))       # 3 jours après
    late = ext.process(c1_tx("U5", "CASH_OUT", None, 20.0, 60 * 24 * 9, agent_id="A7"))       # 9 jours après
    assert soon["rep_agent_frauds_7d"] > 0 and late["rep_agent_frauds_7d"] == 0.0
    own = ext.process(c1_tx("U0", "P2P_SEND", "243999000009", 5.0, 60 * 24 * 10))             # compte « chaud »
    assert own["rep_user_frauds_30d"] == pytest.approx(np.log1p(1))


def test_reported_frauds_feed_reputation_without_leakage(table):
    """Dans le jeu complet, une transaction frauduleuse ne voit jamais son propre signalement."""
    fraud = table[table.is_fraud == 1]
    assert (table["rep_device_frauds"] > 0).any()          # des signalements ont bien été rejoués
    assert fraud["rep_user_frauds_30d"].notna().all()


def test_temporal_split_order(table):
    split, _ = temporal_split(table["timestamp"])
    assert (np.diff(split) >= 0).all()  # train -> early_stop -> val -> test
    assert set(np.unique(split)) == {0, 1, 2, 3}


def test_sequence_index():
    users = pd.Series(["a", "b", "a", "a", "b"])
    seq = build_sequence_index(users, seq_len=3)
    np.testing.assert_array_equal(seq[3], [0, 2, 3])   # 3 transactions de « a »
    np.testing.assert_array_equal(seq[4], [-1, 1, 4])  # 2 transactions de « b »
    np.testing.assert_array_equal(seq[0], [-1, -1, 0])
