"""Tests de cohérence du générateur de données synthétiques."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "generator"))
from generate_synthetic_data import (  # noqa: E402
    CARD_FRAUD_MIX, MM_FRAUD_MIX, RELABELED_FRAUD_TYPES, TRANSACTION_COLUMNS, GeneratorConfig,
    SyntheticDataGenerator,
)

CFG = GeneratorConfig(n_users=800, n_days=90, seed=7, visa_adoption_multiplier=2.0)


@pytest.fixture(scope="module")
def data():
    return SyntheticDataGenerator(CFG).run()


def test_columns_and_ids(data):
    tx = data["transactions"]
    assert list(tx.columns) == TRANSACTION_COLUMNS
    assert tx["transaction_id"].is_unique
    assert data["users"]["user_id"].is_unique


def test_chronological_and_within_period(data):
    ts = data["transactions"]["timestamp"]
    assert ts.is_monotonic_increasing
    start = pd.Timestamp(CFG.start_date)
    assert ts.min() >= start and ts.max() < start + pd.Timedelta(days=CFG.n_days)


def test_all_fraud_types_present(data):
    """Toutes les typologies existent dans la VÉRITÉ terrain (certaines, comme le test de carte
    à 1-2 USD, sont rarement signalées : elles peuvent manquer dans les étiquettes observées)."""
    observed = set(data["transactions"]["fraud_type"].dropna())
    unreported = set(data["label_noise"]["true_fraud_type"].dropna())
    assert observed | unreported == set(MM_FRAUD_MIX) | set(CARD_FRAUD_MIX) | set(RELABELED_FRAUD_TYPES)


def test_label_noise_consistent(data):
    """Les lignes bruitées portent l'étiquette observée ; la vérité terrain est à part."""
    tx = data["transactions"].set_index("transaction_id")
    noise = data["label_noise"]
    assert set(noise["noise_type"]) == {"UNREPORTED_FRAUD", "FRIENDLY_FRAUD"}
    assert (tx.loc[noise.transaction_id, "is_fraud"].to_numpy() == noise.observed_label.to_numpy()).all()
    unrep = noise[noise.noise_type == "UNREPORTED_FRAUD"]
    assert tx.loc[unrep.transaction_id, "fraud_type"].isna().all()
    assert unrep.true_fraud_type.notna().all()
    friendly = noise[noise.noise_type == "FRIENDLY_FRAUD"]
    assert (tx.loc[friendly.transaction_id, "tx_type"] == "CARD_PURCHASE").all()


def test_fraud_rates_close_to_target(data):
    """Les taux cibles portent sur la VÉRITÉ terrain ; le taux observé est plus bas (sous-signalement)."""
    tx = data["transactions"].copy()
    noise = data["label_noise"].set_index("transaction_id")
    tx["true"] = tx["is_fraud"]
    known = tx.transaction_id.isin(noise.index)
    tx.loc[known, "true"] = tx.loc[known, "transaction_id"].map(noise["true_label"]).to_numpy()
    rates = tx.groupby("channel")["true"].mean()
    assert (tx.groupby("channel")["is_fraud"].mean() < rates).all()
    assert rates["MOBILE_MONEY"] == pytest.approx(CFG.mm_fraud_rate, rel=0.35)
    assert rates["VISA_VIRTUAL"] == pytest.approx(CFG.card_fraud_rate, rel=0.5)


def test_balances_consistent(data):
    tx = data["transactions"]
    assert (tx["balance_before_usd"] >= -0.01).all()
    assert (tx["balance_after_usd"] >= -0.01).all()
    failed = tx["status"] != "SUCCESS"
    assert np.allclose(tx.loc[failed, "balance_before_usd"], tx.loc[failed, "balance_after_usd"])
    assert (tx["amount_usd"] > 0).all()


def test_legit_failure_rate_realistic(data):
    tx = data["transactions"]
    legit_fail = (tx.loc[tx.is_fraud == 0, "status"] != "SUCCESS").mean()
    assert legit_fail < 0.05


def test_amounts_respect_kyc_limits(data):
    tx = data["transactions"].merge(data["users"][["user_id", "kyc_tx_limit_usd"]], on="user_id")
    assert (tx["amount_usd"] <= tx["kyc_tx_limit_usd"] + 0.01).all()


def test_no_label_leakage_in_identifiers(data):
    """Les identifiants ne doivent pas trahir la fraude (préfixes, formats)."""
    tx = data["transactions"]
    for col in ["transaction_id", "device_id", "counterparty_id", "merchant_id"]:
        vals = tx[col].dropna().astype(str)
        lengths_fraud = set(vals[tx.loc[vals.index, "is_fraud"] == 1].str.len())
        lengths_legit = set(vals[tx.loc[vals.index, "is_fraud"] == 0].str.len())
        assert lengths_fraud <= lengths_legit, col
    assert not tx["device_id"].str.contains("FRAUD|MULE", case=False).any()
    # les portefeuilles des mules ne se distinguent pas par leur préfixe
    p2p = tx[tx.tx_type.isin(["P2P_SEND", "P2P_RECEIVE"])]
    prefixes = p2p["counterparty_id"].str[:5]
    assert set(prefixes[p2p.is_fraud == 1]) <= set(prefixes[p2p.is_fraud == 0])


def test_reporting_is_partial_and_delayed(data):
    """Réaliste : une partie seulement des fraudes est signalée, toujours APRÈS les faits."""
    tx = data["transactions"]
    observed = tx[tx.is_fraud == 1]
    assert observed["fraud_reported_at"].notna().all()
    assert tx.loc[tx.is_fraud == 0, "fraud_reported_at"].isna().all()
    delay = (pd.to_datetime(observed.fraud_reported_at) - pd.to_datetime(observed.timestamp)).dt.total_seconds()
    assert (delay > 0).all() and delay.max() <= (CFG.report_delay_max_days + 1) * 86400
    unreported = (data["label_noise"].noise_type == "UNREPORTED_FRAUD").sum()
    share_reported = 1 - unreported / (unreported + (observed.fraud_type != "FRIENDLY_FRAUD").sum())
    assert 0.2 < share_reported < 0.6


def test_mules_are_reused_by_several_victims(data):
    tx, noise = data["transactions"], data["label_noise"]
    unrep = set(noise.loc[noise.noise_type == "UNREPORTED_FRAUD", "transaction_id"])
    sends = tx[(tx.tx_type == "P2P_SEND") & ((tx.is_fraud == 1) | tx.transaction_id.isin(unrep))]
    victims_per_mule = sends.groupby("counterparty_id").user_id.nunique()
    assert victims_per_mule.max() >= 3 and victims_per_mule.mean() > 1.2


def test_fraud_happens_after_warmup(data):
    """Les fraudes injectées laissent un historique minimal (la fraude amicale, simple
    contestation d'un achat légitime, peut survenir à tout moment)."""
    tx = data["transactions"]
    start = pd.Timestamp(CFG.start_date)
    injected = (tx.is_fraud == 1) & ~tx.fraud_type.isin(RELABELED_FRAUD_TYPES)
    first_fraud = tx.loc[injected, "timestamp"].min()
    assert first_fraud >= start + pd.Timedelta(days=min(14, CFG.n_days // 3))


def test_reproducible_with_seed():
    small = GeneratorConfig(n_users=100, n_days=30, seed=123)
    a = SyntheticDataGenerator(small).run()["transactions"]
    b = SyntheticDataGenerator(small).run()["transactions"]
    pd.testing.assert_frame_equal(a, b)
