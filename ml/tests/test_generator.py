"""Tests de cohérence du générateur de données synthétiques."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "generator"))
from generate_synthetic_data import (  # noqa: E402
    CARD_FRAUD_MIX, MM_FRAUD_MIX, TRANSACTION_COLUMNS, GeneratorConfig, SyntheticDataGenerator,
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
    present = set(data["transactions"]["fraud_type"].dropna())
    assert present == set(MM_FRAUD_MIX) | set(CARD_FRAUD_MIX)


def test_fraud_rates_close_to_target(data):
    tx = data["transactions"]
    rates = tx.groupby("channel")["is_fraud"].mean()
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


def test_fraud_happens_after_warmup(data):
    tx = data["transactions"]
    start = pd.Timestamp(CFG.start_date)
    first_fraud = tx.loc[tx.is_fraud == 1, "timestamp"].min()
    assert first_fraud >= start + pd.Timedelta(days=min(14, CFG.n_days // 3))


def test_reproducible_with_seed():
    small = GeneratorConfig(n_users=100, n_days=30, seed=123)
    a = SyntheticDataGenerator(small).run()["transactions"]
    b = SyntheticDataGenerator(small).run()["transactions"]
    pd.testing.assert_frame_equal(a, b)
