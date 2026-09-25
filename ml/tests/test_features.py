"""Tests de la phase 2 : variables comportementales, split temporel, séquences."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "generator"))
from generate_synthetic_data import GeneratorConfig, SyntheticDataGenerator  # noqa: E402

from ml.features.feature_engineering import FEATURE_NAMES, build_feature_table  # noqa: E402
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
    """Sanity check : les signaux clés sont plus fréquents dans les fraudes."""
    fraud, legit = table[table.is_fraud == 1], table[table.is_fraud == 0]
    assert fraud["is_new_device"].mean() > 3 * legit["is_new_device"].mean()
    assert fraud["amount_to_balance"].mean() > legit["amount_to_balance"].mean()


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
