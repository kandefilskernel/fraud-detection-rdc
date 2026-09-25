"""Tests de fumée de la phase 3 : les branches s'entraînent, l'ensemble se sauvegarde et se recharge."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "generator"))
from generate_synthetic_data import GeneratorConfig, SyntheticDataGenerator  # noqa: E402

from ml.features.feature_engineering import FEATURE_NAMES, build_feature_table  # noqa: E402
from ml.models.autoencoder_branch import train_autoencoder_branch  # noqa: E402
from ml.models.hybrid_ensemble import HybridEnsemble  # noqa: E402
from ml.models.lstm_attention_branch import LSTMAttentionBranch, gather_sequences, train_lstm_branch  # noqa: E402
from ml.models.meta_learner import build_meta_learner  # noqa: E402
from ml.models.xgboost_branch import train_xgboost_branch  # noqa: E402
from ml.preprocessing.train_test_split_normalize import build_sequence_index, temporal_split  # noqa: E402
from ml.training.evaluation import best_f1_threshold, evaluate, recall_at_fpr  # noqa: E402


@pytest.fixture(scope="module")
def prepared():
    cfg = GeneratorConfig(n_users=300, n_days=60, seed=5, visa_adoption_multiplier=2.0)
    raw = SyntheticDataGenerator(cfg).run()
    table = build_feature_table(raw["transactions"], raw["users"], pd.Timestamp(cfg.start_date))
    X = table[FEATURE_NAMES].to_numpy(np.float32)
    split, _ = temporal_split(table["timestamp"])
    mu, sd = X[split <= 1].mean(0), X[split <= 1].std(0) + 1e-6
    X = np.clip((X - mu) / sd, -10, 10).astype(np.float32)
    return X, table["is_fraud"].to_numpy(np.int8), build_sequence_index(table["user_id"], 10), split


def test_attention_ignores_padding():
    model = LSTMAttentionBranch(input_dim=4, hidden_dim=8)
    X = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    seq_idx = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.int32)
    xb, mb = gather_sequences(X, seq_idx, np.arange(3))
    _, weights = model(xb, mb, return_attention=True)
    assert np.allclose(weights[0, :2].detach().numpy(), 0.0)
    assert np.allclose(weights.sum(1).detach().numpy(), 1.0, atol=1e-5)


def test_metrics():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert recall_at_fpr(y, s, 0.01) == 1.0
    thr = best_f1_threshold(y, s)
    assert evaluate(y, s, thr)["f1"] == 1.0


def test_end_to_end_training_and_reload(prepared, tmp_path):
    X, y, seq_idx, split = prepared
    fit, es = np.where(split == 0)[0], np.where(split == 1)[0]
    val, test = np.where(split == 2)[0], np.where(split == 3)[0]

    xgb_model = train_xgboost_branch(X[fit], y[fit], X[es], y[es], n_estimators=50)
    ae = train_autoencoder_branch(X[fit][y[fit] == 0], X[es][y[es] == 0], epochs=2, log=lambda *_: None)
    lstm = train_lstm_branch(X, y, seq_idx, fit, es, epochs=1, hidden_dim=16, log=lambda *_: None)

    ens = HybridEnsemble(xgb_model, lstm, ae)
    S_val = ens.branch_scores(X, seq_idx, val)
    assert S_val.shape == (len(val), 3) and np.isfinite(S_val).all()
    ens.meta = build_meta_learner().fit(S_val, y[val])
    ens.threshold = 0.4
    p = ens.predict_proba(X, seq_idx, test)
    assert ((p >= 0) & (p <= 1)).all()

    ens.save(tmp_path)
    reloaded = HybridEnsemble.load(tmp_path)
    np.testing.assert_allclose(reloaded.predict_proba(X, seq_idx, test), p, rtol=1e-5, atol=1e-6)
    assert reloaded.threshold == 0.4
