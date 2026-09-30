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


def test_compare_runs_detects_a_better_model(tmp_path):
    """Un modèle B nettement meilleur que A : écart positif et significatif."""
    import json
    from ml.training.compare_runs import compare
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.05).astype(int)
    base = pd.DataFrame({"transaction_id": [f"T{i}" for i in range(n)], "user_id": [f"U{i % 400}" for i in range(n)],
                         "channel": np.where(rng.random(n) < 0.8, "MOBILE_MONEY", "VISA_VIRTUAL"),
                         "is_fraud": y, "fraud_type": np.where(y == 1, "SIM_SWAP", None)})
    for name, noise in [("a", 1.5), ("b", 0.3)]:
        d = tmp_path / name
        d.mkdir()
        base.assign(score_hybrid=y + rng.normal(0, noise, n)).to_csv(d / "test_predictions.csv", index=False)
        (d / "phase3_results.json").write_text(json.dumps({"results": {"hybrid": {"threshold": 0.5}}}))
    res = compare(tmp_path / "a", tmp_path / "b", n_boot=100)
    assert res["metrics"]["pr_auc_global"]["delta"] > 0
    assert res["metrics"]["pr_auc_global"]["significant"]
    assert res["mcnemar"]["a_wrong_b_right"] > res["mcnemar"]["a_right_b_wrong"]


def test_card_history_segments(tmp_path):
    from ml.training.compare_runs import card_history_segments
    ts = pd.date_range("2025-07-01", periods=25, freq="h")
    tx = pd.DataFrame({"transaction_id": [f"T{i}" for i in range(25)], "timestamp": ts,
                       "user_id": "U1", "channel": ["MOBILE_MONEY"] + ["VISA_VIRTUAL"] * 24})
    tx.to_csv(tmp_path / "tx.csv", index=False)
    seg = card_history_segments(tmp_path / "tx.csv")
    assert "T0" not in seg.index                              # Mobile Money : pas de segment carte
    assert seg["T1"] == "carte_0-4_tx" and seg["T5"] == "carte_0-4_tx"
    assert seg["T6"] == "carte_5-19_tx" and seg["T21"] == "carte_20+_tx"


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


def test_ensemble_without_autoencoder_roundtrip(prepared, tmp_path):
    """Ensemble sans autoencodeur (XGBoost + LSTM). La liste des branches voyage avec les
    artefacts (metadata.json -> branch_order)."""
    X, y, seq_idx, split = prepared
    fit, es = np.where(split == 0)[0], np.where(split == 1)[0]
    val, test = np.where(split == 2)[0], np.where(split == 3)[0]
    xgb_model = train_xgboost_branch(X[fit], y[fit], X[es], y[es], n_estimators=50)
    lstm = train_lstm_branch(X, y, seq_idx, fit, es, epochs=1, hidden_dim=16, log=lambda *_: None)

    ens = HybridEnsemble(xgb_model, lstm)
    assert ens.branches == ["xgboost", "lstm_attention"]
    S_val = ens.branch_scores(X, seq_idx, val)
    assert S_val.shape == (len(val), 2)
    ens.meta = build_meta_learner().fit(S_val, y[val])
    p = ens.predict_proba(X, seq_idx, test)

    (tmp_path / "autoencoder_branch.pt").write_bytes(b"ancien")      # reste d'un ancien modèle
    ens.save(tmp_path)
    assert not (tmp_path / "autoencoder_branch.pt").exists()
    reloaded = HybridEnsemble.load(tmp_path)
    assert reloaded.ae is None and reloaded.branches == ["xgboost", "lstm_attention"]
    np.testing.assert_allclose(reloaded.predict_proba(X, seq_idx, test), p, rtol=1e-5, atol=1e-6)


def test_tree_only_with_anomaly_detector_roundtrip(prepared, tmp_path):
    """Sélection finale possible : arbres seuls (sans LSTM) + autoencodeur en veille des
    anomalies, hors méta-apprenant. La veille ne change pas la probabilité."""
    from ml.models.autoencoder_branch import ae_score
    X, y, seq_idx, split = prepared
    fit, es = np.where(split == 0)[0], np.where(split == 1)[0]
    val, test = np.where(split == 2)[0], np.where(split == 3)[0]
    xgb_model = train_xgboost_branch(X[fit], y[fit], X[es], y[es], n_estimators=50)
    ae = train_autoencoder_branch(X[fit][y[fit] == 0], X[es][y[es] == 0], epochs=2, log=lambda *_: None)

    ens = HybridEnsemble(xgb_model, None, None)
    assert ens.branches == ["xgboost"] and ens.input_dim == X.shape[1]
    S_val = ens.branch_scores(X, seq_idx, val)
    ens.meta = build_meta_learner().fit(S_val, y[val])
    p_without = ens.predict_proba(X, seq_idx, test)
    thr = float(np.quantile(ae_score(ae, X[val])[y[val] == 0], 0.995))
    ens.anomaly, ens.anomaly_threshold, ens.anomaly_info = ae, thr, {"quantile": 0.995}
    np.testing.assert_allclose(ens.predict_proba(X, seq_idx, test), p_without)   # aucun effet sur P(fraude)

    for stale in ("lstm_branch.pt", "autoencoder_branch.pt"):
        (tmp_path / stale).write_bytes(b"ancien")
    ens.save(tmp_path)
    assert not (tmp_path / "lstm_branch.pt").exists() and not (tmp_path / "autoencoder_branch.pt").exists()
    reloaded = HybridEnsemble.load(tmp_path)
    assert reloaded.lstm is None and reloaded.ae is None and reloaded.branches == ["xgboost"]
    assert reloaded.anomaly_threshold == thr and reloaded.anomaly_info == {"quantile": 0.995}
    np.testing.assert_allclose(reloaded.predict_proba(X, seq_idx, test), p_without, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(reloaded.anomaly_scores(X, test), ae_score(ae, X[test]), rtol=1e-5, atol=1e-6)

    reloaded.anomaly, reloaded.anomaly_threshold = None, None      # retirer la veille : fichier supprimé
    reloaded.save(tmp_path)
    assert not (tmp_path / "anomaly_autoencoder.pt").exists()
    assert HybridEnsemble.load(tmp_path).anomaly is None


def test_selection_rule_and_holm():
    """Règle de sélection finale (validation seule) et correction de Holm."""
    from ml.training.select_final import apply_rule, holm, paired_bootstrap

    def row(n, obs, ver, eligible=True, b2=True):
        return {"n_components": n, "val": {"observe": obs, "verite": ver}, "eligible": eligible, "b2_ok": b2}

    rows = {"Forêt seule": row(1, 0.509, 0.930), "Forêt + LSTM": row(2, 0.506, 0.938),
            "Forêt + LSTM + AE": row(3, 0.468, 0.939), "XGB seul": row(1, 0.521, 0.878, b2=False),
            "Lent": row(1, 0.9, 0.99, eligible=False)}
    r = apply_rule(rows, "observe")
    assert r["eligibles"] == ["Forêt seule", "Forêt + LSTM", "Forêt + LSTM + AE"]   # B2-B4 filtrent
    assert r["a_egalite"] == ["Forêt seule", "Forêt + LSTM"] and r["choix"] == "Forêt seule"   # B6
    rv = apply_rule(rows, "verite")                        # 0,938 et 0,939 à égalité : le plus simple
    assert rv["a_egalite"] == ["Forêt + LSTM", "Forêt + LSTM + AE"] and rv["choix"] == "Forêt + LSTM"

    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}

    rng = np.random.default_rng(1)
    yb = (rng.random(3000) < 0.05).astype(int)
    users = np.array([f"U{i % 300}" for i in range(3000)])
    good, bad = yb + rng.normal(0, 0.3, 3000), yb + rng.normal(0, 2.0, 3000)
    res = paired_bootstrap(yb, good, bad, users, n_boot=200)
    assert res["delta"] > 0 and res["ci95"][0] > 0 and res["p_value"] < 0.05


def test_flat_forest_matches_sklearn_and_explains_exactly():
    """Forêt aplatie (temps réel) : mêmes probabilités que scikit-learn, et contributions
    additives exactes (biais + somme = probabilité)."""
    from sklearn.ensemble import RandomForestClassifier

    from ml.models.flat_forest import FlatForest
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 8)).astype(np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 3] ** 2 + rng.normal(scale=0.5, size=3000)) > 1.5).astype(int)
    rf = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, class_weight="balanced_subsample",
                                random_state=0).fit(X, y)
    flat = FlatForest(rf)
    ref = rf.predict_proba(X[:200])[:, 1]
    got = np.array([flat.predict_proba_one(x) for x in X[:200]])
    np.testing.assert_allclose(got, ref, atol=1e-9)
    for x in X[:20]:
        p, contrib = flat.explain_one(x)
        assert abs(flat.bias + contrib.sum() - p) < 1e-9
    assert np.argmax(np.abs(flat.explain_one(X[np.argmax(ref)])[1])) in (0, 3)   # variables utiles
