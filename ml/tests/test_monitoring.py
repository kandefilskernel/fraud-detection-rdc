"""Détection de dérive : PSI, KS et familles de variables."""
import numpy as np

from ml.features.feature_engineering import FEATURE_NAMES
from ml.monitoring.drift_detector import compare, feature_family, psi


def _reference(rng, n=5000):
    return {"features": rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32),
            "scores": rng.beta(0.3, 20, size=n).astype(np.float32),
            "feature_names": np.array(FEATURE_NAMES)}


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=20000), rng.normal(size=20000)) < 0.01


def test_psi_detects_a_shift():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(size=20000), rng.normal(loc=1.0, size=20000)) > 0.2


def test_psi_handles_binary_features():
    rng = np.random.default_rng(2)
    ref = (rng.random(10000) < 0.05).astype(float)
    assert psi(ref, (rng.random(10000) < 0.05).astype(float)) < 0.02
    assert psi(ref, (rng.random(10000) < 0.40).astype(float)) > 0.2


def test_no_alarm_without_drift():
    rng = np.random.default_rng(3)
    ref = _reference(rng)
    cur = _reference(np.random.default_rng(4), n=2000)
    rep = compare(ref, cur["features"], cur["scores"])
    assert rep["alarm"] is False and rep["n_drifted"] == 0


def test_behavioral_drift_raises_alarm_but_calendar_drift_does_not():
    rng = np.random.default_rng(5)
    ref = _reference(rng)
    cur = _reference(np.random.default_rng(6), n=2000)
    j_amount = FEATURE_NAMES.index("amount_to_balance")
    j_hour = FEATURE_NAMES.index("hour_sin")
    cur["features"][:, j_hour] += 2.0          # fenêtre courte : heures concentrées
    rep = compare(ref, cur["features"], cur["scores"])
    assert rep["alarm"] is False and "hour_sin" in rep["expected_drift"]
    cur["features"][:, j_amount] += 1.5        # vrai changement de comportement
    rep = compare(ref, cur["features"], cur["scores"])
    assert rep["alarm"] is True and rep["drifted_features"] == ["amount_to_balance"]


def test_feature_families():
    assert feature_family("hour_cos") == "calendaire"
    assert feature_family("log_user_tx_count") == "cumulative"
    assert feature_family("is_new_device") == "comportementale"
