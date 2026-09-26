"""
Détection de dérive du modèle en production.

Compare la fenêtre récente de transactions scorées (table scored_transactions : variables
et probabilités réellement vues par le modèle) à la distribution de référence :

    - test de Kolmogorov-Smirnov à deux échantillons, variable par variable ;
    - PSI (Population Stability Index), variables et score du modèle.

Une variable est « en dérive » si (D_KS > 0,1 et p < 0,01) ou PSI > 0,2. Le KS seul est
trop sensible sur de grands échantillons (p minuscule pour un écart négligeable) : on
combine donc significativité statistique ET taille d'effet.

Mode service (--serve) : calcul toutes les INTERVAL secondes, exposé à Prometheus
(port 9101) ; Grafana affiche les courbes et Prometheus déclenche l'alerte de dérive.
Mode ponctuel (--once) : rapport JSON (pour le mémoire).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
from scipy.stats import ks_2samp

KS_D, KS_P, PSI_MAX = 0.10, 0.01, 0.20
log = logging.getLogger("drift")

# Variables exclues de l'alarme, mais toujours mesurées et affichées :
#  - calendaires : une fenêtre de quelques heures n'a jamais la distribution horaire
#    d'une période de plusieurs mois ; leur écart dépend de la fenêtre, pas du modèle ;
#  - cumulatives : elles croissent mécaniquement avec l'ancienneté des profils
#    (historique, nombre d'opérations, d'appareils...) : dérive structurelle attendue.
CALENDAR = {"hour_sin", "hour_cos", "is_night", "is_weekend", "is_month_end"}
CUMULATIVE = {"log_history_days", "log_user_tx_count", "tx_count_7d", "log_device_prior_uses",
              "n_devices_user", "counterparty_n_users", "log_account_age_days", "user_night_ratio",
              "log_mins_since_topup"}


def feature_family(name: str) -> str:
    return "calendaire" if name in CALENDAR else "cumulative" if name in CUMULATIVE else "comportementale"


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI sur les quantiles de la référence. Variables discrètes (binaires, compteurs à
    peu de valeurs) : une classe par valeur, sinon les quantiles se confondent en une
    seule classe et le PSI resterait nul même en cas de forte dérive."""
    values = np.unique(ref)
    if len(values) <= bins:
        cats = np.union1d(values, np.unique(cur))
        r = np.array([(ref == v).mean() for v in cats])
        c = np.array([(cur == v).mean() for v in cats]) if len(cur) else np.zeros(len(cats))
        r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
        return float(np.sum((c - r) * np.log(c / r)))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def compare(reference: dict, cur_features: np.ndarray, cur_scores: np.ndarray) -> dict:
    names = [str(n) for n in reference["feature_names"]]
    ref = reference["features"]
    per_feature = {}
    for j, name in enumerate(names):
        d, p = ks_2samp(ref[:, j], cur_features[:, j])
        v = psi(ref[:, j], cur_features[:, j])
        per_feature[name] = {"ks_d": round(float(d), 4), "ks_p": float(p), "psi": round(v, 4),
                             "family": feature_family(name),
                             "drift": bool((d > KS_D and p < KS_P) or v > PSI_MAX)}
    sd, sp = ks_2samp(reference["scores"], cur_scores)
    score = {"ks_d": round(float(sd), 4), "ks_p": float(sp), "psi": round(psi(reference["scores"], cur_scores), 4)}
    # alarme : variables comportementales en dérive, ou score du modèle déplacé
    drifted = sorted(n for n, r in per_feature.items() if r["drift"] and r["family"] == "comportementale")
    expected = sorted(n for n, r in per_feature.items() if r["drift"] and r["family"] != "comportementale")
    return {"window": int(len(cur_scores)), "n_drifted": len(drifted), "drifted_features": drifted,
            "expected_drift": expected, "score": score,
            "alarm": bool(drifted) or score["psi"] > PSI_MAX, "features": per_feature}


def fetch_window(engine, window: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from sqlalchemy import text
    with engine.connect() as c:
        rows = c.execute(text("SELECT features, fraud_probability FROM scored_transactions "
                              "ORDER BY scored_at DESC LIMIT :n"), {"n": window}).all()
    return rows


def to_arrays(rows, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    feats = np.array([[r[0].get(n, np.nan) for n in names] for r in rows], dtype=np.float32)
    scores = np.array([r[1] for r in rows], dtype=np.float32)
    return feats, scores


def serve(reference: dict, window: int, interval: int, min_window: int, port: int,
          reference_path: str | None = None) -> None:
    from prometheus_client import Gauge, start_http_server
    from shared.database.session import make_session_factory

    g_ks = Gauge("drift_ks_statistic", "Statistique D de Kolmogorov-Smirnov", ["feature"])
    g_psi = Gauge("drift_psi", "Population Stability Index", ["feature"])
    g_n = Gauge("drift_features_drifted", "Variables comportementales en dérive (alarme)")
    g_alarm = Gauge("drift_alarm", "1 si dérive significative (variables comportementales ou score)")
    g_score_ks = Gauge("drift_score_ks_statistic", "KS du score de fraude (prod vs validation)")
    g_score_psi = Gauge("drift_score_psi", "PSI du score de fraude")
    g_window = Gauge("drift_window_size", "Taille de la fenêtre analysée")
    g_last = Gauge("drift_last_run_timestamp", "Horodatage du dernier calcul")
    start_http_server(port)
    engine, _ = make_session_factory()
    names = [str(n) for n in reference["feature_names"]]
    ref_mtime = os.path.getmtime(reference_path) if reference_path else None
    log.info("détecteur de dérive démarré : fenêtre=%d, intervalle=%ds", window, interval)
    while True:
        try:
            # après un réentraînement promu, la référence est remplacée : on la recharge
            if reference_path and os.path.getmtime(reference_path) != ref_mtime:
                reference = dict(np.load(reference_path, allow_pickle=False))
                ref_mtime = os.path.getmtime(reference_path)
                log.info("nouvelle distribution de référence chargée")
            rows = fetch_window(engine, window)
            g_window.set(len(rows))
            if len(rows) >= min_window:
                feats, scores = to_arrays(rows, names)
                rep = compare(reference, feats, scores)
                for n, r in rep["features"].items():
                    g_ks.labels(n).set(r["ks_d"])
                    g_psi.labels(n).set(r["psi"])
                g_n.set(rep["n_drifted"])
                g_alarm.set(1 if rep["alarm"] else 0)
                g_score_ks.set(rep["score"]["ks_d"])
                g_score_psi.set(rep["score"]["psi"])
                g_last.set(time.time())
                log.info("fenêtre=%d dérive=%d %s score_psi=%.3f", rep["window"], rep["n_drifted"],
                         rep["drifted_features"][:6], rep["score"]["psi"])
            else:
                log.info("fenêtre trop petite (%d < %d)", len(rows), min_window)
        except Exception:  # noqa: BLE001 — base momentanément indisponible
            log.exception("échec du calcul de dérive")
        time.sleep(interval)


def main():
    from shared.logging.logger_config import configure_logging
    configure_logging("drift-monitor")
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default=os.getenv("DRIFT_REFERENCE", "ml/artifacts/drift_reference.npz"))
    p.add_argument("--window", type=int, default=int(os.getenv("DRIFT_WINDOW", "2000")))
    p.add_argument("--min-window", type=int, default=500)
    p.add_argument("--interval", type=int, default=int(os.getenv("DRIFT_INTERVAL_S", "60")))
    p.add_argument("--port", type=int, default=9101)
    p.add_argument("--serve", action="store_true", help="service continu (Prometheus)")
    p.add_argument("--once", action="store_true", help="rapport JSON unique (défaut)")
    a = p.parse_args()
    reference = dict(np.load(a.reference, allow_pickle=False))
    if a.serve:
        serve(reference, a.window, a.interval, a.min_window, a.port, a.reference)
    else:
        from shared.database.session import make_session_factory
        engine, _ = make_session_factory()
        feats, scores = to_arrays(fetch_window(engine, a.window), [str(n) for n in reference["feature_names"]])
        rep = compare(reference, feats, scores)
        rep["features"] = {k: v for k, v in rep["features"].items() if v["drift"]}
        print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
