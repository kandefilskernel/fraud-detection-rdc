"""Génère les tableaux de bord Grafana (JSON) : python infra/monitoring/grafana/build_dashboards.py"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
DS = {"type": "prometheus", "uid": "prometheus"}
_id = 0


def panel(title, exprs, x, y, w=12, h=8, kind="timeseries", unit=None, legends=None, thresholds=None,
          stack=False, decimals=None):
    global _id
    _id += 1
    targets = [{"datasource": DS, "expr": e, "legendFormat": (legends or [""] * len(exprs))[i], "refId": chr(65 + i)}
               for i, e in enumerate(exprs)]
    defaults = {"unit": unit} if unit else {}
    if decimals is not None:
        defaults["decimals"] = decimals
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in thresholds]}
        defaults["color"] = {"mode": "thresholds"}
    if kind == "timeseries":
        defaults["custom"] = {"fillOpacity": 18, "lineWidth": 2, "showPoints": "never",
                              "stacking": {"mode": "normal" if stack else "none"}}
    return {"id": _id, "type": kind, "title": title, "datasource": DS, "targets": targets,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "reduceOptions": {"calcs": ["lastNotNull"]}}}


def dashboard(uid, title, panels):
    return {"uid": uid, "title": title, "schemaVersion": 39, "version": 1, "refresh": "10s",
            "time": {"from": "now-30m", "to": "now"}, "timezone": "browser", "tags": ["fraude", "rdc"],
            "panels": panels}


P99 = 'histogram_quantile({q}, sum(rate(scoring_latency_seconds_bucket[1m])) by (le))'
system = dashboard("sante-systeme", "Santé du système", [
    panel("Services joignables", ["sum(up)"], 0, 0, 4, 4, "stat", thresholds=[(None, "red"), (10, "green")]),
    panel("Débit de scoring (tx/s)", ["sum(rate(scoring_latency_seconds_count[1m]))"], 4, 0, 4, 4, "stat", "reqps", decimals=1),
    panel("Latence p99 du scoring", [P99.format(q=0.99)], 8, 0, 4, 4, "stat", "s",
          thresholds=[(None, "green"), (0.1, "orange"), (0.25, "red")]),
    panel("Erreurs de scoring (5 min)", ["sum(increase(scoring_errors_total[5m]))"], 12, 0, 4, 4, "stat",
          thresholds=[(None, "green"), (1, "red")]),
    panel("Échecs de publication Kafka", ["max(scoring_kafka_publish_failures)"], 16, 0, 4, 4, "stat",
          thresholds=[(None, "green"), (1, "red")]),
    panel("Scores en mode dégradé (5 min)", ["sum(increase(scoring_degraded_total[5m]))"], 20, 0, 4, 4, "stat",
          thresholds=[(None, "green"), (1, "orange")]),
    panel("Latence du scoring", [P99.format(q=q) for q in (0.5, 0.95, 0.99)], 0, 4, 12, 8, unit="s",
          legends=["p50", "p95", "p99"]),
    panel("Latence bout-en-bout par opérateur (p95)",
          ['histogram_quantile(0.95, sum(rate(integration_latency_seconds_bucket[1m])) by (le, provider))'],
          12, 4, 12, 8, unit="s", legends=["{{provider}}"]),
    panel("Messages opérateurs par issue", ['sum(rate(integration_requests_total[1m])) by (outcome)'], 0, 12, 12, 8,
          unit="reqps", legends=["{{outcome}}"], stack=True),
    panel("Workers Kafka : messages traités", ['sum(rate(worker_messages_total{outcome="ok"}[1m])) by (worker)'],
          12, 12, 12, 8, unit="reqps", legends=["{{worker}}"]),
    panel("API back-office : requêtes par code", ['sum(rate(http_requests_total[1m])) by (status)'], 0, 20, 12, 8,
          unit="reqps", legends=["{{status}}"]),
    panel("Redis : mémoire et clés", ["redis_memory_used_bytes", "sum(redis_db_keys)"], 12, 20, 6, 8,
          legends=["mémoire", "clés"]),
    panel("PostgreSQL : connexions actives", ['sum(pg_stat_activity_count{state="active"})'], 18, 20, 6, 8),
])

model = dashboard("modele-derive", "Modèle, décisions et dérive", [
    panel("Alarme de dérive", ["max(drift_alarm)"], 0, 0, 4, 4, "stat", thresholds=[(None, "green"), (1, "red")]),
    panel("Variables comportementales en dérive", ["max(drift_features_drifted)"], 4, 0, 5, 4, "stat",
          thresholds=[(None, "green"), (1, "orange"), (5, "red")]),
    panel("PSI du score (prod / validation)", ["max(drift_score_psi)"], 9, 0, 5, 4, "stat", decimals=3,
          thresholds=[(None, "green"), (0.1, "orange"), (0.2, "red")]),
    panel("Taux d'alerte (VERIFY + BLOCK)",
          ['sum(rate(scoring_decisions_total{action!="APPROVE"}[5m])) / sum(rate(scoring_decisions_total[5m]))'],
          14, 0, 5, 4, "stat", "percentunit", thresholds=[(None, "green"), (0.03, "orange"), (0.05, "red")]),
    panel("Fenêtre analysée", ["max(drift_window_size)"], 19, 0, 5, 4, "stat"),
    panel("Décisions par action", ['sum(rate(scoring_decisions_total[1m])) by (action)'], 0, 4, 12, 8,
          unit="reqps", legends=["{{action}}"], stack=True),
    panel("Niveaux de risque", ['sum(rate(scoring_risk_levels_total[1m])) by (risk_level)'], 12, 4, 12, 8,
          unit="reqps", legends=["{{risk_level}}"], stack=True),
    panel("Décisions par canal", ['sum(rate(scoring_decisions_total{action!="APPROVE"}[5m])) by (channel)'],
          0, 12, 12, 8, unit="reqps", legends=["{{channel}}"]),
    panel("Distribution des probabilités (part des scores < seuil)",
          ['sum(rate(scoring_fraud_probability_bucket[5m])) by (le) / scalar(sum(rate(scoring_fraud_probability_count[5m])))'],
          12, 12, 12, 8, unit="percentunit", legends=["p ≤ {{le}}"]),
    panel("KS par variable (top 10)", ["topk(10, drift_ks_statistic)"], 0, 20, 12, 9, "bargauge",
          legends=["{{feature}}"], thresholds=[(None, "green"), (0.1, "orange"), (0.2, "red")], decimals=3),
    panel("PSI par variable (top 10)", ["topk(10, drift_psi)"], 12, 20, 12, 9, "bargauge",
          legends=["{{feature}}"], thresholds=[(None, "green"), (0.1, "orange"), (0.2, "red")], decimals=3),
])

OUT.mkdir(exist_ok=True)
for d in (system, model):
    (OUT / f"{d['uid']}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    print("écrit", OUT / f"{d['uid']}.json")
