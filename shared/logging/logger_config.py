"""Journalisation structurée (une ligne JSON par événement), commune à tous les services.
Les lignes sont collectées par Promtail et interrogeables dans Grafana (source Loki),
par exemple : {service="scoring-service"} | json | level="ERROR"."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        entry = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
                 "level": record.levelname, "service": self.service, "logger": record.name,
                 "msg": record.getMessage()}
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        for k in ("transaction_id", "user_id", "action", "latency_ms"):
            if hasattr(record, k):
                entry[k] = getattr(record, k)
        return json.dumps(entry, ensure_ascii=False)


class _DropProbes(logging.Filter):
    """Écarte les appels de supervision (/health, /metrics) : ils noieraient les vrais événements."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ('"GET /health' in msg or '"GET /metrics' in msg)


def configure_logging(service: str) -> None:
    """LOG_FORMAT=json (défaut en conteneur) ou text (développement)."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_DropProbes())
    if os.getenv("LOG_FORMAT", "text") == "json":
        handler.setFormatter(JsonFormatter(service))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True


def get_logger(service_name: str) -> logging.Logger:
    return logging.getLogger(service_name)
