"""Exposition Prometheus compatible avec plusieurs workers uvicorn.

Chaque worker est un processus distinct : sans mode multiprocessus, /metrics renverrait
les compteurs d'un seul worker, pris au hasard. Si PROMETHEUS_MULTIPROC_DIR est défini
(Dockerfiles), les valeurs de tous les workers sont agrégées."""
from __future__ import annotations

import os
import shutil

from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess


def reset_multiproc_dir() -> None:
    """À appeler une fois AVANT le lancement des workers (sinon d'anciens fichiers faussent les compteurs)."""
    d = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if d:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)


def metrics_response() -> Response:
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
