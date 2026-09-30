"""Lecture minimale du fichier .env pour les scripts lancés depuis Windows (hors Docker).

Les conteneurs reçoivent déjà leurs variables par docker compose ; les scripts locaux
(simulateur, seed, tests) doivent utiliser les MÊMES secrets que la plateforme, sinon ils
retombent sur les anciennes valeurs de démo, désormais refusées.
Les variables déjà présentes dans l'environnement ne sont jamais écrasées.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | str | None = None) -> None:
    p = Path(path) if path else ROOT / ".env"
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
