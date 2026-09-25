"""Chaînage par hachage du journal d'audit : chaque entrée contient le hachage de la
précédente. Modifier ou supprimer une ligne casse la chaîne, ce que /audit/verify détecte."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

GENESIS = "0" * 64


def entry_hash(prev_hash: str, ts: datetime, actor: str, action: str, entity: str,
               entity_id: str | None, details: dict) -> str:
    canonical = json.dumps({"prev": prev_hash, "ts": ts.isoformat(), "actor": actor, "action": action,
                            "entity": entity, "entity_id": entity_id, "details": details},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_chain(rows) -> dict:
    """rows : entrées triées par id. Renvoie la première rupture éventuelle."""
    prev, n = GENESIS, 0
    for r in rows:
        expected = entry_hash(prev, r.ts, r.actor, r.action, r.entity, r.entity_id, r.details)
        if r.prev_hash != prev or r.hash != expected:
            return {"valid": False, "checked": n, "broken_at_id": r.id}
        prev, n = r.hash, n + 1
    return {"valid": True, "checked": n, "broken_at_id": None}
