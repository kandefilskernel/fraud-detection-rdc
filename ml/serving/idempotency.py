"""
Idempotence du scoring : une transaction renvoyée par l'opérateur (coupure réseau, délai
dépassé, reprise après incident) ne doit être comptée qu'UNE fois dans le profil du client.

Pour chaque transaction, une clé Redis idem:{canal}:{opérateur}:{transaction_id} :

    1er envoi      SET NX « en cours » (verrou court) -> scoring -> on stocke la décision
    renvoi         même contenu  -> la décision d'origine est renvoyée, profil inchangé
                   contenu différent (même identifiant, autre montant...) -> refus (409) :
                   réutilisation d'identifiant ou message falsifié
    envoi simultané (le 1er n'a pas fini) -> on attend brièvement, sinon 409 « en cours »

Si le scoring échoue, la clé est libérée : l'opérateur peut renvoyer la transaction.
Les décisions sont conservées ttl_s (48 h par défaut), bien au-delà des délais de reprise
des opérateurs.
"""
from __future__ import annotations

import hashlib
import json
import time

PREFIX = "idem"
PENDING = "__en_cours__"


class IdempotencyConflict(Exception):
    """Même identifiant de transaction, contenu différent."""


class IdempotencyInProgress(Exception):
    """La même transaction est en cours de traitement (envoi simultané)."""


def fingerprint(payload: dict) -> str:
    """Empreinte du contenu métier (clés triées, indépendante de l'ordre des champs)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class IdempotencyStore:
    def __init__(self, redis_client, ttl_s: int = 48 * 3600, pending_ttl_s: int = 30, wait_s: float = 1.0):
        self.r = redis_client
        self.ttl_s = ttl_s
        self.pending_ttl_s = pending_ttl_s
        self.wait_s = wait_s

    @staticmethod
    def key(channel: str, operator: str | None, transaction_id: str) -> str:
        return f"{PREFIX}:{channel}:{operator or '-'}:{transaction_id}"

    def begin(self, key: str, fp: str) -> dict | None:
        """None : première réception, à traiter. dict : décision déjà rendue, à renvoyer."""
        if self.r.set(key, json.dumps({"state": PENDING, "fp": fp}), nx=True, ex=self.pending_ttl_s):
            return None
        deadline = time.monotonic() + self.wait_s
        while True:
            raw = self.r.get(key)
            if raw is None:                      # le 1er traitement a échoué et libéré la clé
                if self.r.set(key, json.dumps({"state": PENDING, "fp": fp}), nx=True, ex=self.pending_ttl_s):
                    return None
                continue
            entry = json.loads(raw)
            if entry["fp"] != fp:
                raise IdempotencyConflict("identifiant de transaction déjà utilisé avec un contenu différent")
            if entry["state"] != PENDING:
                return entry["result"]
            if time.monotonic() > deadline:
                raise IdempotencyInProgress("transaction en cours de traitement, réessayer")
            time.sleep(0.01)

    def complete(self, key: str, fp: str, result: dict) -> None:
        self.r.set(key, json.dumps({"state": "done", "fp": fp, "result": result}, default=str), ex=self.ttl_s)

    def abort(self, key: str) -> None:
        self.r.delete(key)
