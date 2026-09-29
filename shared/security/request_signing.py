"""
Signature HMAC des messages opérateurs — implémentation de référence, partagée entre la
plateforme (vérification) et les clients (simulateur, tests, kit d'intégration opérateur).

Chaque opérateur reçoit un secret distinct de sa clé d'API. Il signe CHAQUE requête :

    chaîne canonique = horodatage + "\\n" + MÉTHODE + "\\n" + chemin + "\\n" + SHA-256(corps en hex)
    X-Timestamp      = horodatage Unix en secondes
    X-Signature      = HMAC-SHA256(secret, chaîne canonique), en hexadécimal

Garanties :
    - intégrité : un seul octet du corps modifié -> signature invalide ;
    - authenticité : sans le secret, impossible de fabriquer une signature ;
    - anti-rejeu : la requête n'est acceptée que ±window_s secondes autour de l'heure du
      serveur ; un rejeu dans la fenêtre renvoie la décision déjà prise (idempotence du scoring).
Le chemin est signé pour qu'une signature valable sur /v1/feedback ne le soit pas sur
/v1/transactions.
"""
from __future__ import annotations

import hashlib
import hmac
import time

DEFAULT_WINDOW_S = 300


class SignatureError(ValueError):
    """Signature absente, invalide ou expirée (le message indique la cause)."""


def canonical_string(timestamp: str, method: str, path: str, body: bytes) -> bytes:
    return "\n".join([timestamp, method.upper(), path, hashlib.sha256(body).hexdigest()]).encode()


def sign(secret: str, timestamp: str, method: str, path: str, body: bytes) -> str:
    return hmac.new(secret.encode(), canonical_string(timestamp, method, path, body), hashlib.sha256).hexdigest()


def signed_headers(secret: str, method: str, path: str, body: bytes, now: float | None = None) -> dict:
    """En-têtes à ajouter à une requête sortante (côté opérateur)."""
    ts = str(int(now if now is not None else time.time()))
    return {"X-Timestamp": ts, "X-Signature": sign(secret, ts, method, path, body)}


def verify(secret: str, timestamp: str | None, signature: str | None, method: str, path: str, body: bytes,
           window_s: int = DEFAULT_WINDOW_S, now: float | None = None) -> None:
    if not timestamp or not signature:
        raise SignatureError("en-têtes X-Timestamp et X-Signature obligatoires")
    try:
        ts = int(timestamp)
    except ValueError:
        raise SignatureError("X-Timestamp doit être un horodatage Unix en secondes") from None
    now = time.time() if now is None else now
    if abs(now - ts) > window_s:
        raise SignatureError(f"horodatage hors de la fenêtre de ±{window_s} s (horloge décalée ou rejeu)")
    expected = sign(secret, timestamp, method, path, body)
    if not hmac.compare_digest(expected, signature.lower()):   # comparaison à temps constant
        raise SignatureError("signature invalide")


def parse_secrets(spec: str) -> dict[str, str]:
    """'vodacom:secret1,airtel:secret2' -> {'vodacom': 'secret1', 'airtel': 'secret2'}"""
    out = {}
    for pair in (spec or "").split(","):
        if ":" in pair:
            provider, secret = pair.split(":", 1)
            out[provider.strip()] = secret.strip()
    return out
