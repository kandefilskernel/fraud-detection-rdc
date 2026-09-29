"""
Pseudonymisation des identifiants personnels (protection des données, plateforme mutualisée).

Numéro de téléphone (MSISDN), identifiant client, carte et IMEI sont remplacés par un
pseudonyme HMAC-SHA256 calculé avec une clé secrète :

    - déterministe : le même numéro donne toujours le même pseudonyme, donc les profils
      comportementaux et le réseau de contreparties (C1) fonctionnent à l'identique — le
      modèle n'utilise jamais la valeur d'un identifiant, seulement « est-ce le même ? » ;
    - non réversible sans la clé : la plateforme mutualisée ne voit aucun numéro en clair ;
      seul le détenteur de la clé (l'opérateur, ou un tiers de confiance) peut ré-identifier
      un client, par exemple pour l'appeler après une alerte.

La MÊME clé doit servir à l'amorçage du feature store et au trafic temps réel, sinon les
clients apparaissent inconnus. Changer de clé = ré-amorcer.
"""
from __future__ import annotations

import hashlib
import hmac

PREFIX = "P"                 # pseudonymes reconnaissables dans les journaux et le back-office
DIGEST_CHARS = 32            # 128 bits : collisions négligeables, tient dans String(64)
# Champs personnels du schéma pivot ; les identifiants d'agents et de marchands (entreprises)
# restent en clair, utiles aux analystes et non personnels.
PERSONAL_FIELDS = ("user_id", "wallet_id", "card_id", "device_id")
P2P_TYPES = ("P2P_SEND", "P2P_RECEIVE")   # la contrepartie n'est un portefeuille que pour un P2P


class Pseudonymizer:
    def __init__(self, key: str):
        if not key or len(key) < 16:
            raise ValueError("clé de pseudonymisation absente ou trop courte (16 caractères minimum)")
        self._key = key.encode()

    def __call__(self, value: str | None) -> str | None:
        if value is None or value == "":
            return value
        if value.startswith(PREFIX) and len(value) == len(PREFIX) + DIGEST_CHARS:
            return value                                  # déjà pseudonymisé (idempotent)
        return PREFIX + hmac.new(self._key, str(value).encode(), hashlib.sha256).hexdigest()[:DIGEST_CHARS]

    def transaction(self, tx: dict) -> dict:
        """Copie d'une transaction (dict au format pivot) avec identifiants pseudonymisés."""
        out = dict(tx)
        for f in PERSONAL_FIELDS:
            if f in out:
                out[f] = self(out[f])
        if out.get("tx_type") in P2P_TYPES and out.get("counterparty_id"):
            out["counterparty_id"] = self(out["counterparty_id"])
        return out
