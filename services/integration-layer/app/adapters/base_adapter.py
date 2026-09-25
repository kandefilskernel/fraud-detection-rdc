"""
Patron Adapter : chaque opérateur envoie son format propriétaire ; son adaptateur le
convertit vers le schéma pivot UnifiedTransaction. Ajouter un opérateur = écrire un
adaptateur, sans toucher au scoring ni au modèle.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from shared.schemas.unified_transaction import UnifiedTransaction

# Taux de change de référence (à alimenter en production par le cours BCC du jour)
DEFAULT_USD_CDF_RATE = 2850.0


class AdapterError(ValueError):
    """Message opérateur invalide ou incomplet."""


class BaseAdapter(ABC):
    provider: str = ""

    def __init__(self, usd_cdf_rate: float = DEFAULT_USD_CDF_RATE):
        self.rate = usd_cdf_rate

    def to_usd(self, amount: float, currency: str) -> float:
        currency = currency.upper()
        if currency == "USD":
            return round(float(amount), 2)
        if currency == "CDF":
            return round(float(amount) / self.rate, 2)
        raise AdapterError(f"devise non prise en charge : {currency}")

    @staticmethod
    def parse_time(value: str, fmt: str | None = None) -> datetime:
        try:
            return datetime.strptime(value, fmt) if fmt else datetime.fromisoformat(value)
        except (TypeError, ValueError) as e:
            raise AdapterError(f"horodatage invalide : {value!r}") from e

    @staticmethod
    def require(payload: dict, *keys: str):
        missing = [k for k in keys if payload.get(k) in (None, "")]
        if missing:
            raise AdapterError(f"champs obligatoires manquants : {', '.join(missing)}")

    @abstractmethod
    def to_unified(self, payload: dict) -> UnifiedTransaction:
        ...

    @abstractmethod
    def from_unified(self, tx: dict) -> dict:
        """Opération inverse (format opérateur) : utilisée par le simulateur de démo."""
