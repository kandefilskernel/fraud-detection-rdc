"""
Schéma pivot : format unique d'une transaction, quel que soit l'opérateur d'origine
(Vodacom M-Pesa, Airtel Money, Orange Money, carte Visa virtuelle).

Les adaptateurs de l'integration-layer convertissent chaque format propriétaire vers ce
schéma ; le scoring-service ne connaît que lui. Les noms de champs sont ceux du jeu de
données d'entraînement (ml/data/raw/transactions.csv), ce qui garantit que le modèle voit
en production exactement les mêmes informations qu'à l'entraînement.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# Heure légale de Kinshasa (UTC+1) : le modèle a été entraîné en heure locale RDC.
KINSHASA_TZ = timezone(timedelta(hours=1))


ID_FIELDS = ("user_id", "wallet_id", "card_id", "counterparty_id", "merchant_id", "agent_id", "device_id")


def canonical_id(v) -> str | None:
    """Identifiant sous forme texte canonique. Les numéros de portefeuille (MSISDN) lus par
    pandas deviennent des flottants (243917538864.0) : on les ramène à '243917538864' pour
    que l'historique rejoué et les transactions reçues par l'API désignent la même entité."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


class Channel(str, Enum):
    MOBILE_MONEY = "MOBILE_MONEY"
    VISA_VIRTUAL = "VISA_VIRTUAL"


class TxType(str, Enum):
    P2P_SEND = "P2P_SEND"
    P2P_RECEIVE = "P2P_RECEIVE"
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    MERCHANT_PAYMENT = "MERCHANT_PAYMENT"
    AIRTIME = "AIRTIME"
    BILL_PAYMENT = "BILL_PAYMENT"
    CARD_TOPUP = "CARD_TOPUP"
    CARD_PURCHASE = "CARD_PURCHASE"


class AccessChannel(str, Enum):
    USSD = "USSD"
    APP = "APP"
    AGENT = "AGENT"


class UnifiedTransaction(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=64)
    timestamp: datetime = Field(..., description="Horodatage ; sans fuseau = heure locale RDC")
    user_id: str = Field(..., min_length=1, max_length=64)
    wallet_id: str | None = None
    card_id: str | None = None
    channel: Channel
    operator: str | None = Field(None, description="VODACOM, AIRTEL, ORANGE")
    tx_type: TxType
    access_channel: AccessChannel
    amount: float = Field(..., gt=0)
    currency: str = Field(..., pattern="^(CDF|USD)$")
    amount_usd: float = Field(..., gt=0)
    balance_before_usd: float = Field(..., ge=0)
    # Statut connu seulement pour les transactions rejouées (simulateur) ; en temps réel
    # il est absent : la transaction est scorée AVANT d'être exécutée.
    status: str | None = None
    counterparty_id: str | None = None
    merchant_id: str | None = None
    merchant_category: str | None = None
    merchant_country: str | None = None
    agent_id: str | None = None
    device_id: str = Field(..., min_length=1)
    device_type: str = Field("smartphone", pattern="^(smartphone|feature_phone)$")
    ip_country: str | None = None
    location_province: str

    @field_validator(*ID_FIELDS, mode="before")
    @classmethod
    def _ids_as_text(cls, v):
        return canonical_id(v)

    @field_validator("timestamp")
    @classmethod
    def _to_local_naive(cls, v: datetime) -> datetime:
        """Ramène tout horodatage à l'heure locale RDC, sans fuseau (comme à l'entraînement)."""
        if v.tzinfo is not None:
            v = v.astimezone(KINSHASA_TZ).replace(tzinfo=None)
        return v

    def to_feature_input(self, status: str) -> dict:
        """Dictionnaire attendu par ml.features.BehavioralFeatureExtractor."""
        ts = self.timestamp
        return {
            "user_id": self.user_id,
            "tx_type": self.tx_type.value,
            "channel": self.channel.value,
            "amount_usd": self.amount_usd,
            "balance_before_usd": self.balance_before_usd,
            "status": status,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "access_channel": self.access_channel.value,
            "ip_country": self.ip_country,
            "location_province": self.location_province,
            "counterparty_id": self.counterparty_id,
            "agent_id": self.agent_id,
            "merchant_id": self.merchant_id,
            "merchant_category": self.merchant_category,
            "merchant_country": self.merchant_country,
            # secondes depuis 1970 en heure locale « naïve », comme build_feature_table
            "ts": (ts - datetime(1970, 1, 1)).total_seconds(),
            "hour": ts.hour,
            "weekday": ts.weekday(),
            "day": ts.day,
        }
