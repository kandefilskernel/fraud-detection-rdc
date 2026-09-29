"""Carte Visa virtuelle, côté ÉMETTEUR (banque ou fintech qui émet la carte).

Pendant l'autorisation, le réseau Visa transmet la demande à l'émetteur (message ISO 8583,
MTI 0100) ; le processeur de l'émetteur interroge alors cette plateforme avant de répondre.
Le message reprend les champs utiles d'ISO 8583 (STAN, processing code, montant en unités
mineures, devise ISO 4217 numérique : 840 = USD, 976 = CDF) et deux informations fournies par
le réseau : son propre score de risque (ex. Visa Advanced Authorization, 0-99) et le résultat
d'une éventuelle authentification 3-D Secure. Notre score COMPLÈTE celui du réseau.

La décision est traduite en code réponse ISO 8583 (champ 39) pour l'émetteur :
    APPROVE -> 00  approuvée
    VERIFY  -> 1A  authentification supplémentaire requise (déclenche 3-D Secure)
    BLOCK   -> 59  fraude suspectée
"""
from shared.schemas.unified_transaction import UnifiedTransaction

from .base_adapter import AdapterError, BaseAdapter

PROC_MAP = {"00": "CARD_PURCHASE", "28": "CARD_TOPUP"}   # 00 achat, 28 crédit/rechargement
REVERSE_PROC = {v: k for k, v in PROC_MAP.items()}
CURRENCY = {"840": "USD", "976": "CDF"}
REVERSE_CURRENCY = {v: k for k, v in CURRENCY.items()}
ISO8583_RESPONSE = {"APPROVE": "00", "VERIFY": "1A", "BLOCK": "59"}
AUTHORIZATION_REQUEST_MTI = "0100"


class VisaVirtualAdapter(BaseAdapter):
    provider = "visa"

    def to_unified(self, p: dict) -> UnifiedTransaction:
        self.require(p, "stan", "transmission_datetime", "card_id", "cardholder_id", "processing_code",
                     "amount_minor", "currency_code", "available_balance_usd", "device_id", "province")
        mti = p.get("mti", AUTHORIZATION_REQUEST_MTI)
        if mti != AUTHORIZATION_REQUEST_MTI:
            raise AdapterError(f"MTI {mti} non géré : seule la demande d'autorisation (0100) est scorée")
        tx_type = PROC_MAP.get(p["processing_code"])
        if tx_type is None:
            raise AdapterError(f"processing_code non géré : {p['processing_code']}")
        currency = CURRENCY.get(str(p["currency_code"]))
        if currency is None:
            raise AdapterError(f"currency_code inconnu : {p['currency_code']}")
        amount = int(p["amount_minor"]) / 100
        merchant = p.get("merchant") or {}
        network = p.get("network") or {}
        return UnifiedTransaction(
            transaction_id=p["stan"], timestamp=self.parse_time(p["transmission_datetime"]),
            user_id=p["cardholder_id"], card_id=p["card_id"], wallet_id=p.get("funding_wallet"),
            channel="VISA_VIRTUAL", operator=p.get("funding_operator"), tx_type=tx_type,
            access_channel=p.get("access_channel", "APP"), amount=amount, currency=currency,
            amount_usd=self.to_usd(amount, currency),
            balance_before_usd=float(p["available_balance_usd"]), status=p.get("response_status"),
            merchant_id=merchant.get("mid"), merchant_category=merchant.get("category"),
            merchant_country=merchant.get("country"),
            device_id=p["device_id"], device_type=p.get("device_type", "smartphone"),
            ip_country=p.get("ip_country"), location_province=p["province"],
            network_risk_score=network.get("risk_score"),
            three_ds_authenticated=network.get("three_ds_authenticated"),
        )

    def from_unified(self, t: dict) -> dict:
        return {
            "mti": AUTHORIZATION_REQUEST_MTI,
            "stan": t["transaction_id"], "transmission_datetime": t["timestamp"].isoformat(),
            "card_id": t.get("card_id"), "cardholder_id": t["user_id"],
            "funding_wallet": t.get("wallet_id"), "funding_operator": t.get("operator"),
            "processing_code": REVERSE_PROC[t["tx_type"]],
            "amount_minor": int(round(t["amount"] * 100)), "currency_code": REVERSE_CURRENCY[t["currency"]],
            "available_balance_usd": t["balance_before_usd"], "response_status": t.get("status"),
            "merchant": {"mid": t.get("merchant_id"), "category": t.get("merchant_category"),
                         "country": t.get("merchant_country")},
            "device_id": t["device_id"], "device_type": t.get("device_type"),
            "access_channel": t["access_channel"], "ip_country": t.get("ip_country"),
            "province": t["location_province"],
            "network": {"risk_score": t.get("network_risk_score"),
                        "three_ds_authenticated": t.get("three_ds_authenticated")},
        }

    @staticmethod
    def authorization_response(action: str) -> str:
        """Code réponse ISO 8583 (champ 39) renvoyé au processeur de l'émetteur."""
        return ISO8583_RESPONSE.get(action, "05")   # 05 « ne pas honorer » par défaut
