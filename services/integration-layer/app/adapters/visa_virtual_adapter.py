"""Carte Visa virtuelle : message d'autorisation inspiré d'ISO 8583 (STAN, processing code,
montant en unités mineures, devise ISO 4217 numérique : 840 = USD, 976 = CDF)."""
from shared.schemas.unified_transaction import UnifiedTransaction

from .base_adapter import AdapterError, BaseAdapter

PROC_MAP = {"00": "CARD_PURCHASE", "28": "CARD_TOPUP"}   # 00 achat, 28 crédit/rechargement
REVERSE_PROC = {v: k for k, v in PROC_MAP.items()}
CURRENCY = {"840": "USD", "976": "CDF"}
REVERSE_CURRENCY = {v: k for k, v in CURRENCY.items()}


class VisaVirtualAdapter(BaseAdapter):
    provider = "visa"

    def to_unified(self, p: dict) -> UnifiedTransaction:
        self.require(p, "stan", "transmission_datetime", "card_id", "cardholder_id", "processing_code",
                     "amount_minor", "currency_code", "available_balance_usd", "device_id", "province")
        tx_type = PROC_MAP.get(p["processing_code"])
        if tx_type is None:
            raise AdapterError(f"processing_code non géré : {p['processing_code']}")
        currency = CURRENCY.get(str(p["currency_code"]))
        if currency is None:
            raise AdapterError(f"currency_code inconnu : {p['currency_code']}")
        amount = int(p["amount_minor"]) / 100
        merchant = p.get("merchant") or {}
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
        )

    def from_unified(self, t: dict) -> dict:
        return {
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
        }
