"""Airtel Money : JSON imbriqué (transaction / subscriber / device / counterparty), ISO 8601."""
from shared.schemas.unified_transaction import UnifiedTransaction

from .base_adapter import AdapterError, BaseAdapter

TYPE_MAP = {"P2P_OUT": "P2P_SEND", "P2P_IN": "P2P_RECEIVE", "CASHIN": "CASH_IN",
            "CASHOUT": "CASH_OUT", "MERCHPAY": "MERCHANT_PAYMENT", "AIRTIME": "AIRTIME",
            "BILLPAY": "BILL_PAYMENT", "CARD_LOAD": "CARD_TOPUP"}
REVERSE = {v: k for k, v in TYPE_MAP.items()}


class AirtelAdapter(BaseAdapter):
    provider = "airtel"

    def to_unified(self, p: dict) -> UnifiedTransaction:
        try:
            t, s, d = p["transaction"], p["subscriber"], p["device"]
        except KeyError as e:
            raise AdapterError(f"bloc manquant : {e}") from e
        self.require(t, "id", "type", "amount", "currency", "timestamp")
        self.require(s, "msisdn", "customer_id", "balance_usd", "province")
        self.require(d, "imei", "channel")
        tx_type = TYPE_MAP.get(t["type"])
        if tx_type is None:
            raise AdapterError(f"type inconnu : {t['type']}")
        cp = p.get("counterparty") or {}
        return UnifiedTransaction(
            transaction_id=t["id"], timestamp=self.parse_time(t["timestamp"]),
            user_id=s["customer_id"], wallet_id=s["msisdn"],
            channel="MOBILE_MONEY", operator="AIRTEL", tx_type=tx_type,
            access_channel=d["channel"], amount=float(t["amount"]), currency=t["currency"],
            amount_usd=self.to_usd(t["amount"], t["currency"]),
            balance_before_usd=float(s["balance_usd"]), status=t.get("status"),
            counterparty_id=cp.get("msisdn"), agent_id=cp.get("agent_code"),
            merchant_id=cp.get("merchant_code"), merchant_category=cp.get("merchant_category"),
            merchant_country=cp.get("merchant_country"),
            device_id=d["imei"], device_type=d.get("type", "smartphone"),
            ip_country=d.get("ip_country"), location_province=s["province"],
            sim_swap_at=self.parse_optional_time(s.get("last_sim_swap")),
        )

    def from_unified(self, t: dict) -> dict:
        return {
            "transaction": {"id": t["transaction_id"], "type": REVERSE[t["tx_type"]],
                            "amount": t["amount"], "currency": t["currency"],
                            "timestamp": t["timestamp"].isoformat(), "status": t.get("status")},
            "subscriber": {"msisdn": t.get("wallet_id"), "customer_id": t["user_id"],
                           "balance_usd": t["balance_before_usd"], "province": t["location_province"],
                           "last_sim_swap": self.format_optional_time(t.get("sim_swap_at"))},
            "device": {"imei": t["device_id"], "type": t.get("device_type"),
                       "channel": t["access_channel"], "ip_country": t.get("ip_country")},
            "counterparty": {"msisdn": t.get("counterparty_id"), "agent_code": t.get("agent_id"),
                             "merchant_code": t.get("merchant_id"),
                             "merchant_category": t.get("merchant_category"),
                             "merchant_country": t.get("merchant_country")},
        }
