"""Vodacom M-Pesa RDC : format « à la Daraja » (champs PascalCase, horodatage AAAAMMJJhhmmss)."""
from shared.schemas.unified_transaction import UnifiedTransaction

from .base_adapter import AdapterError, BaseAdapter

TYPE_MAP = {"SendMoney": "P2P_SEND", "ReceiveMoney": "P2P_RECEIVE", "Deposit": "CASH_IN",
            "Withdraw": "CASH_OUT", "BuyGoods": "MERCHANT_PAYMENT", "Airtime": "AIRTIME",
            "PayBill": "BILL_PAYMENT", "CardTopUp": "CARD_TOPUP"}
REVERSE = {v: k for k, v in TYPE_MAP.items()}


class VodacomAdapter(BaseAdapter):
    provider = "vodacom"

    def to_unified(self, p: dict) -> UnifiedTransaction:
        self.require(p, "TransID", "TransTime", "MSISDN", "CustomerID", "TransactionType",
                     "TransAmount", "Currency", "AccountBalanceUSD", "DeviceIMEI", "Channel", "Location")
        tx_type = TYPE_MAP.get(p["TransactionType"])
        if tx_type is None:
            raise AdapterError(f"TransactionType inconnu : {p['TransactionType']}")
        return UnifiedTransaction(
            transaction_id=p["TransID"],
            timestamp=self.parse_time(p["TransTime"], "%Y%m%d%H%M%S"),
            user_id=p["CustomerID"], wallet_id=p["MSISDN"],
            channel="MOBILE_MONEY", operator="VODACOM", tx_type=tx_type,
            access_channel=p["Channel"], amount=float(p["TransAmount"]), currency=p["Currency"],
            amount_usd=self.to_usd(p["TransAmount"], p["Currency"]),
            balance_before_usd=float(p["AccountBalanceUSD"]),
            status=p.get("ResultStatus"),
            counterparty_id=p.get("CounterpartyMSISDN"), agent_id=p.get("AgentTill"),
            merchant_id=p.get("ShortCode"), merchant_category=p.get("MerchantCategory"),
            merchant_country=p.get("MerchantCountry"),
            device_id=p["DeviceIMEI"], device_type=p.get("DeviceType", "smartphone"),
            ip_country=p.get("IPCountry"), location_province=p["Location"],
            sim_swap_at=self.parse_optional_time(p.get("LastSimSwapTime"), "%Y%m%d%H%M%S"),
        )

    def from_unified(self, t: dict) -> dict:
        return {
            "TransID": t["transaction_id"], "TransTime": t["timestamp"].strftime("%Y%m%d%H%M%S"),
            "MSISDN": t.get("wallet_id"), "CustomerID": t["user_id"],
            "TransactionType": REVERSE[t["tx_type"]], "TransAmount": t["amount"],
            "Currency": t["currency"], "AccountBalanceUSD": t["balance_before_usd"],
            "ResultStatus": t.get("status"), "CounterpartyMSISDN": t.get("counterparty_id"),
            "AgentTill": t.get("agent_id"), "ShortCode": t.get("merchant_id"),
            "MerchantCategory": t.get("merchant_category"), "MerchantCountry": t.get("merchant_country"),
            "DeviceIMEI": t["device_id"], "DeviceType": t.get("device_type"),
            "Channel": t["access_channel"], "IPCountry": t.get("ip_country"),
            "Location": t["location_province"],
            "LastSimSwapTime": self.format_optional_time(t.get("sim_swap_at"), "%Y%m%d%H%M%S"),
        }
