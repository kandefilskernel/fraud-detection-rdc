"""Orange Money RDC : champs à plat en français, date « JJ/MM/AAAA HH:MM:SS »."""
from shared.schemas.unified_transaction import UnifiedTransaction

from .base_adapter import AdapterError, BaseAdapter

TYPE_MAP = {"TRANSFERT_ENVOI": "P2P_SEND", "TRANSFERT_RECU": "P2P_RECEIVE", "DEPOT": "CASH_IN",
            "RETRAIT": "CASH_OUT", "PAIEMENT_MARCHAND": "MERCHANT_PAYMENT", "CREDIT": "AIRTIME",
            "FACTURE": "BILL_PAYMENT", "RECHARGE_CARTE": "CARD_TOPUP"}
REVERSE = {v: k for k, v in TYPE_MAP.items()}
FMT = "%d/%m/%Y %H:%M:%S"


class OrangeAdapter(BaseAdapter):
    provider = "orange"

    def to_unified(self, p: dict) -> UnifiedTransaction:
        self.require(p, "id_transaction", "date_heure", "numero_client", "id_client", "type_operation",
                     "montant", "devise", "solde_avant_usd", "imei", "canal", "province")
        tx_type = TYPE_MAP.get(p["type_operation"])
        if tx_type is None:
            raise AdapterError(f"type_operation inconnu : {p['type_operation']}")
        return UnifiedTransaction(
            transaction_id=p["id_transaction"], timestamp=self.parse_time(p["date_heure"], FMT),
            user_id=p["id_client"], wallet_id=p["numero_client"],
            channel="MOBILE_MONEY", operator="ORANGE", tx_type=tx_type,
            access_channel=p["canal"], amount=float(p["montant"]), currency=p["devise"],
            amount_usd=self.to_usd(p["montant"], p["devise"]),
            balance_before_usd=float(p["solde_avant_usd"]), status=p.get("statut"),
            counterparty_id=p.get("beneficiaire"), agent_id=p.get("code_agent"),
            merchant_id=p.get("code_marchand"), merchant_category=p.get("categorie_marchand"),
            merchant_country=p.get("pays_marchand"),
            device_id=p["imei"], device_type=p.get("type_terminal", "smartphone"),
            ip_country=p.get("pays_ip"), location_province=p["province"],
            sim_swap_at=self.parse_optional_time(p.get("date_dernier_changement_sim"), FMT),
        )

    def from_unified(self, t: dict) -> dict:
        return {
            "id_transaction": t["transaction_id"], "date_heure": t["timestamp"].strftime(FMT),
            "numero_client": t.get("wallet_id"), "id_client": t["user_id"],
            "type_operation": REVERSE[t["tx_type"]], "montant": t["amount"], "devise": t["currency"],
            "solde_avant_usd": t["balance_before_usd"], "statut": t.get("status"),
            "beneficiaire": t.get("counterparty_id"), "code_agent": t.get("agent_id"),
            "code_marchand": t.get("merchant_id"), "categorie_marchand": t.get("merchant_category"),
            "pays_marchand": t.get("merchant_country"), "imei": t["device_id"],
            "type_terminal": t.get("device_type"), "canal": t["access_channel"],
            "pays_ip": t.get("ip_country"), "province": t["location_province"],
            "date_dernier_changement_sim": self.format_optional_time(t.get("sim_swap_at"), FMT),
        }
