"""Fichier de correspondance d'un opérateur (YAML) : ses noms de colonnes et ses codes
vers le format pivot du système. Un modèle commenté : mappings/modele_operateur.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

OPERATORS = ("VODACOM", "AIRTEL", "ORANGE", "VISA")
CHANNELS = ("MOBILE_MONEY", "VISA_VIRTUAL")
TX_TYPES = ("P2P_SEND", "P2P_RECEIVE", "CASH_IN", "CASH_OUT", "MERCHANT_PAYMENT", "AIRTIME",
            "BILL_PAYMENT", "CARD_TOPUP", "CARD_PURCHASE")
STATUSES = ("SUCCESS", "FAILED")
ACCESS = ("APP", "USSD", "AGENT", "WEB")

# Champs pivot d'une transaction. required = l'import échoue s'ils manquent.
TX_FIELDS = {
    "transaction_id": True, "timestamp": True, "user_id": True, "tx_type": True, "amount": True,
    "wallet_id": False, "card_id": False, "channel": False, "currency": False, "status": False,
    "access_channel": False, "balance_before": False, "balance_after": False,
    "counterparty_id": False, "merchant_id": False, "merchant_category": False, "merchant_country": False,
    "agent_id": False, "device_id": False, "device_type": False, "ip_country": False,
    "location_province": False,
    # étiquettes (facultatives ici : elles peuvent venir d'un fichier de signalements séparé)
    "is_fraud": False, "fraud_type": False, "fraud_reported_at": False,
}
FRAUD_FIELDS = {"transaction_id": True, "reported_at": False, "fraud_type": False}
KYC_FIELDS = {"user_id": True, "wallet_id": False, "province": False, "kyc_level": False,
              "kyc_tx_limit": False, "account_opened_at": False, "monthly_income": False,
              "has_visa_virtual": False}


class MappingError(ValueError):
    pass


@dataclass
class FileFormat:
    type: str = "csv"            # csv | excel | parquet
    sep: str = ","
    encoding: str = "utf-8"
    decimal: str = "."
    sheet: str | int = 0

    @classmethod
    def parse(cls, d: dict | None) -> "FileFormat":
        f = cls(**(d or {}))
        if f.type not in ("csv", "excel", "parquet"):
            raise MappingError(f"format.type inconnu : {f.type} (csv, excel ou parquet)")
        return f


@dataclass
class SubMapping:
    """Fichier annexe (signalements de fraude, référentiel KYC)."""
    columns: dict
    format: FileFormat = field(default_factory=FileFormat)
    timestamp_format: str | None = None
    values: dict = field(default_factory=dict)


@dataclass
class OperatorMapping:
    operator: str
    channel: str
    columns: dict
    values: dict
    format: FileFormat
    timestamp_format: str | None
    source_timezone: str
    target_timezone: str
    default_currency: str
    usd_cdf_rate: float
    amount_scale: float
    fraud_positive_values: list
    default_report_delay_days: float
    fraud_reports: SubMapping | None
    kyc: SubMapping | None
    path: Path

    @property
    def name(self) -> str:
        return self.path.stem


def _check_columns(cols: dict, allowed: dict, where: str) -> None:
    if not isinstance(cols, dict) or not cols:
        raise MappingError(f"{where}.columns : dictionnaire « champ du système : colonne de l'export » attendu")
    unknown = set(cols) - set(allowed)
    if unknown:
        raise MappingError(f"{where}.columns : champs inconnus {sorted(unknown)} (autorisés : {sorted(allowed)})")


def _sub(d: dict | None, allowed: dict, where: str) -> SubMapping | None:
    if not d:
        return None
    _check_columns(d.get("columns"), allowed, where)
    missing = [k for k, req in allowed.items() if req and k not in d["columns"]]
    if missing:
        raise MappingError(f"{where}.columns : champs obligatoires absents {missing}")
    return SubMapping(columns=d["columns"], format=FileFormat.parse(d.get("format")),
                      timestamp_format=d.get("timestamp_format"), values=d.get("values") or {})


def load_mapping(path: str | Path) -> OperatorMapping:
    path = Path(path)
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    op = str(d.get("operator", "")).upper()
    if op not in OPERATORS:
        raise MappingError(f"operator : {op!r} inconnu (valeurs : {OPERATORS})")
    channel = str(d.get("channel", "VISA_VIRTUAL" if op == "VISA" else "MOBILE_MONEY")).upper()
    if channel not in CHANNELS:
        raise MappingError(f"channel : {channel!r} inconnu (valeurs : {CHANNELS})")
    cols = d.get("columns")
    _check_columns(cols, TX_FIELDS, "transactions")
    missing = [k for k, req in TX_FIELDS.items() if req and k not in cols]
    if missing:
        raise MappingError(f"columns : champs obligatoires absents {missing}")
    values = d.get("values") or {}
    for key, allowed in (("tx_type", TX_TYPES), ("status", STATUSES), ("channel", CHANNELS),
                         ("access_channel", ACCESS)):
        bad = {k: v for k, v in (values.get(key) or {}).items() if v not in allowed}
        if bad:
            raise MappingError(f"values.{key} : codes cibles invalides {bad} (autorisés : {allowed})")
    tz = d.get("timezone") or {}
    cur = d.get("currency") or {}
    labels = d.get("labels") or {}
    rate = float(cur.get("usd_cdf_rate", 0) or 0)
    if rate <= 0:
        raise MappingError("currency.usd_cdf_rate : taux CDF pour 1 USD obligatoire (ex. 2850)")
    return OperatorMapping(
        operator=op, channel=channel, columns=cols, values=values,
        format=FileFormat.parse(d.get("format")), timestamp_format=d.get("timestamp_format"),
        source_timezone=tz.get("source", "Africa/Kinshasa"), target_timezone=tz.get("target", "Africa/Kinshasa"),
        default_currency=str(cur.get("default", "CDF")).upper(), usd_cdf_rate=rate,
        amount_scale=float(d.get("amount_scale", 1.0)),
        fraud_positive_values=[str(v).strip().upper() for v in labels.get("positive_values", ["1", "TRUE", "OUI", "YES"])],
        default_report_delay_days=float(labels.get("default_report_delay_days", 3)),
        fraud_reports=_sub(d.get("fraud_reports"), FRAUD_FIELDS, "fraud_reports"),
        kyc=_sub(d.get("kyc"), KYC_FIELDS, "kyc"),
        path=path,
    )
