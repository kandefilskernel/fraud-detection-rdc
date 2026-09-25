"""Chaque adaptateur doit reproduire exactement la transaction pivot (aller-retour)."""
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "integration-layer")]

from app.adapters.airtel_adapter import AirtelAdapter  # noqa: E402
from app.adapters.base_adapter import AdapterError  # noqa: E402
from app.adapters.orange_adapter import OrangeAdapter  # noqa: E402
from app.adapters.visa_virtual_adapter import VisaVirtualAdapter  # noqa: E402
from app.adapters.vodacom_adapter import VodacomAdapter  # noqa: E402

MM_TX = dict(transaction_id="TX1", timestamp=datetime(2025, 11, 2, 21, 15, 7), user_id="U000781",
             wallet_id="243891234567", channel="MOBILE_MONEY", tx_type="CASH_OUT", access_channel="AGENT",
             amount=285000.0, currency="CDF", amount_usd=100.0, balance_before_usd=120.5, status=None,
             counterparty_id=None, merchant_id=None, merchant_category=None, merchant_country=None,
             agent_id="A000123", device_id="DEVabc", device_type="feature_phone", ip_country=None,
             location_province="Haut-Katanga")
CARD_TX = dict(MM_TX, transaction_id="TX2", channel="VISA_VIRTUAL", operator="AIRTEL", tx_type="CARD_PURCHASE",
               access_channel="APP", amount=49.99, currency="USD", amount_usd=49.99, card_id="CARD9",
               agent_id=None, merchant_id="M77", merchant_category="ELECTRONICS", merchant_country="AE",
               device_type="smartphone", ip_country="AE")


@pytest.mark.parametrize("adapter,operator", [(VodacomAdapter(), "VODACOM"), (AirtelAdapter(), "AIRTEL"),
                                              (OrangeAdapter(), "ORANGE")])
def test_mobile_money_roundtrip(adapter, operator):
    u = adapter.to_unified(adapter.from_unified({**MM_TX, "operator": operator}))
    assert u.operator == operator
    for k in ("transaction_id", "user_id", "tx_type", "amount_usd", "agent_id", "device_id",
              "location_province", "timestamp"):
        got = getattr(u, k)
        assert (got.value if hasattr(got, "value") else got) == MM_TX[k]


def test_visa_roundtrip_and_minor_units():
    a = VisaVirtualAdapter()
    msg = a.from_unified(CARD_TX)
    assert msg["amount_minor"] == 4999 and msg["currency_code"] == "840"
    u = a.to_unified(msg)
    assert u.channel.value == "VISA_VIRTUAL" and u.amount_usd == 49.99 and u.merchant_country == "AE"


def test_cdf_is_converted_to_usd():
    assert VodacomAdapter(usd_cdf_rate=2850).to_usd(285000, "CDF") == 100.0


def test_missing_field_is_rejected():
    msg = VodacomAdapter().from_unified({**MM_TX, "operator": "VODACOM"})
    del msg["DeviceIMEI"]
    with pytest.raises(AdapterError):
        VodacomAdapter().to_unified(msg)


def test_unknown_type_is_rejected():
    msg = OrangeAdapter().from_unified({**MM_TX, "operator": "ORANGE"})
    msg["type_operation"] = "INCONNU"
    with pytest.raises(AdapterError):
        OrangeAdapter().to_unified(msg)
