"""Intégration des données réelles : import d'un export « opérateur », qualité, pseudonymisation."""
import json

import pandas as pd
import pytest

from ml.onboarding.importer import build_workspace, import_operator, normalize_msisdn, parse_number
from ml.onboarding.mapping import MappingError, load_mapping
from ml.onboarding.quality import assess
from shared.privacy.pseudonymize import Pseudonymizer

MAPPING = """
operator: AIRTEL
format: {type: csv, sep: ";", decimal: ","}
timestamp_format: "%d/%m/%Y %H:%M"
timezone: {source: UTC, target: Africa/Kinshasa}
currency: {default: CDF, usd_cdf_rate: 2800}
columns:
  transaction_id: ID
  timestamp: DATE
  user_id: TEL
  tx_type: TYPE
  amount: MONTANT
  currency: DEV
  counterparty_id: TEL_DEST
  location_province: PROV
values:
  tx_type: {"Envoi": P2P_SEND, "Retrait": CASH_OUT}
labels: {default_report_delay_days: 2}
fraud_reports:
  format: {type: csv, sep: ";"}
  columns: {transaction_id: ID, fraud_type: TYPE_FRAUDE}
"""
EXPORT = """ID;DATE;TEL;NOM;TYPE;MONTANT;DEV;TEL_DEST;PROV
T1;01/06/2025 10:00;+243 81 000 0001;Jean Mbala;Envoi;28 000,00;FC;0810000002;Kasaï-Oriental
T2;01/06/2025 11:30;+243 81 000 0001;Jean Mbala;Retrait;10,50;USD;;kinshasa
T3;01/06/2025 12:00;0810000002;Marie K;Envoi;5,00;USD;+243810000001;
T4;01/06/2025 12:05;0810000002;Marie K;Rechargement inconnu;5,00;USD;;
T5;xx/06/2025;0810000002;Marie K;Envoi;5,00;USD;;
"""
FRAUDS = "ID;TYPE_FRAUDE\nT2;SIM swap\nT9;inconnue\n"
KEY = "cle-de-test-0123456789abcdef"


@pytest.fixture
def export(tmp_path):
    (tmp_path / "m.yaml").write_text(MAPPING, encoding="utf-8")
    (tmp_path / "tx.csv").write_text(EXPORT, encoding="utf-8")
    (tmp_path / "fr.csv").write_text(FRAUDS, encoding="utf-8")
    return tmp_path


def test_msisdn_and_numbers():
    assert normalize_msisdn("+243 81 234 5678") == normalize_msisdn("0812345678") == "243812345678"
    assert normalize_msisdn("00243812345678") == "243812345678"
    assert normalize_msisdn("A544729") == "A544729"                      # code agent : inchangé
    assert parse_number(pd.Series(["1 234,50", "12,00"]), ",").tolist() == [1234.5, 12.0]


def test_import_pseudonymizes_drops_personal_data_and_logs_everything(export):
    ws = export / "ws"
    log = import_operator(load_mapping(export / "m.yaml"), export / "tx.csv", ws, frauds_path=export / "fr.csv",
                          pseudo=Pseudonymizer(KEY))
    assert log["rows_read"] == 5 and log["rows_kept"] == 3
    assert log["dropped"] == {"horodatage illisible": 1,
                              "type d'opération non reconnu (compléter values.tx_type)": 1}
    assert log["unknown_values"]["tx_type"] == {"RECHARGEMENT INCONNU": 1}
    assert log["fraud_reports"] == {"lines": 2, "matched": 1, "unmatched": 1}
    text = (ws / "sources" / "m" / "transactions.csv").read_text(encoding="utf-8")
    for secret in ("Jean", "Marie", "243810000001", "810000002", "NOM"):
        assert secret not in text                                          # aucune donnée personnelle
    t = pd.read_csv(ws / "sources" / "m" / "transactions.csv").set_index("transaction_id")
    p = Pseudonymizer(KEY)
    assert t.loc["T1", "user_id"] == p("243810000001") == t.loc["T3", "counterparty_id"]  # même personne
    assert t.loc["T1", "timestamp"] == "2025-06-01 11:00:00"               # UTC -> heure de Kinshasa
    assert t.loc["T1", "amount_usd"] == 10.0 and t.loc["T1", "location_province"] == "Kasai-Oriental"
    assert t.loc["T2", "location_province"] == "Kinshasa"
    assert t.loc["T2", "is_fraud"] == 1 and t.loc["T2", "fraud_type"] == "SIM_SWAP"
    assert t.loc["T2", "fraud_reported_at"].startswith("2025-06-03")     # délai par défaut : 2 jours
    assert t.loc["T1", "device_id"].startswith("P")                        # appareil fictif, pseudonymisé


def test_build_derives_users_and_quality_blocks_small_data(export):
    ws = export / "ws"
    import_operator(load_mapping(export / "m.yaml"), export / "tx.csv", ws, frauds_path=export / "fr.csv",
                    pseudo=Pseudonymizer(KEY))
    tx, users, logs = build_workspace(ws)
    assert len(users) == 2 and set(users["kyc_level"]) == {1}              # profil prudent par défaut
    assert tx["location_province"].notna().all()                           # province manquante -> domicile
    assert (tx["balance_before_usd"] >= 1e9).all()                         # solde inconnu -> neutre
    q = assess(tx, users, logs)
    assert not q["ready"] and any("minimum 10,000" in b for b in q["blocking"])
    assert any("solde" in w for w in q["warnings"])


def test_clear_ids_are_blocking_and_bad_mapping_is_explained(export, tmp_path):
    ws = export / "ws"
    import_operator(load_mapping(export / "m.yaml"), export / "tx.csv", ws, frauds_path=export / "fr.csv")
    tx, users, logs = build_workspace(ws)
    assert any("NON pseudonymisés" in b for b in assess(tx, users, logs)["blocking"])
    bad = tmp_path / "bad.yaml"
    bad.write_text(MAPPING.replace("P2P_SEND", "ENVOI_ARGENT"), encoding="utf-8")
    with pytest.raises(MappingError, match="codes cibles invalides"):
        load_mapping(bad)
    json.loads((ws / "sources" / "m" / "import_log.json").read_text(encoding="utf-8"))
