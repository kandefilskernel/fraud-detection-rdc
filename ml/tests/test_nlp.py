"""NLP des SMS signalés : extraction et masquage des numéros, classifieur, règle de décision."""
import math

import pytest

from ml.serving.decision import DecisionPolicy, regulatory_rules
from shared.nlp.phone_numbers import extract_numbers, is_phone_number, mask_numbers, normalize

FEATS = {"is_near_full_drain": 0.0, "amount_to_kyc_limit": 0.1, "is_new_device": 0.0, "is_new_province": 0.0}


# ---------------------------------------------------------------------- numéros
@pytest.mark.parametrize("text, expected", [
    ("Svp renvoyez au +243 81 234 5678 merci", ["243812345678"]),
    ("appelez le 081-234-56-78 ou (+243) 991234567", ["243812345678", "243991234567"]),
    ("numero zéro huit un deux trois quatre cinq six sept huit", ["243812345678"]),
    ("0 8 1 2 3 4 5 6 7 8", ["243812345678"]),
    ("00243 46 052 8183", ["243460528183"]),
    ("composez *144# et entrez votre PIN", []),                 # code court
    ("montant 1 500 000 FC le 12/11/2025", []),                  # montant et date
])
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == expected


def test_mask_numbers_removes_every_number():
    masked = mask_numbers("Reçu 20 000 FC de 0812345678, rappelez le +243 99 123 4567")
    assert "<NUMERO>" in masked and "0812345678" not in masked and "99 123" not in masked
    assert "20 000 FC" in masked                                 # le montant reste


def test_sender_normalization():
    assert normalize("+243 81 234 5678") == "243812345678"
    assert normalize("VODACOM") is None and not is_phone_number("M-PESA")


# ---------------------------------------------------------------------- classifieur
@pytest.fixture(scope="module")
def clf():
    from ml.nlp.scam_sms import ARTIFACT_DIR, ScamSmsClassifier
    if not (ARTIFACT_DIR / "scam_sms.joblib").exists():
        pytest.skip("classifieur non entraîné (python -m ml.nlp.scam_sms)")
    return ScamSmsClassifier.load()


def test_classifier_flags_wrong_transfer_scam(clf):
    text = mask_numbers("Bonjour maman, je vous ai envoyé 50 000 FC par erreur, c'était pour l'hôpital. "
                        "Svp renvoyez au 0812345678, que Dieu vous bénisse")
    res = clf.classify(text, sender_is_number=True)
    assert res["is_scam"] and res["category"] == "ENVOI_ERREUR"


def test_classifier_keeps_operator_confirmation_legitimate(clf):
    text = mask_numbers("Vous avez reçu 20 000 FC de 0812345678 (Kabeya). Nouveau solde : 35 500 FC. Réf : C12345678.")
    res = clf.classify(text, sender_is_number=False)
    assert not res["is_scam"] and res["category"] == "LEGITIME"


# ---------------------------------------------------------------------- règle
def test_sms_report_rule_targets_recent_wallets_only():
    recent = {**FEATS, "cp_log_age_days": math.log1p(3)}
    old = {**FEATS, "cp_log_age_days": math.log1p(400)}
    tx = {"tx_type": "P2P_SEND", "cp_scam_reports_30d": 1}
    assert "BENEFICIAIRE_SIGNALE_PAR_SMS" in [r["id"] for r in regulatory_rules(recent, tx)]
    # une relation établie n'est jamais mise en cause par un signalement (même malveillant)
    assert "BENEFICIAIRE_SIGNALE_PAR_SMS" not in [r["id"] for r in regulatory_rules(old, tx)]
    assert regulatory_rules(recent, {"tx_type": "P2P_SEND", "cp_scam_reports_30d": 0}) == []
    assert regulatory_rules(recent, {"tx_type": "P2P_RECEIVE", "cp_scam_reports_30d": 3}) == []


def test_sms_report_never_blocks_on_its_own():
    d = DecisionPolicy(threshold=0.25).decide(0.003, 40, "MOBILE_MONEY", {**FEATS, "cp_log_age_days": 0.0},
                                              {"tx_type": "P2P_SEND", "cp_scam_reports_30d": 5})
    assert d["action"] == "VERIFY" and d["verification_method"] == "PIN_USSD"
