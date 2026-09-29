"""Fonctions d'intégration opérateurs côté ML : idempotence, règles (SIM, score réseau carte),
méthode de vérification, pseudonymisation neutre pour les variables."""
import threading

import fakeredis
import pandas as pd
import pytest

from ml.features.feature_engineering import BehavioralFeatureExtractor
from ml.serving.decision import DecisionPolicy, regulatory_rules, verification_method
from ml.serving.idempotency import IdempotencyConflict, IdempotencyInProgress, IdempotencyStore, fingerprint
from ml.tests.test_serving import NET_USERS, network_stream
from shared.privacy.pseudonymize import Pseudonymizer

FEATS = {"is_new_device": 0.0, "is_near_full_drain": 0.0, "amount_to_kyc_limit": 0.1, "is_new_province": 0.0}


# ---------------------------------------------------------------------- idempotence
def test_idempotency_first_then_replay():
    store = IdempotencyStore(fakeredis.FakeRedis())
    key, fp = store.key("MOBILE_MONEY", "VODACOM", "T1"), fingerprint({"montant": 10})
    assert store.begin(key, fp) is None                       # 1re réception : à traiter
    store.complete(key, fp, {"action": "VERIFY"})
    assert store.begin(key, fp) == {"action": "VERIFY"}       # renvoi : décision d'origine


def test_idempotency_conflict_on_changed_content():
    store = IdempotencyStore(fakeredis.FakeRedis())
    key = store.key("MOBILE_MONEY", "VODACOM", "T2")
    store.begin(key, fingerprint({"montant": 10}))
    store.complete(key, fingerprint({"montant": 10}), {"action": "APPROVE"})
    with pytest.raises(IdempotencyConflict):
        store.begin(key, fingerprint({"montant": 9999}))


def test_idempotency_abort_allows_retry_and_concurrent_send_waits():
    r = fakeredis.FakeRedis()
    store = IdempotencyStore(r, wait_s=0.2)
    key, fp = store.key("VISA_VIRTUAL", None, "T3"), fingerprint({"a": 1})
    assert store.begin(key, fp) is None
    with pytest.raises(IdempotencyInProgress):                # envoi simultané, 1er pas fini
        store.begin(key, fp)
    store.abort(key)                                          # échec du scoring : clé libérée
    assert store.begin(key, fp) is None
    # le 2e envoi attend que le 1er termine, puis reçoit sa décision
    t = threading.Timer(0.05, lambda: store.complete(key, fp, {"action": "BLOCK"}))
    t.start()
    assert IdempotencyStore(r, wait_s=1.0).begin(key, fp) == {"action": "BLOCK"}


def test_fingerprint_ignores_field_order():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


# ---------------------------------------------------------------------- signal télécom (SIM)
def test_recent_sim_swap_rules():
    tx = {"tx_type": "CASH_OUT", "hours_since_sim_swap": 3.0}
    ids = [r["id"] for r in regulatory_rules(FEATS, tx)]
    assert ids == ["SIM_RECENTE_OPERATION_SORTANTE"]
    drain = [r["id"] for r in regulatory_rules({**FEATS, "is_near_full_drain": 1.0}, tx)]
    assert "SIM_RECENTE_VIDAGE_OU_MONTANT_ELEVE" in drain
    assert regulatory_rules(FEATS, {"tx_type": "CASH_OUT", "hours_since_sim_swap": 72.0}) == []
    assert regulatory_rules(FEATS, {"tx_type": "P2P_RECEIVE", "hours_since_sim_swap": 1.0}) == []  # entrant


def test_sim_swap_drain_is_blocked_even_at_low_probability():
    d = DecisionPolicy(threshold=0.5).decide(0.002, 140, "MOBILE_MONEY", {**FEATS, "is_near_full_drain": 1.0},
                                             {"tx_type": "CASH_OUT", "hours_since_sim_swap": 2.0})
    assert d["action"] == "BLOCK"


def test_verification_method():
    assert verification_method("VERIFY", "MOBILE_MONEY", {"hours_since_sim_swap": 5.0}) == "HORS_SIM"
    assert verification_method("VERIFY", "VISA_VIRTUAL", {}) == "3DS"
    assert verification_method("VERIFY", "MOBILE_MONEY", {}) == "PIN_USSD"
    assert verification_method("APPROVE", "MOBILE_MONEY", {}) is None


# ---------------------------------------------------------------------- friction ciblée (arnaque)
def test_sending_back_much_more_than_received_asks_confirmation():
    import math
    policy = DecisionPolicy(threshold=0.5)
    scam = policy.decide(0.004, 50, "MOBILE_MONEY", {**FEATS, "refund_ratio_to_cp": math.log1p(10)},
                         {"tx_type": "P2P_SEND"})
    assert scam["action"] == "VERIFY" and "RENVOI_TRES_SUPERIEUR_AU_RECU" in scam["rules_triggered"]
    assert scam["verification_method"] == "PIN_USSD"
    normal = policy.decide(0.004, 50, "MOBILE_MONEY", {**FEATS, "refund_ratio_to_cp": math.log1p(1)},
                           {"tx_type": "P2P_SEND"})
    assert normal["action"] == "APPROVE"          # rembourser le même montant : aucune friction


# ---------------------------------------------------------------------- score réseau (Visa)
def test_card_network_score_complements_our_model():
    policy = DecisionPolicy(threshold=0.5)
    card = {"tx_type": "CARD_PURCHASE"}
    assert policy.decide(0.002, 30, "VISA_VIRTUAL", FEATS, {**card, "network_risk_score": 40})["action"] == "APPROVE"
    high = policy.decide(0.002, 30, "VISA_VIRTUAL", FEATS, {**card, "network_risk_score": 93})
    assert high["action"] == "VERIFY" and high["verification_method"] == "3DS"
    assert policy.decide(0.002, 30, "VISA_VIRTUAL", FEATS, {**card, "network_risk_score": 99})["action"] == "BLOCK"


# ---------------------------------------------------------------------- pseudonymisation
def test_pseudonymizer_is_deterministic_keyed_and_idempotent():
    a, b = Pseudonymizer("cle-numero-un-2026-xx"), Pseudonymizer("cle-numero-deux-2026-x")
    assert a("243891234567") == a("243891234567") != b("243891234567")
    assert a(a("243891234567")) == a("243891234567")
    assert a(None) is None
    with pytest.raises(ValueError):
        Pseudonymizer("courte")


def test_pseudonymization_does_not_change_any_feature():
    """Le modèle ne lit jamais la valeur d'un identifiant, seulement « est-ce le même ? » :
    les variables (profils, réseau C1) sont IDENTIQUES avec des identifiants pseudonymisés."""
    pseudo = Pseudonymizer("cle-de-test-tres-longue-2026")
    start = pd.Timestamp("2025-06-01")
    raw_ext = BehavioralFeatureExtractor(NET_USERS, start)
    users_p = NET_USERS.assign(user_id=NET_USERS.user_id.map(pseudo), wallet_id=NET_USERS.wallet_id.map(pseudo))
    ps_ext = BehavioralFeatureExtractor(users_p, start)
    stream = network_stream()
    assert any(t["counterparty_id"] for t in stream)
    for tx in stream:
        assert ps_ext.process(pseudo.transaction(tx)) == pytest.approx(raw_ext.process(tx))
