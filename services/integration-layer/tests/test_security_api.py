"""Sécurité et API de l'integration-layer : signature HMAC, anti-rejeu, IP, mTLS,
pseudonymisation, modes de déploiement, codes ISO 8583 (Visa) et retours des opérateurs.
Le scoring et le back-office sont simulés (httpx.MockTransport)."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "integration-layer")]

from fastapi.testclient import TestClient  # noqa: E402

import app.main as m  # noqa: E402
from app.adapters.visa_virtual_adapter import VisaVirtualAdapter  # noqa: E402
from app.adapters.vodacom_adapter import VodacomAdapter  # noqa: E402
from app.security import OperatorSecurity, SecurityRejection, parse_allowlist, parse_networks  # noqa: E402
from shared.security.request_signing import SignatureError, sign, signed_headers, verify  # noqa: E402

SECRET = {"vodacom": "dev-vodacom-hmac", "airtel": "dev-airtel-hmac", "visa": "dev-visa-hmac"}
KEYS = {"vodacom": "dev-vodacom-key", "airtel": "dev-airtel-key", "visa": "dev-visa-key"}
MM_TX = dict(transaction_id="TXS1", timestamp=datetime(2025, 11, 2, 21, 15, 7), user_id="U000781",
             wallet_id="243891234567", channel="MOBILE_MONEY", operator="VODACOM", tx_type="P2P_SEND",
             access_channel="APP", amount=50.0, currency="USD", amount_usd=50.0, balance_before_usd=120.5,
             counterparty_id="243899999999", agent_id=None, merchant_id=None, merchant_category=None,
             merchant_country=None, device_id="IMEI-356938035643809", device_type="smartphone",
             ip_country="CD", location_province="Kinshasa", status=None,
             sim_swap_at=datetime(2025, 11, 2, 19, 0, 0))
CARD_TX = dict(MM_TX, transaction_id="TXS2", channel="VISA_VIRTUAL", operator="AIRTEL", tx_type="CARD_PURCHASE",
               amount=49.99, amount_usd=49.99, card_id="CARD9", counterparty_id=None, merchant_id="M77",
               merchant_category="ELECTRONICS", merchant_country="AE", sim_swap_at=None,
               network_risk_score=93, three_ds_authenticated=False)


def score_response(tx: dict, action: str = "APPROVE", **kw) -> dict:
    return {"transaction_id": tx["transaction_id"], "fraud_probability": 0.12, "action": action,
            "risk_level": "MOYEN", "reason": "test", **kw}


class Backends:
    """Scoring et back-office simulés ; mémorise les requêtes reçues."""
    def __init__(self):
        self.scoring_calls, self.backoffice_calls, self.scam_calls = [], [], []
        self.scoring = lambda tx: httpx.Response(200, json=score_response(tx))
        self.backoffice = lambda body: httpx.Response(200, json={"label": 1, "changed": True})

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if request.url.path == "/v1/score":
            self.scoring_calls.append(body)
            return self.scoring(body)
        if request.url.path == "/v1/scam-reports":
            self.scam_calls.append(body)
            return httpx.Response(200, json={"report_id": body["report_id"], "category": "ENVOI_ERREUR",
                                             "is_scam": True, "flagged_numbers": body["numbers"]})
        if request.url.path == "/internal/operator-feedback":
            self.backoffice_calls.append((dict(request.headers), body))
            return self.backoffice(body)
        return httpx.Response(200, json={"status": "ok"})


@pytest.fixture
def api():
    """Client de test ; reconfigure l'application avec des réglages propres à chaque test."""
    backends = Backends()

    def make(**overrides):
        m.RT = m.configure(m.Settings(**overrides))
        client = TestClient(m.app)
        client.__enter__()
        transport = httpx.MockTransport(backends.handler)
        m.state["http"] = httpx.AsyncClient(base_url="http://scoring", transport=transport)
        m.state["backoffice"] = httpx.AsyncClient(base_url="http://backoffice", transport=transport)
        made.append(client)
        return client, backends

    made = []
    yield make
    for c in made:
        c.__exit__(None, None, None)
    m.RT = m.configure(m.Settings())


def post_signed(client, provider, payload: dict, path=None, secret=None, key=None, now=None, body=None):
    path = path or f"/v1/transactions/{provider}"
    raw = body if body is not None else json.dumps(payload, default=str).encode()
    signed = json.dumps(payload, default=str).encode()
    headers = {"X-API-Key": key or KEYS[provider], "Content-Type": "application/json",
               **signed_headers(secret or SECRET[provider], "POST", path, signed, now=now)}
    return client.post(path, content=raw, headers=headers)


# ---------------------------------------------------------------------- signature (unitaire)
def test_signature_roundtrip_and_tampering():
    body = b'{"montant": 100}'
    h = signed_headers("s3cret", "POST", "/v1/transactions/vodacom", body, now=1_000_000)
    verify("s3cret", h["X-Timestamp"], h["X-Signature"], "POST", "/v1/transactions/vodacom", body, now=1_000_010)
    with pytest.raises(SignatureError, match="invalide"):
        verify("s3cret", h["X-Timestamp"], h["X-Signature"], "POST", "/v1/transactions/vodacom",
               b'{"montant": 900}', now=1_000_010)
    with pytest.raises(SignatureError, match="fenêtre"):
        verify("s3cret", h["X-Timestamp"], h["X-Signature"], "POST", "/v1/transactions/vodacom", body,
               now=1_000_000 + 3600)
    with pytest.raises(SignatureError, match="invalide"):   # signature valable sur un autre chemin
        verify("s3cret", h["X-Timestamp"], h["X-Signature"], "POST", "/v1/feedback/vodacom", body, now=1_000_010)
    assert sign("a", "1", "POST", "/p", b"x") != sign("b", "1", "POST", "/p", b"x")


# ---------------------------------------------------------------------- API transactions
def test_unsigned_request_is_rejected(api):
    client, _ = api()
    msg = VodacomAdapter().from_unified(MM_TX)
    r = client.post("/v1/transactions/vodacom", json=msg, headers={"X-API-Key": KEYS["vodacom"]})
    assert r.status_code == 401 and "X-Timestamp" in r.json()["detail"]


def test_signed_request_is_scored_and_sim_swap_forwarded(api):
    client, be = api()
    r = post_signed(client, "vodacom", VodacomAdapter().from_unified(MM_TX))
    assert r.status_code == 200, r.text
    assert r.json()["action"] == "APPROVE" and r.json()["provider"] == "vodacom"
    sent = be.scoring_calls[0]
    assert sent["user_id"] == "U000781"                       # mode démo : identifiants en clair
    assert sent["sim_swap_at"].startswith("2025-11-02T19:00")  # signal télécom transmis au scoring


def test_tampered_body_expired_timestamp_and_wrong_key_are_rejected(api):
    client, be = api()
    msg = VodacomAdapter().from_unified(MM_TX)
    tampered = json.dumps({**msg, "TransAmount": 5000}).encode()
    assert post_signed(client, "vodacom", msg, body=tampered).status_code == 401
    assert post_signed(client, "vodacom", msg, now=time.time() - 3600).status_code == 401
    assert post_signed(client, "vodacom", msg, key=KEYS["airtel"]).status_code == 401
    assert be.scoring_calls == []                              # rien n'a atteint le scoring


def test_visa_issuer_gets_iso8583_response_code(api):
    client, be = api()
    be.scoring = lambda tx: httpx.Response(200, json=score_response(tx, "VERIFY"))
    r = post_signed(client, "visa", VisaVirtualAdapter().from_unified(CARD_TX))
    assert r.json()["iso8583_response_code"] == "1A"          # authentification 3-D Secure demandée
    assert be.scoring_calls[0]["network_risk_score"] == 93    # score du réseau transmis


def test_idempotency_conflict_from_scoring_is_passed_through(api):
    client, be = api()
    be.scoring = lambda tx: httpx.Response(409, json={"detail": "identifiant déjà utilisé"})
    assert post_signed(client, "vodacom", VodacomAdapter().from_unified(MM_TX)).status_code == 409


def test_scoring_down_falls_back_to_verify(api):
    client, be = api()
    be.scoring = lambda tx: httpx.Response(503, json={"detail": "indisponible"})
    r = post_signed(client, "visa", VisaVirtualAdapter().from_unified(CARD_TX))
    assert r.json()["action"] == "VERIFY" and r.json()["iso8583_response_code"] == "1A"


def test_pseudonymization_hides_personal_ids_from_scoring(api):
    client, be = api(PSEUDONYMIZE_IDS=True, PSEUDONYMIZATION_KEY="cle-de-test-tres-longue-2026")
    post_signed(client, "vodacom", VodacomAdapter().from_unified({**MM_TX, "agent_id": "A000123"}))
    sent = be.scoring_calls[0]
    for field in ("user_id", "wallet_id", "device_id", "counterparty_id"):
        assert sent[field].startswith("P") and sent[field] != MM_TX[field]
    assert sent["agent_id"] == "A000123"                       # entreprise : reste en clair
    post_signed(client, "vodacom", VodacomAdapter().from_unified({**MM_TX, "transaction_id": "TXS9"}))
    assert be.scoring_calls[1]["user_id"] == sent["user_id"]   # déterministe : même client, même pseudonyme


def test_deployment_modes(api):
    client, _ = api(DEPLOYMENT_MODE="operator_instance", OPERATOR_ID="vodacom")
    assert post_signed(client, "airtel", {"transaction": {}}).status_code == 404
    with pytest.raises(RuntimeError, match="PSEUDONYMIZE_IDS"):
        m.configure(m.Settings(DEPLOYMENT_MODE="shared_platform"))
    with pytest.raises(RuntimeError, match="REQUIRE_SIGNATURE"):
        m.configure(m.Settings(DEPLOYMENT_MODE="shared_platform", PSEUDONYMIZE_IDS=True,
                               PSEUDONYMIZATION_KEY="cle-de-test-tres-longue-2026", REQUIRE_SIGNATURE=False))


# ---------------------------------------------------------------------- API retours opérateurs
def test_operator_feedback_is_signed_and_forwarded(api):
    client, be = api()
    fb = {"transaction_id": "TXS1", "outcome": "CHARGEBACK", "reference": "VDC-778"}
    assert client.post("/v1/feedback/vodacom", json=fb, headers={"X-API-Key": KEYS["vodacom"]}).status_code == 401
    r = post_signed(client, "vodacom", fb, path="/v1/feedback/vodacom")
    assert r.status_code == 200, r.text
    headers, body = be.backoffice_calls[0]
    assert body["provider"] == "vodacom" and body["outcome"] == "CHARGEBACK"
    assert headers["x-internal-key"] == "dev-internal-key"
    assert post_signed(client, "vodacom", {**fb, "outcome": "PEUT_ETRE"}, path="/v1/feedback/vodacom").status_code == 422
    be.backoffice = lambda body: httpx.Response(404, json={"detail": "transaction inconnue"})
    assert post_signed(client, "vodacom", fb, path="/v1/feedback/vodacom").status_code == 404


# ---------------------------------------------------------------------- IP autorisées et mTLS
def fake_request(peer: str, headers: dict, path="/v1/transactions/vodacom") -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "raw_path": path.encode(),
                    "query_string": b"", "client": (peer, 40000), "scheme": "http", "server": ("test", 80),
                    "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]})


def security(**kw) -> OperatorSecurity:
    return OperatorSecurity(api_keys={"k-voda": "vodacom"}, hmac_secrets={}, require_signature=False,
                            trusted_proxies=parse_networks("172.16.0.0/12"), **kw)


def test_ip_allowlist_uses_real_ip_only_from_trusted_proxy():
    sec = security(ip_allowlist=parse_allowlist("vodacom:10.10.0.0/16|41.243.1.7"))
    sec.check("vodacom", fake_request("172.18.0.9", {"x-api-key": "k-voda", "x-real-ip": "10.10.4.2"}), b"")
    with pytest.raises(SecurityRejection) as e:     # IP réelle hors de la liste
        sec.check("vodacom", fake_request("172.18.0.9", {"x-api-key": "k-voda", "x-real-ip": "10.99.0.1"}), b"")
    assert e.value.status == 403
    with pytest.raises(SecurityRejection):          # X-Real-IP forgé par un appelant direct : ignoré
        sec.check("vodacom", fake_request("203.0.113.5", {"x-api-key": "k-voda", "x-real-ip": "10.10.4.2"}), b"")


def test_mtls_certificate_must_match_operator():
    sec = security(require_mtls=True)
    ok = {"x-api-key": "k-voda", "x-client-verify": "SUCCESS", "x-client-cert-cn": "vodacom"}
    sec.check("vodacom", fake_request("172.18.0.9", ok), b"")
    with pytest.raises(SecurityRejection, match="n'appartient pas"):
        sec.check("vodacom", fake_request("172.18.0.9", {**ok, "x-client-cert-cn": "airtel"}), b"")
    with pytest.raises(SecurityRejection, match="absent ou invalide"):
        sec.check("vodacom", fake_request("172.18.0.9", {**ok, "x-client-verify": "FAILED:expired"}), b"")
    with pytest.raises(SecurityRejection, match="absent ou invalide"):   # en-têtes forgés hors nginx
        sec.check("vodacom", fake_request("198.51.100.4", ok), b"")


# ---------------------------------------------------------------------- signalements SMS (NLP)
SMS = {"report_id": "S1", "received_at": "2025-11-01T09:00:00", "sender": "+243 81 555 0001",
       "reporter": "0822000001", "text": "Je vous ai envoyé 50 000 FC par erreur, renvoyez au 0 97 123 4567 svp. "
                                         "Mon autre numéro : 0822000001"}


def test_scam_report_masks_text_and_excludes_reporter(api):
    client, backends = api()
    r = post_signed(client, "vodacom", SMS, path="/v1/scam-reports/vodacom")
    assert r.status_code == 200, r.text
    sent = backends.scam_calls[-1]
    assert "<NUMERO>" in sent["masked_text"] and "971234567" not in sent["masked_text"]
    assert sent["numbers"] == ["243815550001", "243971234567"]      # expéditeur puis numéro du texte ; jamais le client
    assert sent["sender_is_number"] is True and sent["report_id"] == "vodacom:S1"


def test_scam_report_numbers_are_pseudonymized(api):
    client, backends = api(PSEUDONYMIZE_IDS=True, PSEUDONYMIZATION_KEY="cle-de-test-tres-longue-2026")
    assert post_signed(client, "vodacom", SMS, path="/v1/scam-reports/vodacom").status_code == 200
    sent = backends.scam_calls[-1]
    assert all(n.startswith("P") and not n.isdigit() for n in sent["numbers"])
    assert "0 97 123" not in sent["masked_text"]


def test_scam_report_requires_signature(api):
    client, _ = api()
    r = client.post("/v1/scam-reports/vodacom", json=SMS, headers={"X-API-Key": KEYS["vodacom"]})
    assert r.status_code == 401


def test_shadow_mode_never_applies_the_decision(monkeypatch):
    """Pilote silencieux : l'opérateur reçoit APPROVE, la vraie décision est conservée."""
    assert m.parse_shadow("Vodacom, visa", {"vodacom", "airtel", "visa"}) == {"vodacom", "visa"}
    assert m.parse_shadow("all", {"vodacom", "airtel"}) == {"vodacom", "airtel"}
    with pytest.raises(RuntimeError):
        m.parse_shadow("mpesa", {"vodacom"})
    monkeypatch.setattr(m.RT, "shadow", {"vodacom"})
    r = m.apply_shadow("vodacom", {"action": "BLOCK"})
    assert r == {"action": "APPROVE", "shadow_mode": True, "shadow_action": "BLOCK"}
    assert m.apply_shadow("airtel", {"action": "BLOCK"}) == {"action": "BLOCK"}
