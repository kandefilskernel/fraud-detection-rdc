"""Application portefeuille de démonstration : personnages, messages signés au format de
l'opérateur, écrans selon la décision, PIN, SIM swap, renvoi identique, signalement.
La plateforme est simulée (httpx.MockTransport)."""
import json
import sys
from pathlib import Path

import httpx
import pandas as pd
import pytest

ROOT = Path(__file__).absolute().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "wallet-demo"), str(ROOT / "ml" / "generator")]

from fastapi.testclient import TestClient  # noqa: E402
from generate_synthetic_data import GeneratorConfig, SyntheticDataGenerator  # noqa: E402

from shared.security.request_signing import verify  # noqa: E402
from wallet import main  # noqa: E402
from wallet.operator import OperatorGateway, customer_view  # noqa: E402

CUTOFF = pd.Timestamp("2025-07-15")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    cfg = GeneratorConfig(n_users=400, n_days=60, seed=21)
    out = SyntheticDataGenerator(cfg).run()
    out["transactions"].to_csv(d / "transactions.csv", index=False)
    out["users"].to_csv(d / "users.csv", index=False)
    (d / "preprocessing.json").write_text(json.dumps({"split_bounds": {"test_from": str(CUTOFF)}}))
    return d


class Platform:
    """Plateforme simulée : répond selon self.decision et mémorise les requêtes reçues."""
    def __init__(self):
        self.requests = []
        self.decision = {"action": "APPROVE", "risk_level": "FAIBLE", "fraud_probability": 0.01}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.startswith("/v1/feedback/"):
            return httpx.Response(200, json={"label": 1, "changed": True})
        body = json.loads(request.content)
        tx_id = body.get("TransID") or body.get("id_transaction") or body.get("transaction", {}).get("id")
        return httpx.Response(200, json={"transaction_id": tx_id, **self.decision})


@pytest.fixture()
def app(data_dir):
    platform = Platform()
    main.settings.DATA_DIR = str(data_dir)
    main.settings.PREPROCESSING_JSON = str(data_dir / "preprocessing.json")
    main.state["sessions"].clear()
    gw = OperatorGateway("http://plateforme", main.settings.OPERATOR_API_KEYS, main.settings.OPERATOR_HMAC_SECRETS,
                         client=httpx.Client(base_url="http://plateforme", transport=httpx.MockTransport(platform.handler)))
    main.init(gw)
    with TestClient(main.app) as c:
        yield c, platform


def start(client) -> tuple[str, dict]:
    personas = client.get("/api/personas").json()["personas"]
    assert personas, "au moins un client exemplaire"
    p = personas[0]
    return client.post("/api/session", json={"persona_id": p["id"]}).json()["session_id"], p


def test_personas_are_real_customers_with_history(app):
    client, _ = app
    data = client.get("/api/personas").json()
    assert {p["operator"] for p in data["personas"]} <= {"VODACOM", "AIRTEL", "ORANGE"}
    p = data["personas"][0]
    assert p["contacts"] and p["balance_usd"] >= 0 and "device_id" not in p   # l'IMEI n'est pas exposé


def test_transfer_is_signed_in_operator_format_and_debits_on_approve(app):
    client, platform = app
    sid, p = start(client)
    r = client.post("/api/transfer", json={"session_id": sid, "to_wallet": p["contacts"][0]["wallet"],
                                           "amount": 1.0, "currency": "USD"}).json()
    assert r["status"] == "EXECUTEE" and r["view"]["screen"] == "success"
    assert r["balance_usd"] == pytest.approx(p["balance_usd"] - 1.0)
    req = platform.requests[-1]
    provider = p["operator"].lower()
    assert req.url.path == f"/v1/transactions/{provider}"
    verify(f"dev-{provider}-hmac", req.headers["x-timestamp"], req.headers["x-signature"], "POST",
           req.url.path, req.content)                                          # signature valide
    assert req.headers["x-api-key"] == f"dev-{provider}-key"


def test_verify_requires_pin_then_executes(app):
    client, platform = app
    platform.decision = {"action": "VERIFY", "verification_method": "PIN_USSD", "risk_level": "ELEVE",
                         "features": {"is_new_counterparty": 1.0}}
    sid, p = start(client)
    r = client.post("/api/transfer", json={"session_id": sid, "to_wallet": "243899000111", "amount": 2.0,
                                           "currency": "USD"}).json()
    assert r["status"] == "EN_ATTENTE_PIN" and r["view"]["screen"] == "pin"
    assert r["balance_usd"] == pytest.approx(p["balance_usd"])                # rien de débité avant le PIN
    assert client.post("/api/confirm", json={"session_id": sid, "tx_id": r["tx_id"], "pin": "0000"}).status_code == 401
    ok = client.post("/api/confirm", json={"session_id": sid, "tx_id": r["tx_id"], "pin": main.DEMO_PIN}).json()
    assert ok["status"] == "EXECUTEE" and ok["balance_usd"] == pytest.approx(p["balance_usd"] - 2.0)


def test_sim_swap_sends_sim_date_and_new_device(app):
    client, platform = app
    sid, p = start(client)
    client.post("/api/scenario/sim-swap", json={"session_id": sid, "enabled": True})
    client.post("/api/withdraw", json={"session_id": sid, "amount": 1.0, "currency": "USD"})
    body = json.loads(platform.requests[-1].content)
    sim_field = {"vodacom": "LastSimSwapTime", "orange": "date_dernier_changement_sim"}.get(p["operator"].lower())
    if sim_field:
        assert body[sim_field]
    else:
        assert body["subscriber"]["last_sim_swap"]


def test_resend_is_byte_identical_and_report_sends_complaint(app):
    client, platform = app
    sid, p = start(client)
    r = client.post("/api/transfer", json={"session_id": sid, "to_wallet": p["contacts"][0]["wallet"],
                                           "amount": 1.0, "currency": "USD"}).json()
    first = platform.requests[-1].content
    client.post("/api/resend", json={"session_id": sid, "tx_id": r["tx_id"]})
    assert platform.requests[-1].content == first                             # même message : idempotence
    rep = client.post("/api/report", json={"session_id": sid, "tx_id": r["tx_id"]}).json()
    fb = platform.requests[-1]
    assert rep["http_status"] == 200 and fb.url.path.startswith("/v1/feedback/")
    assert json.loads(fb.content)["outcome"] == "CUSTOMER_COMPLAINT"


def test_customer_views():
    scam = customer_view({"action": "VERIFY", "verification_method": "PIN_USSD",
                          "features": {"refund_ratio_to_cp": 2.4}})
    assert scam["screen"] == "pin" and "arnaque" in scam["warnings"][0]
    assert customer_view({"action": "VERIFY", "verification_method": "HORS_SIM"})["screen"] == "agency"
    assert customer_view({"action": "BLOCK", "features": {"rep_cp_frauds": 0.7}})["warnings"] == [
        "Ce numéro a déjà été signalé pour fraude."]
    assert customer_view({"action": "ERREUR", "detail": "x"})["screen"] == "error"
