"""Assistant d'enquête (RAG) : recherche de cas et de procédures, note extractive, appel à
Claude simulé (aucun appel réseau), minimisation des données envoyées, journal d'audit."""
import math
from datetime import datetime, timezone
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from sqlalchemy import insert

from app.assistant import synthesis
from app.assistant.knowledge import get_knowledge
from app.db import engine
from app.events import get_publisher
from app.main import app
from shared.database.models import ScoredTransaction
from shared.investigation.signals import describe

# profil « échange de SIM » : nouvel appareil, vidage, bénéficiaire récent, nuit
SIM_SWAP_FEATS = {
    "log_amount_usd": math.log1p(180), "amount_to_balance": 0.97, "is_near_full_drain": 1.0,
    "amount_to_kyc_limit": 0.6, "log_balance_before": math.log1p(185), "is_new_device": 1.0,
    "log_device_prior_uses": 0.0, "is_new_counterparty": 1.0, "cp_log_age_days": math.log1p(2),
    "cp_in_senders_7d": math.log1p(4), "type_P2P_SEND": 1.0, "is_night": 1.0, "night_unusual": 0.9,
    "log_history_days": math.log1p(200), "log_user_tx_count": math.log1p(150), "access_app": 1.0,
    "is_smartphone": 1.0, "hour_sin": math.sin(2 * math.pi * 2 / 24), "hour_cos": math.cos(2 * math.pi * 2 / 24),
}
IDS = {"user_id": "U-SECRET-77", "device_id": "DEV-SECRET-99", "counterparty_id": "243899000111"}


def seed_tx(tx_id: str, feats: dict, label=None, **over) -> None:
    values = dict(
        transaction_id=tx_id, tx_time=datetime(2025, 11, 6, 2, 10), scored_at=datetime.now(timezone.utc),
        channel="MOBILE_MONEY", operator="VODACOM", tx_type="P2P_SEND", access_channel="APP", amount=180.0,
        currency="USD", amount_usd=180.0, province="Kinshasa", fraud_probability=0.91, risk_level="CRITIQUE",
        action="BLOCK", reason="probabilité de fraude très élevée", rules_triggered=["NOUVEL_APPAREIL_VIDAGE_COMPTE"],
        model_version="t", latency_ms=15.0, degraded=False, label=label, features=feats,
        explanation={"top_features": [{"feature": "is_new_device", "shap": 1.2, "value": 1.0},
                                      {"feature": "amount_to_balance", "shap": 0.9, "value": 0.97}]},
        **IDS)
    values.update(over)
    with engine.begin() as c:
        c.execute(insert(ScoredTransaction).values(**values))


class FakePublisher:
    def __init__(self):
        self.audits = []

    def audit(self, actor, action, entity, entity_id, details):
        self.audits.append((action, entity_id, details))


@pytest.fixture
def publisher():
    pub = FakePublisher()
    app.dependency_overrides[get_publisher] = lambda: pub
    yield pub
    app.dependency_overrides.pop(get_publisher, None)


# ---------------------------------------------------------------------- recherche
def test_knowledge_loads_archive_and_procedures():
    kn = get_knowledge()
    assert len(kn.cases.cases) > 1000 and len(kn.procedures) >= 20
    assert {c["outcome"] for c in kn.cases.cases} == {"FRAUDE_CONFIRMEE", "FAUX_POSITIF"}


def test_similar_cases_point_to_sim_swap_and_procedure_follows():
    kn = get_knowledge()
    from shared.investigation.case_index import summarize_neighbors
    s = summarize_neighbors(kn.cases.search(SIM_SWAP_FEATS, "MOBILE_MONEY", k=8))
    assert s["n_confirmed"] >= 5
    assert s["typologies"][0]["typology"] in ("SIM_SWAP", "ACCOUNT_TAKEOVER")
    procs = kn.search_procedures("appareil jamais utilisé vidage du compte", ["SIM_SWAP"], ["NOUVEL_APPAREIL_VIDAGE_COMPTE"])
    assert procs[0].typologies & {"SIM_SWAP", "ACCOUNT_TAKEOVER"}


def test_rule_brings_its_procedure():
    kn = get_knowledge()
    procs = kn.search_procedures("renvoi montant reçu", [], ["RENVOI_TRES_SUPERIEUR_AU_RECU"])
    assert any("SOCIAL_ENGINEERING" in p.typologies for p in procs)


def test_facts_never_contain_identifiers():
    facts = describe({**SIM_SWAP_FEATS, "rep_cp_frauds": math.log1p(3)}, "P2P_SEND")
    assert "Ce bénéficiaire : déjà lié à 3 fraude(s) confirmée(s)" in facts
    assert "Appareil jamais utilisé par ce client" in facts


# ---------------------------------------------------------------------- endpoint, mode extractif
def test_assistant_extractive_mode(client, analyst, publisher, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    seed_tx("TX-RAG-1", SIM_SWAP_FEATS)
    seed_tx("TX-RAG-PAST", SIM_SWAP_FEATS, label=1,
            tx_time=datetime(2025, 11, 4, 3, 0), user_id="U-AUTRE")   # même appareil, fraude confirmée
    r = client.post("/transactions/TX-RAG-1/assistant", headers=analyst)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "extractif" and "clé" in body["notice"]
    assert "### Vérifications recommandées" in body["synthesis"] and "[P" in body["synthesis"]
    assert body["similar_cases"] and body["procedures"]
    assert any(c.get("transaction_id") == "TX-RAG-PAST" for c in body["similar_cases"])   # verdict plateforme
    device = next(e for e in body["entity_links"] if e["entity"] == "l'appareil")
    assert device["confirmed_frauds"] == 1
    action, entity_id, details = publisher.audits[-1]
    assert action == "ASSISTANT_ENQUETE" and entity_id == "TX-RAG-1" and details["sources"]
    # 2e appel identique : servi depuis le cache (pas de nouvelle facturation)
    assert client.post("/transactions/TX-RAG-1/assistant", headers=analyst).json()["cached"] is True
    assert client.post("/transactions/INCONNUE/assistant", headers=analyst).status_code == 404


def test_assistant_requires_login(client):
    assert client.post("/transactions/TX-RAG-1/assistant").status_code == 401


# ---------------------------------------------------------------------- mode LLM (client simulé)
class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, msg=None, error=None):
        self.calls, self.msg, self.error = [], msg, error
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return FakeStream(self.msg)


def fake_message(text="### Synthèse\nNote [C1] [P1]", stop_reason="end_turn", fallback=False):
    usage = SimpleNamespace(input_tokens=1200, output_tokens=300,
                            iterations=[SimpleNamespace(type="fallback_message")] if fallback else None)
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           stop_reason=stop_reason, model="claude-opus-5", usage=usage)


def _context():
    from app.assistant.retrieval import build_context
    from app.db import SessionLocal
    from sqlalchemy import select
    with SessionLocal() as db:
        tx = db.scalar(select(ScoredTransaction).where(ScoredTransaction.transaction_id == "TX-RAG-1"))
        return build_context(db, tx, get_knowledge())


def test_llm_request_shape_and_no_identifier_leaves(client, analyst, publisher):
    fake = FakeClient(fake_message())
    note = synthesis.synthesize(_context(), client=fake)
    assert note.mode == "llm" and note.markdown.startswith("### Synthèse") and not note.fallback_used
    kw = fake.calls[0]
    assert kw["model"] == "claude-opus-5" and kw["thinking"] == {"type": "adaptive"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    sent = kw["system"] + kw["messages"][0]["content"]
    for secret in (*IDS.values(), "TX-RAG-1", "TX-RAG-PAST"):
        assert secret not in sent                  # minimisation : aucun identifiant ne sort
    assert "[C1]" in sent and "[P1]" in sent


def test_llm_refusal_and_errors_fall_back_to_extractive():
    ctx = _context()
    refused = synthesis.synthesize(ctx, client=FakeClient(fake_message(stop_reason="refusal")))
    assert refused.mode == "extractif" and "refusé" in refused.notice
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    down = synthesis.synthesize(ctx, client=FakeClient(error=anthropic.APIConnectionError(request=req)))
    assert down.mode == "extractif" and "injoignable" in down.notice
    served = synthesis.synthesize(ctx, client=FakeClient(fake_message(fallback=True)))
    assert served.mode == "llm" and served.fallback_used
