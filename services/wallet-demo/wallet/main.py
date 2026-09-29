"""
Application « portefeuille client » de démonstration (web mobile installable) et serveur
opérateur simulé.

Montre au jury ce que vit le CLIENT quand la plateforme décide en temps réel : envoi normal,
confirmation par PIN, alerte contre l'arnaque « envoi par erreur », vérification en agence
après un changement de SIM, blocage, renvoi après coupure (idempotence), signalement d'une
fraude (étiquette + réputation).

Horloge : la plateforme a été amorcée jusqu'au début de la période de test ; la démo
commence donc à cette date (+12 h) et avance en temps réel.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from wallet.operator import USD_CDF, OperatorGateway, backstage, customer_view
from wallet.personas import build_personas

STATIC = Path(__file__).absolute().parent / "static"
DEMO_PIN = "1234"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    INTEGRATION_URL: str = "http://localhost:8002"
    OPERATOR_API_KEYS: str = ("vodacom:dev-vodacom-key,airtel:dev-airtel-key,"
                              "orange:dev-orange-key,visa:dev-visa-key")
    OPERATOR_HMAC_SECRETS: str = ("vodacom:dev-vodacom-hmac,airtel:dev-airtel-hmac,"
                                  "orange:dev-orange-hmac,visa:dev-visa-hmac")
    DATA_DIR: str = "ml/data/raw"
    PREPROCESSING_JSON: str = "ml/artifacts/preprocessing.json"


settings = Settings()
state: dict = {"sessions": {}, "lock": threading.Lock()}


def init(gateway: OperatorGateway | None = None) -> None:
    prep = json.loads(Path(settings.PREPROCESSING_JSON).read_text(encoding="utf-8"))
    start = pd.Timestamp(prep["split_bounds"]["test_from"]).ceil("h") + pd.Timedelta(hours=12)
    personas, mule = build_personas(Path(settings.DATA_DIR), pd.Timestamp(prep["split_bounds"]["test_from"]))
    state.update(personas={p.id: p for p in personas}, mule=mule, demo_start=start.to_pydatetime(),
                 real_start=time.time(),
                 gateway=gateway or OperatorGateway(settings.INTEGRATION_URL, settings.OPERATOR_API_KEYS,
                                                    settings.OPERATOR_HMAC_SECRETS))


app = FastAPI(title="Portefeuille client — démonstration", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def _startup() -> None:
    if "personas" not in state:
        init()


# ---------------------------------------------------------------------- session du client
class Session:
    def __init__(self, persona):
        self.persona = persona
        self.balance = persona.balance_usd
        self.last_ts = state["demo_start"] + timedelta(seconds=time.time() - state["real_start"])
        self.sim_swapped = False
        self.device_id = persona.device_id
        self.history: list[dict] = []

    def now(self) -> datetime:
        """Horloge de démonstration COMMUNE à tous les clients : les événements de deux clients
        (un signalement de SMS par l'un, un envoi par l'autre) arrivent dans le bon ordre."""
        real = state["demo_start"] + timedelta(seconds=time.time() - state["real_start"])
        with state["lock"]:
            state["clock"] = max(real, state.get("clock", real) + timedelta(seconds=20))
            self.last_ts = state["clock"]
        return self.last_ts.replace(microsecond=0)

    def item(self, tx_id: str) -> dict:
        for h in self.history:
            if h["tx_id"] == tx_id:
                return h
        raise HTTPException(404, "opération inconnue")


def _session(sid: str) -> Session:
    s = state["sessions"].get(sid)
    if s is None:
        raise HTTPException(404, "session expirée : choisissez un profil")
    return s


def _submit(s: Session, kind: str, tx_type: str, amount_usd: float, currency: str, counterparty=None,
            agent=None, label: str = "") -> dict:
    if amount_usd <= 0:
        raise HTTPException(422, "montant invalide")
    debit = tx_type != "P2P_RECEIVE"
    if debit and amount_usd > s.balance + 1e-9:
        raise HTTPException(422, "solde insuffisant")
    ts = s.now()
    tx_id = f"DEMO{uuid.uuid4().hex[:12].upper()}"
    swap_at = ts - timedelta(hours=2) if s.sim_swapped else None
    unified = OperatorGateway.unified(s.persona, tx_id=tx_id, ts=ts, tx_type=tx_type, amount_usd=amount_usd,
                                      currency=currency, balance_usd=s.balance, device_id=s.device_id,
                                      counterparty=counterparty, agent=agent, sim_swap_at=swap_at)
    gw: OperatorGateway = state["gateway"]
    provider, path, body = gw.build(s.persona, unified)
    decision = gw.send(provider, path, body)
    view = customer_view(decision)
    status = {"success": "EXECUTEE", "pin": "EN_ATTENTE_PIN", "agency": "EN_ATTENTE_AGENCE",
              "blocked": "REFUSEE"}.get(view["screen"], "ECHEC")
    if status == "EXECUTEE":
        s.balance += -amount_usd if debit else amount_usd
    item = {"tx_id": tx_id, "kind": kind, "label": label, "tx_type": tx_type, "amount_usd": round(amount_usd, 2),
            "currency": currency, "time": ts.isoformat(), "status": status, "provider": provider,
            "path": path, "body": body.decode(), "debit": debit}
    s.history.insert(0, item)
    return {"tx_id": tx_id, "status": status, "balance_usd": round(s.balance, 2), "view": view,
            "backstage": backstage(decision)}


def _usd(amount: float, currency: str) -> float:
    return amount / USD_CDF if currency == "CDF" else amount


# ---------------------------------------------------------------------- API de l'appli
class Start(BaseModel):
    persona_id: str


class Transfer(BaseModel):
    session_id: str
    to_wallet: str = Field(..., min_length=6, max_length=20)
    to_name: str | None = None
    amount: float = Field(..., gt=0)
    currency: str = Field("CDF", pattern="^(CDF|USD)$")


class Withdraw(BaseModel):
    session_id: str
    amount: float = Field(..., gt=0)
    currency: str = Field("CDF", pattern="^(CDF|USD)$")
    other_agent: bool = False


class Action(BaseModel):
    session_id: str
    tx_id: str
    pin: str | None = None


class Toggle(BaseModel):
    session_id: str
    enabled: bool


@app.get("/api/personas")
def personas():
    return {"personas": [p.public() for p in state["personas"].values()], "known_mule": state["mule"],
            "demo_pin": DEMO_PIN, "usd_cdf": USD_CDF}


@app.post("/api/session")
def start(body: Start):
    p = state["personas"].get(body.persona_id)
    if p is None:
        raise HTTPException(404, "profil inconnu")
    sid = secrets.token_urlsafe(12)
    with state["lock"]:
        state["sessions"][sid] = Session(p)
    return {"session_id": sid, "persona": p.public(), "balance_usd": p.balance_usd}


@app.get("/api/session/{sid}")
def session_info(sid: str):
    s = _session(sid)
    return {"persona": s.persona.public(), "balance_usd": round(s.balance, 2), "sim_swapped": s.sim_swapped,
            "history": [{k: v for k, v in h.items() if k not in ("body", "path")} for h in s.history],
            "demo_time": s.last_ts.isoformat(timespec="minutes")}


@app.post("/api/transfer")
def transfer(body: Transfer):
    s = _session(body.session_id)
    return _submit(s, "envoi", "P2P_SEND", _usd(body.amount, body.currency), body.currency,
                   counterparty=body.to_wallet, label=f"Envoi à {body.to_name or body.to_wallet}")


@app.post("/api/withdraw")
def withdraw(body: Withdraw):
    s = _session(body.session_id)
    agent = s.persona.agent_id
    if body.other_agent or agent is None:
        agent = f"A{900000 + secrets.randbelow(99999)}"   # agent jamais utilisé par ce client
    return _submit(s, "retrait", "CASH_OUT", _usd(body.amount, body.currency), body.currency, agent=agent,
                   label=f"Retrait chez l'agent {agent}")


@app.post("/api/scenario/unknown-sender")
def unknown_sender(body: Toggle):
    """Arnaque « envoi par erreur », étape 1 : un inconnu envoie un petit montant."""
    s = _session(body.session_id)
    scammer = f"24389{secrets.randbelow(10 ** 7):07d}"
    res = _submit(s, "reception", "P2P_RECEIVE", 5.0, s.persona.currency, counterparty=scammer,
                  label=f"Reçu de {scammer}")
    amount = round(5.0 * USD_CDF) if s.persona.currency == "CDF" else 5.0
    asked = amount * 10
    res["scam"] = {"from": scammer, "received": amount, "asked": asked, "currency": s.persona.currency,
                   "message": f"Bonjour, je vous ai envoyé {asked:,.0f} {s.persona.currency} par erreur, c'était "
                              f"pour ma mère malade. Renvoyez-les s'il vous plaît, Dieu vous bénisse."
                              .replace(",", " ")}
    return res


@app.post("/api/scenario/sim-swap")
def sim_swap(body: Toggle):
    """SIM swap : le fraudeur a obtenu une nouvelle SIM et l'utilise dans SON téléphone."""
    s = _session(body.session_id)
    s.sim_swapped = body.enabled
    s.device_id = f"DEVFRAUD{uuid.uuid4().hex[:8]}" if body.enabled else s.persona.device_id
    return {"sim_swapped": s.sim_swapped}


@app.post("/api/confirm")
def confirm(body: Action):
    s = _session(body.session_id)
    h = s.item(body.tx_id)
    if h["status"] != "EN_ATTENTE_PIN":
        raise HTTPException(409, "rien à confirmer")
    if body.pin is None:
        h["status"] = "ANNULEE"
        return {"status": "ANNULEE", "balance_usd": round(s.balance, 2)}
    if body.pin != DEMO_PIN:
        raise HTTPException(401, "code PIN incorrect")
    h["status"] = "EXECUTEE"
    s.balance += -h["amount_usd"] if h["debit"] else h["amount_usd"]
    return {"status": "EXECUTEE", "balance_usd": round(s.balance, 2)}


@app.post("/api/resend")
def resend(body: Action):
    """Coupure réseau simulée : l'opérateur renvoie EXACTEMENT le même message."""
    s = _session(body.session_id)
    h = s.item(body.tx_id)
    decision = state["gateway"].send(h["provider"], h["path"], h["body"].encode())
    return {"tx_id": h["tx_id"], "idempotent_replay": decision.get("idempotent_replay", False),
            "action": decision.get("action"), "balance_usd": round(s.balance, 2), "backstage": backstage(decision)}


@app.post("/api/report")
def report(body: Action):
    """Le client signale une fraude : l'opérateur transmet une plainte à la plateforme."""
    s = _session(body.session_id)
    h = s.item(body.tx_id)
    res = state["gateway"].feedback(h["provider"], h["tx_id"], "CUSTOMER_COMPLAINT",
                                    reference=f"PLAINTE-{h['tx_id'][-6:]}")
    if res.get("http_status") == 200:
        h["reported"] = True
    return res


class SmsReport(BaseModel):
    session_id: str
    sender: str = Field(..., max_length=40)
    text: str = Field(..., min_length=1, max_length=1000)


@app.post("/api/report-sms")
def report_sms(body: SmsReport):
    """Le client transfère le SMS suspect au numéro court de son opérateur (analyse NLP)."""
    s = _session(body.session_id)
    provider = s.persona.operator.lower()
    res = state["gateway"].scam_report(provider, f"SMS{uuid.uuid4().hex[:10].upper()}", s.now().isoformat(),
                                       body.text, body.sender, s.persona.wallet_id)
    if res.get("http_status") != 200:
        raise HTTPException(502, res.get("detail") or "signalement non transmis")
    return {"category": res["category_label"], "scam_probability": res["scam_probability"],
            "is_scam": res["is_scam"], "flagged_numbers": res["flagged_numbers"]}


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return FileResponse(STATIC / "sw.js", media_type="application/javascript")


@app.get("/health")
def health():
    return {"status": "ok", "personas": len(state.get("personas", {})), "sessions": len(state["sessions"])}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
