"""
Serveur de l'opérateur (simulé) : ce que ferait le système de Vodacom, Airtel ou Orange
entre l'application du client et la plateforme de détection.

    appli du client ──► serveur opérateur ──► plateforme (/v1/transactions/{opérateur})
                        · message au format PROPRIÉTAIRE de l'opérateur (mêmes adaptateurs
                          que l'integration-layer, dans l'autre sens)
                        · signature HMAC avec le secret de l'opérateur (jamais dans le téléphone)
                        · traduction de la décision en écran compréhensible par le client
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

# Formats propriétaires des opérateurs : les adaptateurs de l'integration-layer (from_unified)
_OPFMT = os.getenv("OPFMT_DIR") or str(Path(__file__).absolute().parents[2] / "integration-layer" / "app")
if _OPFMT not in sys.path:
    sys.path.insert(0, _OPFMT)

from adapters.airtel_adapter import AirtelAdapter  # noqa: E402
from adapters.orange_adapter import OrangeAdapter  # noqa: E402
from adapters.vodacom_adapter import VodacomAdapter  # noqa: E402
from shared.security.request_signing import parse_secrets, signed_headers  # noqa: E402

ADAPTERS = {"vodacom": VodacomAdapter(), "airtel": AirtelAdapter(), "orange": OrangeAdapter()}
USD_CDF = 2850.0

# Libellés des variables pour le panneau « coulisses » (jury)
FEATURE_LABELS = {
    "is_new_device": "nouvel appareil", "log_device_prior_uses": "usages de l'appareil",
    "device_n_users": "autres clients sur cet appareil", "amount_to_balance": "part du solde",
    "is_near_full_drain": "vidage du compte", "is_new_counterparty": "destinataire jamais utilisé",
    "counterparty_n_users": "clients liés au destinataire", "amount_zscore_user": "montant inhabituel",
    "amount_ratio_user_mean": "montant vs habitude", "log_amount_usd": "montant",
    "refund_ratio_to_cp": "renvoi > montant reçu (arnaque « erreur »)",
    "log_mins_since_received_from_cp": "délai depuis la réception", "cp_log_age_days": "ancienneté du destinataire",
    "cp_in_senders_7d": "expéditeurs vers ce destinataire", "cp_in_new_ratio_7d": "part d'inconnus qui le paient",
    "rep_cp_frauds": "destinataire déjà signalé", "rep_device_frauds": "appareil déjà signalé",
    "rep_agent_frauds_7d": "agent déjà signalé", "tx_count_1h": "opérations dans l'heure",
    "log_secs_since_last_tx": "temps depuis la dernière opération", "is_new_province": "nouvelle province",
    "hour_deviation_user": "heure inhabituelle", "amount_to_kyc_limit": "montant / plafond KYC",
    "in_new_senders_24h": "inconnus qui vous ont payé (24 h)", "passthrough_ratio": "argent reçu puis ressorti",
}


class OperatorGateway:
    def __init__(self, base_url: str, api_keys_spec: str, secrets_spec: str, client: httpx.Client | None = None):
        self.keys = {p: k for k, p in (pair.split(":", 1)[::-1] for pair in api_keys_spec.split(",") if ":" in pair)}
        self.secrets = parse_secrets(secrets_spec)
        self.http = client or httpx.Client(base_url=base_url, timeout=5.0)

    # ------------------------------------------------------------------ message opérateur
    @staticmethod
    def unified(persona, *, tx_id: str, ts: datetime, tx_type: str, amount_usd: float, currency: str,
                balance_usd: float, device_id: str, counterparty: str | None = None, agent: str | None = None,
                sim_swap_at: datetime | None = None) -> dict:
        amount = round(amount_usd * USD_CDF) if currency == "CDF" else round(amount_usd, 2)
        access = "AGENT" if tx_type in ("CASH_OUT", "CASH_IN") else "APP"
        return {
            "transaction_id": tx_id, "timestamp": ts, "user_id": persona.user_id, "wallet_id": persona.wallet_id,
            "channel": "MOBILE_MONEY", "operator": persona.operator, "tx_type": tx_type, "access_channel": access,
            "amount": amount, "currency": currency, "amount_usd": round(amount_usd, 2),
            "balance_before_usd": round(balance_usd, 2), "status": None, "counterparty_id": counterparty,
            "merchant_id": None, "merchant_category": None, "merchant_country": None, "agent_id": agent,
            "device_id": device_id, "device_type": persona.device_type,
            "ip_country": "CD" if access == "APP" else None, "location_province": persona.province,
            "sim_swap_at": sim_swap_at,
        }

    def build(self, persona, unified: dict) -> tuple[str, str, bytes]:
        provider = persona.operator.lower()
        body = json.dumps(ADAPTERS[provider].from_unified(unified), default=str).encode()
        return provider, f"/v1/transactions/{provider}", body

    def _post(self, provider: str, path: str, body: bytes) -> httpx.Response:
        headers = {"X-API-Key": self.keys[provider], "Content-Type": "application/json",
                   **signed_headers(self.secrets[provider], "POST", path, body)}   # signature fraîche à chaque envoi
        return self.http.post(path, content=body, headers=headers)

    def send(self, provider: str, path: str, body: bytes) -> dict:
        r = self._post(provider, path, body)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text}
        if r.status_code != 200:
            return {"action": "ERREUR", "http_status": r.status_code, "detail": data.get("detail")}
        return data

    def scam_report(self, provider: str, report_id: str, received_at: str, text: str, sender: str,
                    reporter: str) -> dict:
        """Le client transfère un SMS suspect au numéro court de signalement de son opérateur."""
        path = f"/v1/scam-reports/{provider}"
        body = json.dumps({"report_id": report_id, "received_at": received_at, "text": text,
                           "sender": sender, "reporter": reporter}).encode()
        r = self._post(provider, path, body)
        return {"http_status": r.status_code, **(r.json() if r.content else {})}

    def feedback(self, provider: str, tx_id: str, outcome: str, reference: str) -> dict:
        path = f"/v1/feedback/{provider}"
        body = json.dumps({"transaction_id": tx_id, "outcome": outcome, "reference": reference}).encode()
        r = self._post(provider, path, body)
        return {"http_status": r.status_code, **(r.json() if r.content else {})}


# ---------------------------------------------------------------------- écran du client
def customer_view(decision: dict) -> dict:
    """Ce que l'application affiche au client, selon la décision de la plateforme."""
    action = decision.get("action")
    feats = decision.get("features") or {}
    warnings = []
    # refund_ratio_to_cp = log(1 + montant renvoyé / montant reçu) : > 0,79 <=> renvoi > 1,2 × reçu
    if feats.get("refund_ratio_to_cp", 0) > 0.79:
        warnings.append("Ce numéro vient de vous envoyer de l'argent et vous lui renvoyez BEAUCOUP plus. "
                        "C'est l'arnaque la plus courante (« je me suis trompé de numéro »). Ne continuez que si "
                        "vous connaissez cette personne.")
    if "BENEFICIAIRE_SIGNALE_PAR_SMS" in (decision.get("rules_triggered") or []):
        warnings.append("D'autres clients ont signalé ce numéro pour des SMS d'arnaque. Ne continuez que si "
                        "vous connaissez personnellement cette personne.")
    if feats.get("rep_cp_frauds", 0) > 0:
        warnings.append("Ce numéro a déjà été signalé pour fraude.")
    elif feats.get("is_new_counterparty", 0) > 0 and action != "APPROVE":
        warnings.append("Vous n'avez jamais envoyé d'argent à ce numéro.")

    if action == "APPROVE":
        return {"screen": "success", "title": "Opération réussie", "message": "", "warnings": []}
    if action == "VERIFY":
        if decision.get("verification_method") == "HORS_SIM":
            return {"screen": "agency", "title": "Vérification en agence",
                    "message": "Votre carte SIM a été changée récemment. Par sécurité, un code envoyé par SMS "
                               "ne suffit pas : confirmez cette opération dans une agence avec votre pièce "
                               "d'identité.", "warnings": warnings}
        return {"screen": "pin", "title": "Confirmation requise",
                "message": "Pour votre sécurité, confirmez cette opération avec votre code PIN.", "warnings": warnings}
    if action == "BLOCK":
        return {"screen": "blocked", "title": "Opération refusée",
                "message": "Cette opération a été bloquée pour protéger votre compte. Si vous n'en êtes pas "
                           "l'auteur, appelez le service client.", "warnings": warnings}
    return {"screen": "error", "title": "Service momentanément indisponible",
            "message": decision.get("detail") or "Réessayez dans un instant.", "warnings": []}


def backstage(decision: dict) -> dict:
    """Panneau « coulisses » pour le jury : ce que la plateforme a calculé."""
    expl = decision.get("explanation") or {}
    return {
        "action": decision.get("action"), "risk_level": decision.get("risk_level"),
        "fraud_probability": decision.get("fraud_probability"), "reason": decision.get("reason"),
        "rules_triggered": decision.get("rules_triggered", []),
        "verification_method": decision.get("verification_method"),
        "top_features": [{"label": FEATURE_LABELS.get(f["feature"], f["feature"]), "shap": f["shap"],
                          "value": f["value"]} for f in expl.get("top_features", [])],
        "branch_contributions": expl.get("branch_contributions", {}),
        "latency_ms": decision.get("latency_ms"), "end_to_end_ms": decision.get("end_to_end_ms"),
        "idempotent_replay": decision.get("idempotent_replay", False),
        "model_version": decision.get("model_version"),
    }
