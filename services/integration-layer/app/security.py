"""
Sécurité des échanges avec les opérateurs — défense en profondeur. Chaque couche arrête ce
que la précédente aurait laissé passer :

    réseau      VPN / lien dédié entre l'opérateur et la plateforme (infrastructure, voir
                docs/INTEGRATION_OPERATEURS.md) : l'API n'est pas joignable depuis Internet
    transport   mTLS : l'opérateur présente un certificat signé par l'autorité de la
                plateforme ; nginx le vérifie et transmet son nom (CN), comparé ici à l'opérateur
    origine     liste d'adresses IP autorisées par opérateur
    identité    clé d'API propre à chaque opérateur
    message     signature HMAC du corps + horodatage : intégrité, authenticité, anti-rejeu

Les en-têtes posés par nginx (adresse IP réelle, résultat mTLS) ne sont crus que s'ils
viennent d'un proxy de confiance (TRUSTED_PROXIES), sinon n'importe qui pourrait les forger.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field

from fastapi import Request

from shared.security.request_signing import SignatureError, verify


class SecurityRejection(Exception):
    def __init__(self, status: int, outcome: str, detail: str):
        super().__init__(detail)
        self.status, self.outcome, self.detail = status, outcome, detail


def parse_networks(spec: str) -> list:
    return [ipaddress.ip_network(c.strip(), strict=False) for c in (spec or "").replace("|", ",").split(",")
            if c.strip()]


def parse_allowlist(spec: str) -> dict[str, list]:
    """'vodacom:10.10.0.0/16|10.20.0.5,airtel:10.30.0.0/16' -> {opérateur: [réseaux]}"""
    out: dict[str, list] = {}
    for part in (spec or "").split(","):
        if ":" in part:
            provider, cidrs = part.split(":", 1)
            out.setdefault(provider.strip(), []).extend(parse_networks(cidrs))
    return out


@dataclass
class OperatorSecurity:
    api_keys: dict[str, str]                       # clé d'API -> opérateur
    hmac_secrets: dict[str, str]                   # opérateur -> secret HMAC
    require_signature: bool = True
    signature_window_s: int = 300
    ip_allowlist: dict[str, list] = field(default_factory=dict)
    trusted_proxies: list = field(default_factory=list)
    require_mtls: bool = False

    def _from_trusted_proxy(self, request: Request) -> bool:
        peer = request.client.host if request.client else None
        try:
            return peer is not None and any(ipaddress.ip_address(peer) in n for n in self.trusted_proxies)
        except ValueError:
            return False

    def client_ip(self, request: Request) -> str | None:
        """Adresse de l'opérateur : X-Real-IP posée par nginx si la requête vient de nginx."""
        if self._from_trusted_proxy(request) and request.headers.get("x-real-ip"):
            return request.headers["x-real-ip"].strip()
        return request.client.host if request.client else None

    def check(self, provider: str, request: Request, body: bytes) -> None:
        # identité
        if self.api_keys.get(request.headers.get("x-api-key", "")) != provider:
            raise SecurityRejection(401, "unauthorized", "clé d'API opérateur invalide")
        # origine
        allowed = self.ip_allowlist.get(provider)
        if allowed:
            ip = self.client_ip(request)
            try:
                ok = ip is not None and any(ipaddress.ip_address(ip) in n for n in allowed)
            except ValueError:
                ok = False
            if not ok:
                raise SecurityRejection(403, "ip_refused", f"adresse IP non autorisée pour {provider}")
        # transport (mTLS vérifié par nginx)
        if self.require_mtls:
            if not self._from_trusted_proxy(request) or request.headers.get("x-client-verify") != "SUCCESS":
                raise SecurityRejection(403, "mtls_refused", "certificat client (mTLS) absent ou invalide")
            if request.headers.get("x-client-cert-cn") != provider:
                raise SecurityRejection(403, "mtls_refused", "le certificat client n'appartient pas à cet opérateur")
        # message
        if self.require_signature:
            secret = self.hmac_secrets.get(provider)
            if not secret:
                raise SecurityRejection(401, "unauthorized", f"aucun secret de signature configuré pour {provider}")
            try:
                verify(secret, request.headers.get("x-timestamp"), request.headers.get("x-signature"),
                       request.method, request.url.path, body, self.signature_window_s)
            except SignatureError as e:
                raise SecurityRejection(401, "bad_signature", str(e)) from None
