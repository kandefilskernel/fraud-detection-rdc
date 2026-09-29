"""
Génère une PKI de TEST pour le mTLS entre les opérateurs et la plateforme :

    infra/nginx/certs/operators-ca.crt / .key   autorité de certification de la plateforme
    infra/nginx/certs/server.crt / .key          certificat du serveur (localhost, nginx)
    infra/nginx/certs/<opérateur>.crt / .key     certificat client de chaque opérateur (CN = vodacom...)

En production : une vraie autorité (interne ou commerciale), des clés générées CHEZ chaque
opérateur (seule la demande de signature CSR est transmise), des durées courtes et une liste
de révocation. Ce dossier contient des clés privées : il est exclu de Git.

Usage : python scripts/security/generate_dev_pki.py [--out infra/nginx/certs] [--force]
"""
from __future__ import annotations

import argparse
import ipaddress
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

OPERATORS = {"vodacom": "Vodacom Congo", "airtel": "Airtel Money RDC", "orange": "Orange Money RDC",
             "visa": "Emetteur carte Visa virtuelle"}
ROOT = Path(__file__).resolve().parents[2]


def _name(cn: str, org: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
                      x509.NameAttribute(NameOID.COUNTRY_NAME, "CD")])


def _write(out: Path, stem: str, key, cert: x509.Certificate) -> None:
    (out / f"{stem}.key").write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                                        serialization.PrivateFormat.PKCS8,
                                                        serialization.NoEncryption()))
    (out / f"{stem}.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _issue(subject: x509.Name, key, issuer: x509.Name, issuer_key, days: int, extensions: list) -> x509.Certificate:
    now = datetime.now(timezone.utc)
    b = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(key.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=5))
         .not_valid_after(now + timedelta(days=days)))
    for ext, critical in extensions:
        b = b.add_extension(ext, critical=critical)
    return b.sign(issuer_key, hashes.SHA256())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "infra" / "nginx" / "certs"))
    p.add_argument("--force", action="store_true", help="remplacer une PKI existante")
    a = p.parse_args()
    out = Path(a.out)
    if (out / "operators-ca.crt").exists() and not a.force:
        print(f"[OK] PKI déjà présente dans {out} (--force pour la régénérer)")
        return
    out.mkdir(parents=True, exist_ok=True)

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = _name("CA de TEST - plateforme fraude RDC", "Plateforme fraude RDC (TEST)")
    ca = _issue(ca_name, ca_key, ca_name, ca_key, 365 * 3, [
        (x509.BasicConstraints(ca=True, path_length=0), True),
        (x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                       key_encipherment=False, data_encipherment=False, key_agreement=False,
                       encipher_only=False, decipher_only=False), True)])
    _write(out, "operators-ca", ca_key, ca)

    srv_key = ec.generate_private_key(ec.SECP256R1())
    srv = _issue(_name("localhost", "Plateforme fraude RDC (TEST)"), srv_key, ca_name, ca_key, 365, [
        (x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.DNSName("nginx"),
                                      x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), False),
        (x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), False),
        (x509.BasicConstraints(ca=False, path_length=None), True)])
    _write(out, "server", srv_key, srv)

    for cn, org in OPERATORS.items():
        k = ec.generate_private_key(ec.SECP256R1())
        c = _issue(_name(cn, org), k, ca_name, ca_key, 365, [
            (x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), False),
            (x509.BasicConstraints(ca=False, path_length=None), True)])
        _write(out, cn, k, c)

    print(f"[OK] PKI de test générée dans {out} : autorité, serveur, clients {', '.join(OPERATORS)}")


if __name__ == "__main__":
    main()
