"""Remplace, dans .env, les secrets par défaut ou vides par des valeurs aléatoires fortes.

    python scripts/security/generate_secrets.py            # ne touche qu'aux secrets faibles
    python scripts/security/generate_secrets.py --rotate   # régénère TOUS les secrets

Après une rotation sur une plateforme déjà lancée : le mot de passe PostgreSQL, celui du
compte admin et celui de Grafana sont conservés dans leurs volumes. Il faut les changer aussi
dans chaque service (ALTER USER dans PostgreSQL, page Utilisateurs, profil Grafana) avant de
recréer les conteneurs qui lisent .env.
Aucun secret n'est affiché.
"""
from __future__ import annotations

import argparse
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPS = ("vodacom", "airtel", "orange", "visa")
WEAK = re.compile(r"^$|a_changer|change-?me|^dev-|dev-[a-z]+-(key|hmac)|nuru_secure_password|ChangeMoi", re.I)


def _pw() -> str:
    return secrets.token_urlsafe(18).replace("-", "x").replace("_", "y") + "-A9"


def generators() -> dict:
    scoring = secrets.token_hex(32)
    return {
        "POSTGRES_PASSWORD": lambda: secrets.token_hex(24),
        "REDIS_PASSWORD": lambda: secrets.token_hex(24),
        "JWT_SECRET_KEY": lambda: secrets.token_urlsafe(48),
        "SCORING_API_KEY": lambda: scoring,
        "SCORING_API_KEYS": lambda: scoring,
        "OPERATOR_API_KEYS": lambda: ",".join(f"{o}:{secrets.token_hex(24)}" for o in OPS),
        "OPERATOR_HMAC_SECRETS": lambda: ",".join(f"{o}:{secrets.token_hex(32)}" for o in OPS),
        "INTERNAL_API_KEY": lambda: secrets.token_hex(32),
        # pseudonymisation des identifiants (import des données réelles ET temps réel : même clé)
        "PSEUDONYMIZATION_KEY": lambda: secrets.token_hex(32),
        "GRAFANA_ADMIN_PASSWORD": _pw,
        "BOOTSTRAP_ADMIN_PASSWORD": _pw,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=str(ROOT / ".env"))
    ap.add_argument("--rotate", action="store_true", help="régénérer tous les secrets")
    a = ap.parse_args()
    path = Path(a.env)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    gens, changed, seen = generators(), [], set()
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Z_]+)=(.*)$", line)
        if m and m.group(1) in gens:
            k, v = m.groups()
            seen.add(k)
            # la clé de pseudonymisation n'est jamais changée par --rotate : tous les profils et
            # historiques pseudonymisés deviendraient inutilisables (ré-import complet nécessaire)
            rotate = a.rotate and k != "PSEUDONYMIZATION_KEY"
            if rotate or any(WEAK.search(part.split(":", 1)[-1]) for part in v.split(",")):
                lines[i] = f"{k}={gens[k]()}"
                changed.append(k)
    missing = [k for k in gens if k not in seen]
    if missing:
        lines += ["", "# --- Secrets générés par scripts/security/generate_secrets.py"]
        lines += [f"{k}={gens[k]()}" for k in missing]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{path.name} : {len(changed)} secret(s) remplacé(s), {len(missing)} ajouté(s)"
          + (f" -> {', '.join(changed + missing)}" if changed or missing else " (rien à faire)"))


if __name__ == "__main__":
    main()
