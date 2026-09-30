"""
Crée les comptes de DÉMONSTRATION du back-office (environnement local uniquement).
Ne jamais utiliser ces mots de passe en production : créer les vrais comptes depuis
l'écran « Utilisateurs » avec un administrateur.

Usage : python scripts/seed_database.py
"""
import sys
from pathlib import Path

import bcrypt
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.database.models import BackofficeUser  # noqa: E402
from shared.database.session import make_session_factory  # noqa: E402
from shared.env_file import load_env  # noqa: E402

load_env()   # mot de passe PostgreSQL de .env

DEMO_USERS = [
    ("analyste@fraud-rdc.local", "Analyste Démo", "ANALYSTE", "Analyste-Demo-2026"),
    ("superviseur@fraud-rdc.local", "Superviseur Démo", "SUPERVISEUR", "Superviseur-Demo-2026"),
]


def main():
    _, Session = make_session_factory()
    with Session() as db:
        for email, name, role, password in DEMO_USERS:
            if db.scalar(select(BackofficeUser).where(BackofficeUser.email == email)):
                print(f"= {email} existe déjà")
                continue
            db.add(BackofficeUser(email=email, full_name=name, role=role,
                                  password_hash=bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()))
            print(f"+ {email} ({role})")
        db.commit()


if __name__ == "__main__":
    main()
