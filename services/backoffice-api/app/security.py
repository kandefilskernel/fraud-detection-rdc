"""Authentification JWT et contrôle d'accès par rôle (RBAC).

    ANALYSTE     consulte les transactions, traite les dossiers
    SUPERVISEUR  + rapports, réaffectation des dossiers, journal d'audit
    ADMIN        + gestion des utilisateurs
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from shared.database.models import BackofficeUser

ROLES = ("ANALYSTE", "SUPERVISEUR", "ADMIN")
ROLE_RANK = {r: i for i, r in enumerate(ROLES)}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user: BackofficeUser) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": str(user.id), "email": user.email, "role": user.role, "exp": exp},
                      settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> BackofficeUser:
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "authentification requise",
                                 headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise unauthorized
    user = db.get(BackofficeUser, user_id)
    if user is None or not user.is_active:
        raise unauthorized
    return user


def require_role(minimum: str):
    def checker(user: BackofficeUser = Depends(get_current_user)) -> BackofficeUser:
        if ROLE_RANK[user.role] < ROLE_RANK[minimum]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"rôle {minimum} requis")
        return user
    return checker
