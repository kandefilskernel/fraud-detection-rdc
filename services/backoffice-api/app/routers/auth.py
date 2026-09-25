from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.events import Publisher, get_publisher
from app.security import ROLES, create_access_token, get_current_user, hash_password, require_role, verify_password
from shared.database.models import BackofficeUser

router = APIRouter(tags=["authentification & utilisateurs"])


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2)
    role: str = Field(pattern="^(ANALYSTE|SUPERVISEUR|ADMIN)$")
    password: str = Field(min_length=10)


class UserUpdate(BaseModel):
    full_name: str | None = None
    role: str | None = Field(None, pattern="^(ANALYSTE|SUPERVISEUR|ADMIN)$")
    is_active: bool | None = None
    password: str | None = Field(None, min_length=10)


@router.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db),
          pub: Publisher = Depends(get_publisher)):
    user = db.scalar(select(BackofficeUser).where(BackofficeUser.email == form.username.lower()))
    if user is None or not user.is_active or not verify_password(form.password, user.password_hash):
        pub.audit(form.username, "LOGIN_ECHEC", "bo_user", None, {})
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "identifiants invalides")
    pub.audit(user.email, "LOGIN", "bo_user", str(user.id), {})
    return {"access_token": create_access_token(user), "token_type": "bearer",
            "user": UserOut.model_validate(user).model_dump()}


@router.get("/auth/me", response_model=UserOut)
def me(user: BackofficeUser = Depends(get_current_user)):
    return user


@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db), _=Depends(require_role("SUPERVISEUR"))):
    return db.scalars(select(BackofficeUser).order_by(BackofficeUser.id)).all()


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, db: Session = Depends(get_db), admin=Depends(require_role("ADMIN")),
                pub: Publisher = Depends(get_publisher)):
    if db.scalar(select(BackofficeUser).where(BackofficeUser.email == body.email.lower())):
        raise HTTPException(409, "e-mail déjà utilisé")
    u = BackofficeUser(email=body.email.lower(), full_name=body.full_name, role=body.role,
                       password_hash=hash_password(body.password))
    db.add(u)
    db.commit()
    pub.audit(admin.email, "UTILISATEUR_CREE", "bo_user", str(u.id), {"email": u.email, "role": u.role})
    return u


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(user_id: int, body: UserUpdate, db: Session = Depends(get_db),
                admin=Depends(require_role("ADMIN")), pub: Publisher = Depends(get_publisher)):
    u = db.get(BackofficeUser, user_id)
    if u is None:
        raise HTTPException(404, "utilisateur introuvable")
    changes = body.model_dump(exclude_none=True)
    if "password" in changes:
        u.password_hash = hash_password(changes.pop("password"))
        changes["password"] = "modifié"
    for k, v in changes.items():
        if k != "password":
            setattr(u, k, v)
    db.commit()
    pub.audit(admin.email, "UTILISATEUR_MODIFIE", "bo_user", str(u.id), changes)
    return u


def ensure_bootstrap_admin(db: Session, email: str, password: str) -> None:
    if db.scalar(select(BackofficeUser.id).limit(1)) is None:
        db.add(BackofficeUser(email=email.lower(), full_name="Administrateur", role="ADMIN",
                              password_hash=hash_password(password)))
        try:
            db.commit()
        except IntegrityError:   # plusieurs workers uvicorn démarrent en même temps
            db.rollback()


__all__ = ["router", "ensure_bootstrap_admin", "ROLES"]
