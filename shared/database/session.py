from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def database_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    return ("postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
        u=os.getenv("POSTGRES_USER", "nuru_admin"), p=os.getenv("POSTGRES_PASSWORD", "nuru_secure_password_2026"),
        h=os.getenv("POSTGRES_HOST", "localhost"), port=os.getenv("POSTGRES_PORT", "5432"),
        db=os.getenv("POSTGRES_DB", "fraud_detection_db")))


def make_session_factory(url: str | None = None, **kw):
    engine = create_engine(url or database_url(), pool_pre_ping=True, pool_size=10, max_overflow=10, **kw)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)
