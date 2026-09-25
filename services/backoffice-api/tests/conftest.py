"""Tests d'intégration : base PostgreSQL dédiée `fraud_test` (conteneur nuru_postgres),
schéma créé par les vraies migrations Alembic. Kafka désactivé."""
import os
import subprocess
import sys
from pathlib import Path

import psycopg2
import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / "services" / "backoffice-api"
sys.path[:0] = [str(ROOT), str(SERVICE)]

PG = dict(user=os.getenv("POSTGRES_USER", "nuru_admin"),
          password=os.getenv("POSTGRES_PASSWORD", "nuru_secure_password_2026"),
          host=os.getenv("TEST_POSTGRES_HOST", "localhost"), port=os.getenv("POSTGRES_PORT", "5432"))
TEST_DB = "fraud_test"
os.environ.update(
    DATABASE_URL=f"postgresql+psycopg2://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{TEST_DB}",
    KAFKA_ENABLED="false", JWT_SECRET_KEY="test-secret",
    BOOTSTRAP_ADMIN_EMAIL="admin@test.local", BOOTSTRAP_ADMIN_PASSWORD="AdminTest-2026!")


def _admin_conn():
    try:
        c = psycopg2.connect(dbname="postgres", **PG)
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL indisponible (lancer docker compose up -d postgres)")
    c.autocommit = True
    return c


@pytest.fixture(scope="session", autouse=True)
def database():
    c = _admin_conn()
    with c.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        cur.execute(f"CREATE DATABASE {TEST_DB}")
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=SERVICE, check=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True)
    yield
    with c.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    c.close()


@pytest.fixture(scope="session")
def client(database):
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


def login(client, email, password):
    r = client.post("/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def admin(client):
    return login(client, "admin@test.local", "AdminTest-2026!")


@pytest.fixture(scope="session")
def analyst(client, admin):
    client.post("/users", headers=admin, json={"email": "analyste@exemple.cd", "full_name": "Analyste Test",
                                               "role": "ANALYSTE", "password": "Analyste-2026!"})
    return login(client, "analyste@exemple.cd", "Analyste-2026!")
