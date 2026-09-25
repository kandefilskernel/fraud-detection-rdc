"""Migrations Alembic : la cible est le modèle partagé shared/database/models.py."""
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# racine du dépôt (dev) ou /srv (conteneur) : rend `shared` importable
for p in Path(__file__).resolve().parents:
    if (p / "shared" / "database").is_dir():
        sys.path.insert(0, str(p))
        break

from shared.database.models import Base  # noqa: E402
from shared.database.session import database_url  # noqa: E402

config = context.config
config.set_main_option("sqlalchemy.url", database_url())
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):
    # index créés automatiquement par TimescaleDB (create_hypertable) : hors du modèle
    return not (type_ == "index" and reflected and name and name.endswith("_tx_time_idx"))


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata,
                      literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(config.get_section(config.config_ini_section, {}),
                                     prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          include_object=include_object)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
