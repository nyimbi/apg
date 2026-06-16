"""Alembic migration environment — APG platform root schema."""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Alembic Config object — gives access to values in alembic.ini
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
	fileConfig(config.config_file_name)

# Pull database URL from env, falling back to alembic.ini sqlalchemy.url
_db_url = os.environ.get("APG_DATABASE_URL") or config.get_main_option("sqlalchemy.url")
if not _db_url:
	raise RuntimeError(
		"APG_DATABASE_URL is not set. "
		"Export it or set sqlalchemy.url in alembic.ini before running migrations."
	)
config.set_main_option("sqlalchemy.url", _db_url)

# Import metadata from all capability models so autogenerate can detect changes.
# Add imports here as new capability models are created.
target_metadata = None
try:
	from capabilities.composition.registry.models import Base as RegistryBase
	from capabilities.composition.events.models import Base as EventsBase
	target_metadata = [RegistryBase.metadata, EventsBase.metadata]
except ImportError:
	pass  # capability packages not installed in this env — offline migration is still possible


def run_migrations_offline() -> None:
	"""Run migrations in 'offline' mode (no DB connection required, emits SQL)."""
	context.configure(
		url=_db_url,
		target_metadata=target_metadata,
		literal_binds=True,
		dialect_opts={"paramstyle": "named"},
	)
	with context.begin_transaction():
		context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
	context.configure(connection=connection, target_metadata=target_metadata)
	with context.begin_transaction():
		context.run_migrations()


async def run_async_migrations() -> None:
	engine = async_engine_from_config(
		config.get_section(config.config_ini_section, {}),
		prefix="sqlalchemy.",
		poolclass=pool.NullPool,
	)
	async with engine.connect() as connection:
		await connection.run_sync(do_run_migrations)
	await engine.dispose()


def run_migrations_online() -> None:
	asyncio.run(run_async_migrations())


if context.is_offline_mode():
	run_migrations_offline()
else:
	run_migrations_online()
