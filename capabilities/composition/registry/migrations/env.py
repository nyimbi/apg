"""
APG Capability Registry - Alembic Migration Environment

This module configures the Alembic migration environment for the APG Capability Registry.
It handles both synchronous and asynchronous database connections.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import os
from logging.config import fileConfig
from typing import Optional

from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import AsyncEngine
from alembic import context

# Import your models here to ensure they are registered with SQLAlchemy
from capability_registry.models import Base

# Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

# Other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_database_url() -> str:
    """Get database URL from environment variable or config."""
    # Try environment variable first (for production/staging)
    database_url = os.environ.get("DATABASE_URL")
    
    if database_url:
        # Handle special cases for different database drivers
        if database_url.startswith("postgres://"):
            # Convert postgres:// to postgresql:// for SQLAlchemy 1.4+
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return database_url
    
    # Fall back to config file
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is also acceptable here. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        render_as_batch=True,  # For SQLite compatibility during testing
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Run migrations with database connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        render_as_batch=True,  # For SQLite compatibility during testing
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    """Run migrations in async mode."""
    from sqlalchemy.ext.asyncio import create_async_engine
    
    # Convert sync URL to async URL
    database_url = get_database_url()
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("sqlite://"):
        database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    
    connectable = create_async_engine(
        database_url,
        poolclass=pool.NullPool,
        future=True,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection
    with the context.
    """
    # Check if we're running in async mode
    try:
        asyncio.get_running_loop()
        # We're already in an async context, run async migrations
        asyncio.create_task(run_async_migrations())
    except RuntimeError:
        # No event loop running, try async approach
        try:
            asyncio.run(run_async_migrations())
        except Exception:
            # Fall back to sync approach
            database_url = get_database_url()
            
            connectable = engine_from_config(
                config.get_section(config.config_ini_section),
                prefix="sqlalchemy.",
                poolclass=pool.NullPool,
                url=database_url,
            )

            with connectable.connect() as connection:
                do_run_migrations(connection)

            connectable.dispose()


# Determine if we're running offline or online
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()