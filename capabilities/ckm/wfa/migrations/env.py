"""
Alembic Migration Environment Configuration

Multi-tenant database migration support for APG Workflow & Business Process Management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import os
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

# Add the capability directory to Python path
import sys
from pathlib import Path
capability_dir = Path(__file__).parent.parent
sys.path.insert(0, str(capability_dir))

# Import the target metadata
from models import APGBaseModel

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support
target_metadata = APGBaseModel.metadata

# Configure logger
logger = logging.getLogger('alembic.env')


def get_database_url() -> str:
    """Get database URL from environment or config."""
    # Priority: Environment variable -> Config file -> Default
    database_url = (
        os.getenv('WBPM_DATABASE_URL') or
        os.getenv('DATABASE_URL') or
        config.get_main_option("sqlalchemy.url") or
        "postgresql://postgres:password@localhost:5432/apg_wbpm"
    )
    
    logger.info(f"Using database URL: {database_url.split('@')[0]}@***")
    return database_url


def get_tenant_schemas() -> list[str]:
    """Get list of tenant schemas to migrate."""
    # Get tenant list from environment or default
    tenant_list = os.getenv('WBPM_TENANT_SCHEMAS', 'tenant_default').split(',')
    schemas = ['wbpm'] + [schema.strip() for schema in tenant_list if schema.strip()]
    
    logger.info(f"Migrating schemas: {schemas}")
    return schemas


def include_name(name, type_, parent_names):
    """
    Filter for objects to include in migrations.
    
    Args:
        name: Object name
        type_: Object type (table, index, etc.)
        parent_names: Parent object names
    
    Returns:
        bool: Whether to include the object
    """
    # Include only WBPM-related objects
    if type_ == "schema":
        return name in get_tenant_schemas()
    
    if type_ == "table":
        return name.startswith('wbpm_') or name.startswith('process_') or name.startswith('task_')
    
    if type_ == "index":
        return any(table in name for table in ['wbpm_', 'process_', 'task_'])
    
    # Include other objects by default
    return True


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.
    """
    url = get_database_url()
    
    # Configure context for each schema
    for schema in get_tenant_schemas():
        logger.info(f"Running offline migration for schema: {schema}")
        
        context.configure(
            url=url,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            version_table_schema=schema,
            include_schemas=True,
            include_name=include_name,
            compare_type=True,
            compare_server_default=True,
        )
        
        with context.begin_transaction():
            context.run_migrations()


def do_run_migrations(connection: Connection, schema: str = None) -> None:
    """
    Run migrations for a specific schema.
    
    Args:
        connection: Database connection
        schema: Target schema name
    """
    # Set search path for the schema
    if schema and schema != 'public':
        connection.execute(f"SET search_path TO {schema}, public")
        logger.info(f"Set search path to: {schema}, public")
    
    # Configure migration context
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=schema,
        include_schemas=True,
        include_name=include_name,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,  # For SQLite compatibility in tests
    )
    
    # Run the migration
    with context.begin_transaction():
        logger.info(f"Running migration for schema: {schema}")
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in async mode for each tenant schema.
    """
    # Create async engine
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = get_database_url()
    
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    # Run migrations for each schema
    schemas = get_tenant_schemas()
    
    async with connectable.connect() as connection:
        # Ensure main schema exists
        await connection.execute("CREATE SCHEMA IF NOT EXISTS wbpm")
        await connection.commit()
        
        # Create tenant schemas if they don't exist
        for schema in schemas:
            if schema != 'wbpm':
                await connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        await connection.commit()
        
        # Run migrations for each schema
        for schema in schemas:
            logger.info(f"Starting migration for schema: {schema}")
            
            # Run migration in sync context
            await connection.run_sync(do_run_migrations, schema)
            
            logger.info(f"Completed migration for schema: {schema}")
    
    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode with async support.
    """
    logger.info("Running online migrations with async support")
    asyncio.run(run_async_migrations())


# Determine migration mode and run
if context.is_offline_mode():
    logger.info("Running migrations in offline mode")
    run_migrations_offline()
else:
    logger.info("Running migrations in online mode")
    run_migrations_online()