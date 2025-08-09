"""
APG Document Service Database Management

Database connection, migration, and management utilities for APG-compatible
multi-tenant document service with performance optimization.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import asyncpg
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import alembic.config
import alembic.script
from alembic.migration import MigrationContext
from alembic.operations import Operations

from .config import APGDocumentConfig
from .models import Base, Document, DocumentTemplate, Metric, MetricSummary, DocumentAccess

logger = logging.getLogger(__name__)


class DatabaseManager:
	"""
	APG-compatible database manager for document service.
	
	Provides connection pooling, migration management, multi-tenant support,
	and performance optimization following APG database patterns.
	"""
	
	def __init__(self, config: APGDocumentConfig):
		assert config, "Configuration is required"
		
		self.config = config
		self.engine = None
		self.async_engine = None
		self.session_factory = None
		self.async_session_factory = None
		self._initialized = False
		
		self._log_database_manager_initialized()
	
	def _log_database_manager_initialized(self) -> None:
		"""Log database manager initialization"""
		logger.info("APG Document Database Manager initialized")
		logger.info(f"Database URL: {self._sanitize_database_url(self.config.database_url)}")
		logger.info(f"Connection pool size: {self.config.connection_pool_size}")
		logger.info(f"Multi-tenant mode: {self.config.tenant_mode}")
	
	def _sanitize_database_url(self, url: str) -> str:
		"""Sanitize database URL for logging (remove credentials)"""
		try:
			# Simple sanitization - replace password with asterisks
			if "@" in url and "://" in url:
				parts = url.split("://", 1)
				if len(parts) == 2:
					scheme = parts[0]
					rest = parts[1]
					if "@" in rest:
						auth_part, host_part = rest.split("@", 1)
						if ":" in auth_part:
							user, password = auth_part.split(":", 1)
							return f"{scheme}://{user}:***@{host_part}"
			return url
		except Exception:
			return "***"
	
	async def initialize(self) -> None:
		"""Initialize database engines and connection pools"""
		assert not self._initialized, "Database manager already initialized"
		
		self._log_database_initialization_start()
		
		try:
			# Create synchronous engine for migrations and admin tasks
			self.engine = create_engine(
				self.config.database_url,
				poolclass=QueuePool,
				pool_size=self.config.connection_pool_size,
				max_overflow=self.config.connection_pool_overflow,
				pool_pre_ping=True,
				pool_recycle=3600,  # Recycle connections after 1 hour
				echo=False  # Set to True for SQL debugging
			)
			
			# Create async engine for application use
			async_url = self.config.database_url.replace("postgresql://", "postgresql+asyncpg://")
			self.async_engine = create_async_engine(
				async_url,
				pool_size=self.config.connection_pool_size,
				max_overflow=self.config.connection_pool_overflow,
				pool_pre_ping=True,
				pool_recycle=3600,
				echo=False
			)
			
			# Create session factories
			self.session_factory = sessionmaker(bind=self.engine)
			self.async_session_factory = async_sessionmaker(
				bind=self.async_engine,
				expire_on_commit=False
			)
			
			# Test database connectivity
			await self._test_database_connectivity()
			
			# Set up database event listeners
			self._setup_database_event_listeners()
			
			self._initialized = True
			self._log_database_initialization_complete()
			
		except Exception as e:
			logger.error(f"Database initialization failed: {e}")
			raise
	
	async def _test_database_connectivity(self) -> None:
		"""Test database connectivity"""
		try:
			async with self.async_engine.begin() as conn:
				result = await conn.execute(text("SELECT 1"))
				assert result.fetchone()[0] == 1
			logger.info("Database connectivity test successful")
		except Exception as e:
			logger.error(f"Database connectivity test failed: {e}")
			raise
	
	def _setup_database_event_listeners(self) -> None:
		"""Set up database event listeners for monitoring and optimization"""
		
		@event.listens_for(self.engine, "connect")
		def set_sqlite_pragma(dbapi_connection, connection_record):
			"""Set SQLite pragmas for better performance (if using SQLite)"""
			if "sqlite" in str(dbapi_connection):
				cursor = dbapi_connection.cursor()
				cursor.execute("PRAGMA foreign_keys=ON")
				cursor.execute("PRAGMA journal_mode=WAL")
				cursor.close()
		
		@event.listens_for(self.engine, "before_cursor_execute")
		def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
			"""Log slow queries"""
			conn.info.setdefault('query_start_time', []).append(datetime.utcnow())
		
		@event.listens_for(self.engine, "after_cursor_execute")
		def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
			"""Log completed queries and identify slow ones"""
			total = datetime.utcnow() - conn.info['query_start_time'].pop(-1)
			if total.total_seconds() > 1.0:  # Log queries slower than 1 second
				logger.warning(f"Slow query detected ({total.total_seconds():.2f}s): {statement[:100]}...")
	
	@asynccontextmanager
	async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
		"""Get async database session with proper cleanup"""
		assert self._initialized, "Database manager must be initialized first"
		
		async with self.async_session_factory() as session:
			try:
				yield session
				await session.commit()
			except Exception as e:
				await session.rollback()
				logger.error(f"Database session error: {e}")
				raise
			finally:
				await session.close()
	
	def get_session(self):
		"""Get synchronous database session (for migrations and admin tasks)"""
		assert self._initialized, "Database manager must be initialized first"
		return self.session_factory()
	
	async def create_all_tables(self) -> None:
		"""Create all database tables"""
		assert self._initialized, "Database manager must be initialized first"
		
		logger.info("Creating all database tables")
		
		async with self.async_engine.begin() as conn:
			await conn.run_sync(Base.metadata.create_all)
		
		logger.info("All database tables created successfully")
	
	async def drop_all_tables(self) -> None:
		"""Drop all database tables (use with caution!)"""
		assert self._initialized, "Database manager must be initialized first"
		
		logger.warning("Dropping all database tables")
		
		async with self.async_engine.begin() as conn:
			await conn.run_sync(Base.metadata.drop_all)
		
		logger.warning("All database tables dropped")
	
	async def run_migrations(self) -> None:
		"""Run database migrations using Alembic"""
		assert self._initialized, "Database manager must be initialized first"
		
		logger.info("Running database migrations")
		
		try:
			# Run migrations in sync context
			alembic_cfg = alembic.config.Config("alembic.ini")
			
			with self.engine.begin() as connection:
				alembic_cfg.attributes['connection'] = connection
				
				# Get current revision
				context = MigrationContext.configure(connection)
				current_rev = context.get_current_revision()
				
				# Get available revisions
				script_directory = alembic.script.ScriptDirectory.from_config(alembic_cfg)
				head_rev = script_directory.get_current_head()
				
				if current_rev == head_rev:
					logger.info("Database is already up to date")
				else:
					logger.info(f"Upgrading database from {current_rev} to {head_rev}")
					alembic.command.upgrade(alembic_cfg, "head")
					logger.info("Database migration completed successfully")
					
		except Exception as e:
			logger.error(f"Database migration failed: {e}")
			raise
	
	async def create_tenant_schema(self, tenant_id: str) -> None:
		"""Create tenant-specific schema if using schema-based multi-tenancy"""
		if self.config.tenant_mode != "multi" or self.config.tenant_isolation_level != "strict":
			return
		
		logger.info(f"Creating schema for tenant: {tenant_id}")
		
		schema_name = f"tenant_{tenant_id.replace('-', '_')}"
		
		async with self.async_engine.begin() as conn:
			# Create schema
			await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
			
			# Create tables in tenant schema
			# This would require modifying table definitions to include schema
			# For now, we'll use the default approach with tenant_id columns
		
		logger.info(f"Tenant schema created: {schema_name}")
	
	async def cleanup_expired_documents(self, days_old: int = 30) -> int:
		"""Clean up expired documents and related data"""
		assert self._initialized, "Database manager must be initialized first"
		
		logger.info(f"Cleaning up documents older than {days_old} days")
		
		cutoff_date = datetime.utcnow() - timedelta(days=days_old)
		deleted_count = 0
		
		async with self.get_async_session() as session:
			# Find expired documents
			result = await session.execute(
				text("""
					SELECT document_id FROM ds_documents 
					WHERE created_at < :cutoff_date 
					AND status = 'expired'
					LIMIT 1000
				"""),
				{"cutoff_date": cutoff_date}
			)
			
			expired_docs = result.fetchall()
			
			if expired_docs:
				doc_ids = [doc[0] for doc in expired_docs]
				
				# Delete related access logs
				await session.execute(
					text("DELETE FROM ds_document_access WHERE document_id = ANY(:doc_ids)"),
					{"doc_ids": doc_ids}
				)
				
				# Delete documents
				result = await session.execute(
					text("DELETE FROM ds_documents WHERE document_id = ANY(:doc_ids)"),
					{"doc_ids": doc_ids}
				)
				
				deleted_count = result.rowcount
				logger.info(f"Cleaned up {deleted_count} expired documents")
		
		return deleted_count
	
	async def vacuum_analyze_tables(self) -> None:
		"""Run VACUUM ANALYZE on all tables for PostgreSQL optimization"""
		assert self._initialized, "Database manager must be initialized first"
		
		if "postgresql" not in self.config.database_url:
			logger.info("VACUUM ANALYZE only supported for PostgreSQL")
			return
		
		logger.info("Running VACUUM ANALYZE on all tables")
		
		tables = [
			"ds_documents",
			"ds_document_templates", 
			"ds_metrics",
			"ds_metric_summaries",
			"ds_document_access"
		]
		
		async with self.async_engine.begin() as conn:
			for table in tables:
				await conn.execute(text(f"VACUUM ANALYZE {table}"))
		
		logger.info("VACUUM ANALYZE completed")
	
	async def get_database_stats(self) -> Dict[str, Any]:
		"""Get database statistics for monitoring"""
		assert self._initialized, "Database manager must be initialized first"
		
		stats = {
			"connection_pool": {
				"size": self.async_engine.pool.size(),
				"checked_in": self.async_engine.pool.checkedin(),
				"checked_out": self.async_engine.pool.checkedout(),
				"overflow": self.async_engine.pool.overflow(),
			},
			"table_counts": {},
			"database_size": None
		}
		
		async with self.get_async_session() as session:
			# Get table row counts
			tables = ["ds_documents", "ds_document_templates", "ds_metrics", "ds_metric_summaries", "ds_document_access"]
			
			for table in tables:
				result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
				stats["table_counts"][table] = result.scalar()
			
			# Get database size (PostgreSQL)
			if "postgresql" in self.config.database_url:
				result = await session.execute(text("SELECT pg_database_size(current_database())"))
				stats["database_size"] = result.scalar()
		
		return stats
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check database health status"""
		if not self._initialized:
			return {"healthy": False, "error": "Database manager not initialized"}
		
		try:
			async with self.get_async_session() as session:
				# Test basic query
				result = await session.execute(text("SELECT 1"))
				assert result.scalar() == 1
				
				# Check connection pool
				pool_stats = {
					"size": self.async_engine.pool.size(),
					"checked_out": self.async_engine.pool.checkedout(),
					"overflow": self.async_engine.pool.overflow()
				}
				
				return {
					"healthy": True,
					"database_url": self._sanitize_database_url(self.config.database_url),
					"pool_stats": pool_stats,
					"multi_tenant_mode": self.config.tenant_mode
				}
				
		except Exception as e:
			logger.error(f"Database health check failed: {e}")
			return {
				"healthy": False,
				"error": str(e),
				"database_url": self._sanitize_database_url(self.config.database_url)
			}
	
	def _log_database_initialization_start(self) -> None:
		"""Log database initialization start"""
		logger.info("Initializing APG document database")
	
	def _log_database_initialization_complete(self) -> None:
		"""Log database initialization completion"""
		logger.info("APG document database initialization complete")
	
	async def close(self) -> None:
		"""Close database connections and clean up resources"""
		if not self._initialized:
			return
		
		logger.info("Closing database connections")
		
		try:
			if self.async_engine:
				await self.async_engine.dispose()
			
			if self.engine:
				self.engine.dispose()
			
			self._initialized = False
			logger.info("Database connections closed successfully")
			
		except Exception as e:
			logger.error(f"Error closing database connections: {e}")


class DatabaseMigration:
	"""Database migration utilities for APG document service"""
	
	def __init__(self, db_manager: DatabaseManager):
		self.db_manager = db_manager
	
	async def create_migration_script(self, message: str) -> str:
		"""Create a new migration script"""
		logger.info(f"Creating migration script: {message}")
		
		# This would typically use Alembic's autogenerate feature
		# For now, return a placeholder
		timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
		script_name = f"{timestamp}_{message.lower().replace(' ', '_')}"
		
		return script_name
	
	async def apply_schema_changes(self, changes: List[Dict[str, Any]]) -> None:
		"""Apply schema changes to the database"""
		logger.info(f"Applying {len(changes)} schema changes")
		
		async with self.db_manager.get_async_session() as session:
			for change in changes:
				change_type = change.get("type")
				
				if change_type == "add_column":
					await self._add_column(session, change)
				elif change_type == "drop_column":
					await self._drop_column(session, change)
				elif change_type == "add_index":
					await self._add_index(session, change)
				elif change_type == "drop_index":
					await self._drop_index(session, change)
				else:
					logger.warning(f"Unknown schema change type: {change_type}")
	
	async def _add_column(self, session: AsyncSession, change: Dict[str, Any]) -> None:
		"""Add column to table"""
		table = change["table"]
		column = change["column"]
		column_type = change["type"]
		nullable = change.get("nullable", True)
		default = change.get("default")
		
		sql = f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
		if not nullable:
			sql += " NOT NULL"
		if default is not None:
			sql += f" DEFAULT {default}"
		
		await session.execute(text(sql))
		logger.info(f"Added column {column} to table {table}")
	
	async def _drop_column(self, session: AsyncSession, change: Dict[str, Any]) -> None:
		"""Drop column from table"""
		table = change["table"]
		column = change["column"]
		
		await session.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))
		logger.info(f"Dropped column {column} from table {table}")
	
	async def _add_index(self, session: AsyncSession, change: Dict[str, Any]) -> None:
		"""Add index to table"""
		index_name = change["index_name"]
		table = change["table"]
		columns = change["columns"]
		unique = change.get("unique", False)
		
		index_type = "UNIQUE INDEX" if unique else "INDEX"
		columns_str = ", ".join(columns)
		
		await session.execute(text(f"CREATE {index_type} {index_name} ON {table} ({columns_str})"))
		logger.info(f"Created index {index_name} on table {table}")
	
	async def _drop_index(self, session: AsyncSession, change: Dict[str, Any]) -> None:
		"""Drop index from table"""
		index_name = change["index_name"]
		
		await session.execute(text(f"DROP INDEX {index_name}"))
		logger.info(f"Dropped index {index_name}")


async def create_database_manager(config: APGDocumentConfig) -> DatabaseManager:
	"""Create and initialize database manager"""
	db_manager = DatabaseManager(config)
	await db_manager.initialize()
	return db_manager