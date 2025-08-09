#!/usr/bin/env python3
"""
APG Accounts Receivable - Database Migration Runner
Executes database migrations with APG multi-tenant support and validation

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import asyncpg
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hashlib
import json

# Configure logging with APG-compatible format
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[
		logging.StreamHandler(sys.stdout),
		logging.FileHandler('migrations.log')
	]
)

logger = logging.getLogger('apg_ar_migrations')


class APGMigrationRunner:
	"""APG-compatible database migration runner with multi-tenant support."""
	
	def __init__(self, database_url: str, schema_name: str = 'apg_accounts_receivable'):
		assert database_url, "database_url required for APG database connection"
		assert schema_name, "schema_name required for APG multi-tenancy"
		
		self.database_url = database_url
		self.schema_name = schema_name
		self.migrations_dir = Path(__file__).parent / 'migrations'
		self.connection_pool: Optional[asyncpg.Pool] = None
		
		logger.info(f"Initialized migration runner for schema: {schema_name}")
	
	def _log_migration_step(self, step: str, details: str = None) -> str:
		"""Log migration steps with consistent formatting."""
		log_parts = [f"Migration step: {step}"]
		if details:
			log_parts.append(f"Details: {details}")
		return " | ".join(log_parts)
	
	async def _create_connection_pool(self) -> asyncpg.Pool:
		"""Create APG-compatible database connection pool."""
		try:
			pool = await asyncpg.create_pool(
				self.database_url,
				min_size=2,
				max_size=5,
				command_timeout=60,
				server_settings={
					'search_path': f'{self.schema_name},public',
					'timezone': 'UTC'
				}
			)
			logger.info("Database connection pool created successfully")
			return pool
		except Exception as e:
			logger.error(f"Failed to create connection pool: {str(e)}")
			raise
	
	async def initialize_connection(self) -> None:
		"""Initialize database connection pool."""
		if not self.connection_pool:
			self.connection_pool = await self._create_connection_pool()
	
	async def close_connection(self) -> None:
		"""Close database connection pool."""
		if self.connection_pool:
			await self.connection_pool.close()
			self.connection_pool = None
			logger.info("Database connection pool closed")
	
	def _calculate_file_checksum(self, file_path: Path) -> str:
		"""Calculate SHA-256 checksum of migration file."""
		hasher = hashlib.sha256()
		with open(file_path, 'rb') as f:
			for chunk in iter(lambda: f.read(4096), b""):
				hasher.update(chunk)
		return hasher.hexdigest()
	
	def _discover_migration_files(self) -> List[Tuple[str, Path]]:
		"""Discover and sort migration files."""
		if not self.migrations_dir.exists():
			logger.error(f"Migrations directory not found: {self.migrations_dir}")
			return []
		
		migration_files = []
		for file_path in self.migrations_dir.glob('*.sql'):
			# Extract version from filename (e.g., "001_initial_schema.sql" -> "001_initial_schema")
			version = file_path.stem
			migration_files.append((version, file_path))
		
		# Sort by version number
		migration_files.sort(key=lambda x: x[0])
		logger.info(f"Discovered {len(migration_files)} migration files")
		
		return migration_files
	
	async def _migration_applied(self, connection: asyncpg.Connection, migration_version: str) -> bool:
		"""Check if migration has already been applied."""
		result = await connection.fetchval(
			"""
			SELECT migration_applied($1, $2)
			""",
			self.schema_name,
			migration_version
		)
		return bool(result)
	
	async def _record_migration(self, connection: asyncpg.Connection, 
							   migration_version: str, migration_name: str,
							   execution_time_ms: int, checksum: str) -> int:
		"""Record successful migration application."""
		migration_id = await connection.fetchval(
			"""
			INSERT INTO apg_schema_migrations (
				schema_name, migration_version, migration_name, 
				execution_time_ms, checksum
			)
			VALUES ($1, $2, $3, $4, $5)
			RETURNING id
			""",
			self.schema_name,
			migration_version,
			migration_name,
			execution_time_ms,
			checksum
		)
		
		logger.info(f"Recorded migration {self.schema_name}.{migration_version} with ID {migration_id}")
		return migration_id
	
	async def _execute_migration_file(self, connection: asyncpg.Connection, 
									 migration_version: str, file_path: Path) -> Tuple[bool, int, str]:
		"""Execute a single migration file with error handling."""
		try:
			logger.info(f"Executing migration: {migration_version}")
			start_time = time.time()
			
			# Read migration file
			migration_sql = file_path.read_text(encoding='utf-8')
			checksum = self._calculate_file_checksum(file_path)
			
			# Execute migration in a transaction
			async with connection.transaction():
				await connection.execute(migration_sql)
			
			execution_time_ms = int((time.time() - start_time) * 1000)
			
			logger.info(self._log_migration_step(
				f"Migration {migration_version} completed",
				f"Execution time: {execution_time_ms}ms"
			))
			
			return True, execution_time_ms, checksum
			
		except Exception as e:
			logger.error(f"Migration {migration_version} failed: {str(e)}")
			return False, 0, ""
	
	async def _validate_schema_health(self, connection: asyncpg.Connection) -> bool:
		"""Validate schema health after migrations."""
		try:
			# Check if schema exists
			schema_exists = await connection.fetchval(
				"SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = $1)",
				self.schema_name
			)
			
			if not schema_exists:
				logger.error(f"Schema {self.schema_name} does not exist after migrations")
				return False
			
			# Check if core tables exist
			core_tables = [
				'ar_customers', 'ar_invoices', 'ar_payments', 
				'ar_collection_activities', 'ar_disputes'
			]
			
			for table_name in core_tables:
				table_exists = await connection.fetchval(
					"""
					SELECT EXISTS(
						SELECT 1 FROM information_schema.tables 
						WHERE table_schema = $1 AND table_name = $2
					)
					""",
					self.schema_name,
					table_name
				)
				
				if not table_exists:
					logger.error(f"Core table {table_name} missing after migrations")
					return False
			
			# Check if RLS is enabled on core tables
			for table_name in core_tables:
				rls_enabled = await connection.fetchval(
					"""
					SELECT rowsecurity FROM pg_class c
					JOIN pg_namespace n ON c.relnamespace = n.oid
					WHERE n.nspname = $1 AND c.relname = $2
					""",
					self.schema_name,
					table_name
				)
				
				if not rls_enabled:
					logger.warning(f"RLS not enabled on table {table_name}")
			
			logger.info("Schema health validation passed")
			return True
			
		except Exception as e:
			logger.error(f"Schema health validation failed: {str(e)}")
			return False
	
	async def _get_migration_status(self, connection: asyncpg.Connection) -> Dict[str, any]:
		"""Get current migration status and statistics."""
		try:
			# Get applied migrations
			applied_migrations = await connection.fetch(
				"""
				SELECT migration_version, migration_name, applied_at, execution_time_ms
				FROM apg_schema_migrations
				WHERE schema_name = $1
				ORDER BY applied_at
				""",
				self.schema_name
			)
			
			# Get schema statistics
			table_count = await connection.fetchval(
				"""
				SELECT COUNT(*) FROM information_schema.tables
				WHERE table_schema = $1 AND table_type = 'BASE TABLE'
				""",
				self.schema_name
			)
			
			index_count = await connection.fetchval(
				"""
				SELECT COUNT(*) FROM pg_indexes
				WHERE schemaname = $1
				""",
				self.schema_name
			)
			
			return {
				'schema_name': self.schema_name,
				'applied_migrations': len(applied_migrations),
				'table_count': table_count,
				'index_count': index_count,
				'last_migration': applied_migrations[-1] if applied_migrations else None,
				'migration_history': [dict(m) for m in applied_migrations]
			}
			
		except Exception as e:
			logger.error(f"Failed to get migration status: {str(e)}")
			return {}
	
	async def run_migrations(self, dry_run: bool = False) -> bool:
		"""Run all pending migrations with APG multi-tenant support."""
		await self.initialize_connection()
		
		try:
			async with self.connection_pool.acquire() as connection:
				logger.info("Starting APG Accounts Receivable migrations")
				
				# Discover migration files
				migration_files = self._discover_migration_files()
				if not migration_files:
					logger.warning("No migration files found")
					return True
				
				migrations_applied = 0
				total_execution_time = 0
				
				for migration_version, file_path in migration_files:
					# Check if already applied
					if await self._migration_applied(connection, migration_version):
						logger.info(f"Migration {migration_version} already applied, skipping")
						continue
					
					if dry_run:
						logger.info(f"DRY RUN: Would execute migration {migration_version}")
						continue
					
					# Execute migration
					success, execution_time_ms, checksum = await self._execute_migration_file(
						connection, migration_version, file_path
					)
					
					if not success:
						logger.error(f"Migration {migration_version} failed, stopping")
						return False
					
					# Record migration
					await self._record_migration(
						connection, migration_version, file_path.stem,
						execution_time_ms, checksum
					)
					
					migrations_applied += 1
					total_execution_time += execution_time_ms
				
				if dry_run:
					logger.info(f"DRY RUN completed. Would apply {migrations_applied} migrations")
					return True
				
				if migrations_applied == 0:
					logger.info("No pending migrations to apply")
				else:
					logger.info(f"Applied {migrations_applied} migrations in {total_execution_time}ms")
				
				# Validate schema health
				if not await self._validate_schema_health(connection):
					logger.error("Schema validation failed after migrations")
					return False
				
				# Log final status
				status = await self._get_migration_status(connection)
				logger.info(f"Migration completed successfully: {json.dumps(status, indent=2, default=str)}")
				
				return True
				
		except Exception as e:
			logger.error(f"Migration execution failed: {str(e)}")
			return False
		finally:
			await self.close_connection()
	
	async def rollback_migration(self, target_version: str) -> bool:
		"""Rollback to a specific migration version (placeholder)."""
		logger.warning("Migration rollback not yet implemented")
		# TODO: Implement rollback functionality
		return False
	
	async def get_status(self) -> Dict[str, any]:
		"""Get current migration status."""
		await self.initialize_connection()
		
		try:
			async with self.connection_pool.acquire() as connection:
				return await self._get_migration_status(connection)
		finally:
			await self.close_connection()


async def main():
	"""Main entry point for migration runner."""
	import argparse
	
	parser = argparse.ArgumentParser(description='APG Accounts Receivable Migration Runner')
	parser.add_argument('--database-url', 
						default=os.getenv('DATABASE_URL', 'postgresql://localhost/apg_development'),
						help='PostgreSQL database URL')
	parser.add_argument('--schema', 
						default='apg_accounts_receivable',
						help='Schema name for migrations')
	parser.add_argument('--dry-run', 
						action='store_true',
						help='Show what would be executed without making changes')
	parser.add_argument('--status', 
						action='store_true',
						help='Show current migration status')
	parser.add_argument('--rollback', 
						help='Rollback to specific migration version')
	
	args = parser.parse_args()
	
	# Initialize migration runner
	runner = APGMigrationRunner(args.database_url, args.schema)
	
	try:
		if args.status:
			status = await runner.get_status()
			print(json.dumps(status, indent=2, default=str))
			return
		
		if args.rollback:
			success = await runner.rollback_migration(args.rollback)
			sys.exit(0 if success else 1)
		
		# Run migrations
		success = await runner.run_migrations(dry_run=args.dry_run)
		sys.exit(0 if success else 1)
		
	except KeyboardInterrupt:
		logger.info("Migration interrupted by user")
		sys.exit(1)
	except Exception as e:
		logger.error(f"Migration runner failed: {str(e)}")
		sys.exit(1)


if __name__ == '__main__':
	asyncio.run(main())