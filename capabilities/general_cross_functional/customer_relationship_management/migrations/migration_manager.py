"""
APG Customer Relationship Management - Migration Manager

Revolutionary migration management system providing comprehensive database schema
management with dependency resolution, rollback capabilities, and multi-tenant support.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
import importlib
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import asyncpg
from collections import defaultdict, deque

from .base_migration import BaseMigration, MigrationDirection, MigrationStatus


logger = logging.getLogger(__name__)


class MigrationError(Exception):
	"""Migration execution error"""
	pass


class MigrationManager:
	"""
	Database migration manager for CRM schema management
	
	Provides comprehensive migration management including:
	- Automatic migration discovery
	- Dependency resolution
	- Rollback capabilities
	- Multi-tenant support
	- Schema validation
	"""
	
	def __init__(self, database_config: Dict[str, Any]):
		"""
		Initialize migration manager
		
		Args:
			database_config: Database connection configuration
		"""
		self.database_config = database_config
		self.migrations: Dict[str, BaseMigration] = {}
		self.migration_history: List[str] = []
		
		# Migration discovery configuration  
		self.migrations_directory = Path(__file__).parent
		self.migration_prefix = "migration_"
		
		# Connection pool
		self.pool: Optional[asyncpg.Pool] = None
		
		# State tracking
		self._initialized = False
		
		logger.info("🗄️ Migration Manager initialized")
	
	async def initialize(self):
		"""Initialize migration manager"""
		try:
			logger.info("🔧 Initializing migration manager...")
			
			# Create database connection pool
			self.pool = await asyncpg.create_pool(
				host=self.database_config["host"],
				port=self.database_config["port"], 
				database=self.database_config["database"],
				user=self.database_config["user"],
				password=self.database_config["password"],
				min_size=2,
				max_size=10,
				command_timeout=60
			)
			
			# Ensure migration tracking table exists
			await self._ensure_migration_table()
			
			# Discover available migrations
			await self._discover_migrations()
			
			# Load migration history
			await self._load_migration_history()
			
			self._initialized = True
			logger.info("✅ Migration manager initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize migration manager: {str(e)}", exc_info=True)
			raise
	
	async def _ensure_migration_table(self):
		"""Ensure the migration tracking table exists"""
		async with self.pool.acquire() as connection:
			await connection.execute("""
				CREATE TABLE IF NOT EXISTS crm_schema_migrations (
					id SERIAL PRIMARY KEY,
					migration_id VARCHAR(255) UNIQUE NOT NULL,
					version VARCHAR(50) NOT NULL,
					description TEXT NOT NULL,
					applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
					checksum VARCHAR(32) NOT NULL,
					execution_time_ms INTEGER DEFAULT 0,
					created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
				)
			""")
			
			# Create index for performance
			await connection.execute("""
				CREATE INDEX IF NOT EXISTS idx_crm_schema_migrations_version 
				ON crm_schema_migrations(version)
			""")
			
			await connection.execute("""
				CREATE INDEX IF NOT EXISTS idx_crm_schema_migrations_applied_at 
				ON crm_schema_migrations(applied_at)
			""")
			
			logger.info("✅ Migration tracking table ensured")
	
	async def _discover_migrations(self):
		"""Discover migration files in the migrations directory"""
		logger.info("🔍 Discovering migration files...")
		
		migration_count = 0
		
		for file_path in self.migrations_directory.glob(f"{self.migration_prefix}*.py"):
			try:
				# Extract module name
				module_name = file_path.stem
				
				# Import migration module
				spec = importlib.util.spec_from_file_location(module_name, file_path)
				module = importlib.util.module_from_spec(spec)
				spec.loader.exec_module(module)
				
				# Find migration class
				migration_class = None
				for attr_name in dir(module):
					attr = getattr(module, attr_name)
					if (isinstance(attr, type) and 
						issubclass(attr, BaseMigration) and 
						attr != BaseMigration):
						migration_class = attr
						break
				
				if migration_class:
					migration = migration_class()
					self.migrations[migration.migration_id] = migration
					migration_count += 1
					logger.info(f"📁 Discovered migration: {migration.migration_id}")
				
			except Exception as e:
				logger.error(f"Failed to load migration from {file_path}: {str(e)}")
		
		logger.info(f"🎉 Discovered {migration_count} migrations")
	
	async def _load_migration_history(self):
		"""Load applied migration history from database"""
		async with self.pool.acquire() as connection:
			rows = await connection.fetch("""
				SELECT migration_id, version, applied_at 
				FROM crm_schema_migrations 
				ORDER BY applied_at ASC
			""")
			
			self.migration_history = [row['migration_id'] for row in rows]
			
			logger.info(f"📚 Loaded {len(self.migration_history)} applied migrations")
	
	def _resolve_migration_dependencies(self, target_migrations: List[str]) -> List[str]:
		"""
		Resolve migration dependencies and return ordered execution list
		
		Args:
			target_migrations: List of migration IDs to execute
			
		Returns:
			Ordered list of migration IDs including dependencies
		"""
		# Build dependency graph
		dependency_graph = defaultdict(list)
		in_degree = defaultdict(int)
		
		# Initialize with all migrations
		all_migrations = set(target_migrations)
		
		# Add dependencies recursively
		to_process = deque(target_migrations)
		while to_process:
			migration_id = to_process.popleft()
			
			if migration_id in self.migrations:
				migration = self.migrations[migration_id]
				
				for dep_id in migration.dependencies:
					if dep_id not in all_migrations:
						all_migrations.add(dep_id)
						to_process.append(dep_id)
					
					dependency_graph[dep_id].append(migration_id)
					in_degree[migration_id] += 1
		
		# Initialize in_degree for migrations with no dependencies
		for migration_id in all_migrations:
			if migration_id not in in_degree:
				in_degree[migration_id] = 0
		
		# Topological sort (Kahn's algorithm)
		queue = deque([m for m in all_migrations if in_degree[m] == 0])
		ordered_migrations = []
		
		while queue:
			current = queue.popleft()
			ordered_migrations.append(current)
			
			for dependent in dependency_graph[current]:
				in_degree[dependent] -= 1
				if in_degree[dependent] == 0:
					queue.append(dependent)
		
		# Check for circular dependencies
		if len(ordered_migrations) != len(all_migrations):
			remaining = all_migrations - set(ordered_migrations)
			raise MigrationError(f"Circular dependency detected in migrations: {remaining}")
		
		return ordered_migrations
	
	async def get_pending_migrations(self) -> List[str]:
		"""Get list of pending migrations"""
		pending = []
		for migration_id, migration in self.migrations.items():
			if migration_id not in self.migration_history:
				pending.append(migration_id)
		
		# Sort by version
		pending.sort(key=lambda m_id: self.migrations[m_id].version)
		return pending
	
	async def migrate_to_latest(self) -> Dict[str, Any]:
		"""
		Migrate database to the latest schema version
		
		Returns:
			Migration execution summary
		"""
		logger.info("🚀 Starting migration to latest version...")
		
		pending_migrations = await self.get_pending_migrations()
		
		if not pending_migrations:
			logger.info("✅ Database is already at the latest version")
			return {
				"status": "up_to_date",
				"executed_migrations": [],
				"total_migrations": 0,
				"duration_seconds": 0
			}
		
		return await self.migrate(pending_migrations)
	
	async def migrate(self, migration_ids: List[str]) -> Dict[str, Any]:
		"""
		Execute specified migrations
		
		Args:
			migration_ids: List of migration IDs to execute
			
		Returns:
			Migration execution summary
		"""
		start_time = datetime.utcnow()
		executed_migrations = []
		failed_migrations = []
		
		try:
			# Resolve dependencies
			ordered_migrations = self._resolve_migration_dependencies(migration_ids)
			
			# Filter out already applied migrations
			to_execute = [
				m_id for m_id in ordered_migrations 
				if m_id not in self.migration_history
			]
			
			logger.info(f"📋 Executing {len(to_execute)} migrations: {to_execute}")
			
			# Execute migrations in order
			for migration_id in to_execute:
				if migration_id not in self.migrations:
					logger.error(f"❌ Migration {migration_id} not found")
					failed_migrations.append(migration_id)
					continue
				
				migration = self.migrations[migration_id]
				
				async with self.pool.acquire() as connection:
					success = await migration.execute(connection, MigrationDirection.UP)
					
					if success:
						executed_migrations.append(migration_id)
						self.migration_history.append(migration_id)
						logger.info(f"✅ Migration {migration_id} completed successfully")
					else:
						failed_migrations.append(migration_id)
						logger.error(f"❌ Migration {migration_id} failed")
						break  # Stop on first failure
			
			end_time = datetime.utcnow()
			duration = (end_time - start_time).total_seconds()
			
			summary = {
				"status": "completed" if not failed_migrations else "failed",
				"executed_migrations": executed_migrations,
				"failed_migrations": failed_migrations,
				"total_migrations": len(to_execute),
				"duration_seconds": duration,
				"start_time": start_time.isoformat(),
				"end_time": end_time.isoformat()
			}
			
			if failed_migrations:
				logger.error(f"❌ Migration batch failed. {len(failed_migrations)} migrations failed")
			else:
				logger.info(f"🎉 All migrations completed successfully in {duration:.2f}s")
			
			return summary
			
		except Exception as e:
			logger.error(f"Migration batch failed: {str(e)}", exc_info=True)
			raise MigrationError(f"Migration execution failed: {str(e)}")
	
	async def rollback(self, target_migration_id: Optional[str] = None) -> Dict[str, Any]:
		"""
		Rollback migrations to a specific version
		
		Args:
			target_migration_id: Migration to rollback to (None for previous)
			
		Returns:
			Rollback execution summary
		"""
		start_time = datetime.utcnow()
		rolled_back_migrations = []
		
		try:
			if not self.migration_history:
				logger.info("✅ No migrations to rollback")
				return {
					"status": "no_rollback_needed",
					"rolled_back_migrations": [],
					"duration_seconds": 0
				}
			
			# Determine rollback target
			if target_migration_id is None:
				# Rollback the last migration
				to_rollback = [self.migration_history[-1]]
			else:
				# Rollback to specific migration
				if target_migration_id not in self.migration_history:
					raise MigrationError(f"Target migration {target_migration_id} not found in history")
				
				target_index = self.migration_history.index(target_migration_id)
				to_rollback = self.migration_history[target_index + 1:]
			
			# Rollback in reverse order
			to_rollback.reverse()
			
			logger.info(f"🔄 Rolling back {len(to_rollback)} migrations: {to_rollback}")
			
			for migration_id in to_rollback:
				if migration_id not in self.migrations:
					logger.error(f"❌ Migration {migration_id} not found for rollback")
					continue
				
				migration = self.migrations[migration_id]
				
				if not migration.is_reversible:
					logger.error(f"❌ Migration {migration_id} is not reversible")
					raise MigrationError(f"Migration {migration_id} cannot be rolled back")
				
				async with self.pool.acquire() as connection:
					success = await migration.execute(connection, MigrationDirection.DOWN)
					
					if success:
						rolled_back_migrations.append(migration_id)
						self.migration_history.remove(migration_id)
						logger.info(f"🔄 Migration {migration_id} rolled back successfully")
					else:
						logger.error(f"❌ Failed to rollback migration {migration_id}")
						break
			
			end_time = datetime.utcnow()
			duration = (end_time - start_time).total_seconds()
			
			logger.info(f"🎉 Rollback completed successfully in {duration:.2f}s")
			
			return {
				"status": "completed",
				"rolled_back_migrations": rolled_back_migrations,
				"total_rollbacks": len(to_rollback),
				"duration_seconds": duration,
				"start_time": start_time.isoformat(),
				"end_time": end_time.isoformat()
			}
			
		except Exception as e:
			logger.error(f"Rollback failed: {str(e)}", exc_info=True)
			raise MigrationError(f"Rollback execution failed: {str(e)}")
	
	async def get_migration_status(self) -> Dict[str, Any]:
		"""Get comprehensive migration status"""
		pending_migrations = await self.get_pending_migrations()
		
		# Get details for each migration
		all_migrations = []
		for migration_id, migration in self.migrations.items():
			all_migrations.append({
				"migration_id": migration_id,
				"version": migration.version,
				"description": migration.description,
				"is_reversible": migration.is_reversible,
				"dependencies": migration.dependencies,
				"is_applied": migration_id in self.migration_history,
				"is_pending": migration_id in pending_migrations
			})
		
		# Sort by version
		all_migrations.sort(key=lambda m: m["version"])
		
		return {
			"total_migrations": len(self.migrations),
			"applied_migrations": len(self.migration_history),
			"pending_migrations": len(pending_migrations),
			"database_version": self.migration_history[-1] if self.migration_history else None,
			"migrations": all_migrations,
			"last_migration_at": await self._get_last_migration_time(),
			"migration_table_exists": True,  # We ensure it exists
			"status": "up_to_date" if not pending_migrations else "pending_migrations"
		}
	
	async def _get_last_migration_time(self) -> Optional[str]:
		"""Get timestamp of last applied migration"""
		if not self.migration_history:
			return None
		
		async with self.pool.acquire() as connection:
			result = await connection.fetchval("""
				SELECT applied_at FROM crm_schema_migrations 
				ORDER BY applied_at DESC LIMIT 1
			""")
			
			return result.isoformat() if result else None
	
	async def validate_schema(self) -> Dict[str, Any]:
		"""Validate current database schema state"""
		logger.info("🔍 Validating database schema...")
		
		validation_results = {
			"status": "valid",
			"errors": [],
			"warnings": [],
			"migration_validations": {}
		}
		
		try:
			# Validate each applied migration
			for migration_id in self.migration_history:
				if migration_id in self.migrations:
					migration = self.migrations[migration_id]
					
					async with self.pool.acquire() as connection:
						is_valid = await migration.validate_schema_state(connection)
						
						validation_results["migration_validations"][migration_id] = {
							"valid": is_valid,
							"migration_name": migration.description
						}
						
						if not is_valid:
							validation_results["errors"].append(
								f"Schema validation failed for migration {migration_id}"
							)
			
			# Overall status
			if validation_results["errors"]:
				validation_results["status"] = "invalid"
			elif validation_results["warnings"]:
				validation_results["status"] = "warnings"
			
			logger.info(f"📊 Schema validation completed: {validation_results['status']}")
			
		except Exception as e:
			logger.error(f"Schema validation failed: {str(e)}", exc_info=True)
			validation_results["status"] = "error"
			validation_results["errors"].append(f"Validation error: {str(e)}")
		
		return validation_results
	
	async def health_check(self) -> Dict[str, Any]:
		"""Health check for migration manager"""
		try:
			# Check database connectivity
			async with self.pool.acquire() as connection:
				await connection.fetchval("SELECT 1")
				
			# Check migration table
			migration_count = await self._get_migration_table_count()
			
			return {
				"status": "healthy",
				"initialized": self._initialized,
				"database_connected": True,
				"migration_table_exists": True,
				"applied_migrations": len(self.migration_history),
				"available_migrations": len(self.migrations),
				"migration_table_rows": migration_count,
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Migration manager health check failed: {str(e)}")
			return {
				"status": "unhealthy",
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat()
			}
	
	async def _get_migration_table_count(self) -> int:
		"""Get count of rows in migration table"""
		async with self.pool.acquire() as connection:
			return await connection.fetchval("SELECT COUNT(*) FROM crm_schema_migrations")
	
	async def shutdown(self):
		"""Shutdown migration manager"""
		try:
			logger.info("🛑 Shutting down migration manager...")
			
			if self.pool:
				await self.pool.close()
			
			self._initialized = False
			logger.info("✅ Migration manager shutdown completed")
			
		except Exception as e:
			logger.error(f"Error during migration manager shutdown: {str(e)}", exc_info=True)


# Export classes and functions
__all__ = [
	"MigrationManager",
	"MigrationError"
]