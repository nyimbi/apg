"""
APG Customer Relationship Management - Base Migration

Base migration class providing the foundation for all database migrations
with comprehensive validation, rollback support, and tenant-aware operations.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import asyncpg


logger = logging.getLogger(__name__)


class MigrationDirection(str, Enum):
	"""Migration direction"""
	UP = "up"
	DOWN = "down"


class MigrationStatus(str, Enum):
	"""Migration status"""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	ROLLED_BACK = "rolled_back"


class BaseMigration(ABC):
	"""
	Base class for all database migrations
	
	Provides the framework for creating database schema changes with
	comprehensive validation, rollback support, and audit tracking.
	"""
	
	def __init__(self):
		"""Initialize base migration"""
		self.migration_id: str = self._get_migration_id()
		self.version: str = self._get_version()
		self.description: str = self._get_description()
		self.dependencies: List[str] = self._get_dependencies()
		self.is_reversible: bool = self._is_reversible()
		
		# Execution state
		self.status = MigrationStatus.PENDING
		self.started_at: Optional[datetime] = None
		self.completed_at: Optional[datetime] = None
		self.error_message: Optional[str] = None
		
		logger.info(f"🔧 Migration initialized: {self.migration_id} - {self.description}")
	
	@abstractmethod
	def _get_migration_id(self) -> str:
		"""Get unique migration identifier"""
		pass
	
	@abstractmethod 
	def _get_version(self) -> str:
		"""Get migration version (e.g., '001', '002')"""
		pass
	
	@abstractmethod
	def _get_description(self) -> str:
		"""Get migration description"""
		pass
	
	def _get_dependencies(self) -> List[str]:
		"""Get list of migration IDs this migration depends on"""
		return []
	
	def _is_reversible(self) -> bool:
		"""Whether this migration can be rolled back"""
		return True
	
	@abstractmethod
	async def up(self, connection: asyncpg.Connection) -> None:
		"""
		Apply the migration (forward direction)
		
		Args:
			connection: Database connection
		"""
		pass
	
	async def down(self, connection: asyncpg.Connection) -> None:
		"""
		Rollback the migration (reverse direction)
		
		Args:
			connection: Database connection
		"""
		if not self.is_reversible:
			raise NotImplementedError(f"Migration {self.migration_id} is not reversible")
		
		raise NotImplementedError(f"Rollback not implemented for {self.migration_id}")
	
	async def execute(
		self, 
		connection: asyncpg.Connection, 
		direction: MigrationDirection = MigrationDirection.UP
	) -> bool:
		"""
		Execute the migration in the specified direction
		
		Args:
			connection: Database connection
			direction: Migration direction (up/down)
			
		Returns:
			True if successful, False otherwise
		"""
		try:
			self.status = MigrationStatus.RUNNING
			self.started_at = datetime.utcnow()
			
			logger.info(f"🚀 Executing migration {self.migration_id} ({direction.value})")
			
			# Start transaction
			async with connection.transaction():
				# Validate preconditions
				await self._validate_preconditions(connection, direction)
				
				# Execute migration
				if direction == MigrationDirection.UP:
					await self.up(connection)
				else:
					await self.down(connection)
				
				# Validate postconditions
				await self._validate_postconditions(connection, direction)
				
				# Record migration in schema history
				await self._record_migration(connection, direction)
			
			self.status = MigrationStatus.COMPLETED
			self.completed_at = datetime.utcnow()
			
			duration = (self.completed_at - self.started_at).total_seconds()
			logger.info(f"✅ Migration {self.migration_id} completed in {duration:.2f}s")
			
			return True
			
		except Exception as e:
			self.status = MigrationStatus.FAILED
			self.error_message = str(e)
			self.completed_at = datetime.utcnow()
			
			logger.error(f"❌ Migration {self.migration_id} failed: {str(e)}", exc_info=True)
			return False
	
	async def _validate_preconditions(
		self, 
		connection: asyncpg.Connection, 
		direction: MigrationDirection
	) -> None:
		"""
		Validate preconditions before executing migration
		
		Args:
			connection: Database connection
			direction: Migration direction
		"""
		# Override in subclasses for custom validation
		pass
	
	async def _validate_postconditions(
		self, 
		connection: asyncpg.Connection, 
		direction: MigrationDirection
	) -> None:
		"""
		Validate postconditions after executing migration
		
		Args:
			connection: Database connection  
			direction: Migration direction
		"""
		# Override in subclasses for custom validation
		pass
	
	async def _record_migration(
		self, 
		connection: asyncpg.Connection, 
		direction: MigrationDirection
	) -> None:
		"""
		Record migration execution in schema history
		
		Args:
			connection: Database connection
			direction: Migration direction
		"""
		try:
			if direction == MigrationDirection.UP:
				# Record migration as applied
				await connection.execute("""
					INSERT INTO crm_schema_migrations 
					(migration_id, version, description, applied_at, checksum)
					VALUES ($1, $2, $3, $4, $5)
					ON CONFLICT (migration_id) DO UPDATE SET
						applied_at = EXCLUDED.applied_at,
						checksum = EXCLUDED.checksum
				""", 
				self.migration_id, 
				self.version, 
				self.description, 
				datetime.utcnow(),
				self._calculate_checksum()
				)
			else:
				# Remove migration from history (rollback)
				await connection.execute("""
					DELETE FROM crm_schema_migrations 
					WHERE migration_id = $1
				""", self.migration_id)
			
		except Exception as e:
			logger.error(f"Failed to record migration {self.migration_id}: {str(e)}")
			raise
	
	def _calculate_checksum(self) -> str:
		"""Calculate checksum for migration integrity verification"""
		import hashlib
		
		# Create checksum based on migration code
		content = f"{self.migration_id}{self.version}{self.description}"
		return hashlib.md5(content.encode()).hexdigest()
	
	async def validate_schema_state(self, connection: asyncpg.Connection) -> bool:
		"""
		Validate that the database schema is in the expected state
		
		Args:
			connection: Database connection
			
		Returns:
			True if schema state is valid
		"""
		try:
			# Override in subclasses for custom validation
			return True
			
		except Exception as e:
			logger.error(f"Schema validation failed for {self.migration_id}: {str(e)}")
			return False
	
	def get_execution_summary(self) -> Dict[str, Any]:
		"""Get summary of migration execution"""
		duration = None
		if self.started_at and self.completed_at:
			duration = (self.completed_at - self.started_at).total_seconds()
		
		return {
			"migration_id": self.migration_id,
			"version": self.version,
			"description": self.description,
			"status": self.status.value,
			"is_reversible": self.is_reversible,
			"dependencies": self.dependencies,
			"started_at": self.started_at.isoformat() if self.started_at else None,
			"completed_at": self.completed_at.isoformat() if self.completed_at else None,
			"duration_seconds": duration,
			"error_message": self.error_message
		}


# Helper functions for common migration operations

async def table_exists(connection: asyncpg.Connection, table_name: str) -> bool:
	"""Check if a table exists"""
	result = await connection.fetchval("""
		SELECT EXISTS (
			SELECT 1 FROM information_schema.tables 
			WHERE table_name = $1 AND table_schema = 'public'
		)
	""", table_name)
	return bool(result)


async def column_exists(connection: asyncpg.Connection, table_name: str, column_name: str) -> bool:
	"""Check if a column exists in a table"""
	result = await connection.fetchval("""
		SELECT EXISTS (
			SELECT 1 FROM information_schema.columns 
			WHERE table_name = $1 AND column_name = $2 AND table_schema = 'public'
		)
	""", table_name, column_name)
	return bool(result)


async def index_exists(connection: asyncpg.Connection, index_name: str) -> bool:
	"""Check if an index exists"""
	result = await connection.fetchval("""
		SELECT EXISTS (
			SELECT 1 FROM pg_indexes 
			WHERE indexname = $1 AND schemaname = 'public'
		)
	""", index_name)
	return bool(result)


async def constraint_exists(connection: asyncpg.Connection, constraint_name: str) -> bool:
	"""Check if a constraint exists"""
	result = await connection.fetchval("""
		SELECT EXISTS (
			SELECT 1 FROM information_schema.table_constraints 
			WHERE constraint_name = $1 AND table_schema = 'public'
		)
	""", constraint_name)
	return bool(result)


async def get_table_row_count(connection: asyncpg.Connection, table_name: str) -> int:
	"""Get approximate row count for a table"""
	try:
		result = await connection.fetchval(f"SELECT COUNT(*) FROM {table_name}")
		return int(result)
	except Exception:
		return 0


# Export classes and functions
__all__ = [
	"BaseMigration",
	"MigrationDirection", 
	"MigrationStatus",
	"table_exists",
	"column_exists",
	"index_exists",
	"constraint_exists",
	"get_table_row_count"
]