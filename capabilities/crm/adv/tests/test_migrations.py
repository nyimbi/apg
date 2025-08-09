"""
APG Customer Relationship Management - Migration Tests

Comprehensive unit tests for database migration system including migration
execution, rollback functionality, dependency resolution, and validation.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
import asyncio
import asyncpg
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from ..migrations.migration_manager import MigrationManager, MigrationError
from ..migrations.base_migration import BaseMigration, MigrationDirection, MigrationStatus
from ..migrations.migration_001_initial_schema import InitialSchemaMigration
from ..migrations.migration_002_advanced_features import AdvancedFeaturesMigration
from . import TEST_DATABASE_CONFIG


@pytest.mark.unit
class TestBaseMigration:
	"""Test base migration functionality"""
	
	def test_migration_initialization(self):
		"""Test migration initialization"""
		migration = InitialSchemaMigration()
		
		assert migration.migration_id == "001_initial_schema"
		assert migration.version == "001"
		assert migration.description == "Create initial CRM database schema with core tables"
		assert migration.status == MigrationStatus.PENDING
		assert migration.is_reversible is True
		assert migration.dependencies == []
	
	def test_migration_with_dependencies(self):
		"""Test migration with dependencies"""
		migration = AdvancedFeaturesMigration()
		
		assert migration.migration_id == "002_advanced_features"
		assert migration.dependencies == ["001_initial_schema"]
	
	def test_migration_execution_summary(self):
		"""Test migration execution summary"""
		migration = InitialSchemaMigration()
		
		summary = migration.get_execution_summary()
		
		assert summary["migration_id"] == "001_initial_schema"
		assert summary["version"] == "001"
		assert summary["status"] == MigrationStatus.PENDING.value
		assert summary["is_reversible"] is True
		assert summary["dependencies"] == []
		assert summary["started_at"] is None
		assert summary["duration_seconds"] is None


@pytest.mark.unit
class TestMigrationManager:
	"""Test migration manager functionality"""
	
	@pytest.mark.asyncio
	async def test_manager_initialization(self, test_database):
		"""Test migration manager initialization"""
		manager = MigrationManager(test_database)
		
		assert not manager._initialized
		
		await manager.initialize()
		assert manager._initialized
		assert manager.pool is not None
		assert len(manager.migrations) > 0
		
		await manager.shutdown()
		assert not manager._initialized
	
	@pytest.mark.asyncio
	async def test_migration_discovery(self, test_database):
		"""Test automatic migration discovery"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Should discover our test migrations
		assert "001_initial_schema" in manager.migrations
		assert "002_advanced_features" in manager.migrations
		
		# Check migration objects
		initial_migration = manager.migrations["001_initial_schema"]
		assert isinstance(initial_migration, InitialSchemaMigration)
		
		advanced_migration = manager.migrations["002_advanced_features"]
		assert isinstance(advanced_migration, AdvancedFeaturesMigration)
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_migration_table_creation(self, test_database):
		"""Test migration tracking table creation"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Check that migration table exists
		async with manager.pool.acquire() as conn:
			table_exists = await conn.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.tables 
					WHERE table_name = 'crm_schema_migrations'
				)
			""")
			assert table_exists
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_dependency_resolution(self, test_database):
		"""Test migration dependency resolution"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Test resolving dependencies for advanced features migration
		ordered = manager._resolve_migration_dependencies(["002_advanced_features"])
		
		# Should include both migrations in correct order
		assert len(ordered) == 2
		assert ordered[0] == "001_initial_schema"  # Dependency first
		assert ordered[1] == "002_advanced_features"  # Then dependent
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_circular_dependency_detection(self, test_database):
		"""Test circular dependency detection"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Create mock migration with circular dependency
		class CircularMigration(BaseMigration):
			def _get_migration_id(self):
				return "003_circular"
			def _get_version(self):
				return "003"
			def _get_description(self):
				return "Circular dependency test"
			def _get_dependencies(self):
				return ["002_advanced_features"]
			async def up(self, connection):
				pass
		
		# Add circular dependency (advanced features depends on circular)
		manager.migrations["003_circular"] = CircularMigration()
		manager.migrations["002_advanced_features"].dependencies = ["003_circular"]
		
		# Should raise error for circular dependency
		with pytest.raises(MigrationError, match="Circular dependency"):
			manager._resolve_migration_dependencies(["002_advanced_features"])
		
		await manager.shutdown()


@pytest.mark.integration
class TestMigrationExecution:
	"""Test migration execution functionality"""
	
	@pytest.mark.asyncio
	async def test_single_migration_execution(self, test_database):
		"""Test executing a single migration"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Execute initial schema migration
		result = await manager.migrate(["001_initial_schema"])
		
		assert result["status"] == "completed"
		assert "001_initial_schema" in result["executed_migrations"]
		assert len(result["failed_migrations"]) == 0
		assert result["duration_seconds"] > 0
		
		# Verify migration was recorded
		assert "001_initial_schema" in manager.migration_history
		
		# Verify tables were created
		async with manager.pool.acquire() as conn:
			tables = ["crm_contacts", "crm_accounts", "crm_leads", 
					 "crm_opportunities", "crm_activities", "crm_campaigns"]
			
			for table in tables:
				table_exists = await conn.fetchval(f"""
					SELECT EXISTS (
						SELECT 1 FROM information_schema.tables 
						WHERE table_name = '{table}'
					)
				""")
				assert table_exists, f"Table {table} was not created"
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_migration_to_latest(self, test_database):
		"""Test migrating to latest version"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Should have no applied migrations initially
		pending = await manager.get_pending_migrations()
		assert len(pending) >= 2  # At least our two test migrations
		
		# Migrate to latest
		result = await manager.migrate_to_latest()
		
		assert result["status"] == "completed"
		assert len(result["executed_migrations"]) >= 2
		
		# Should have no pending migrations now
		pending_after = await manager.get_pending_migrations()
		assert len(pending_after) == 0
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_migration_with_dependencies(self, test_database):
		"""Test migration execution with dependencies"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Execute advanced features migration (should also execute initial schema)
		result = await manager.migrate(["002_advanced_features"])
		
		assert result["status"] == "completed"
		assert "001_initial_schema" in result["executed_migrations"]
		assert "002_advanced_features" in result["executed_migrations"]
		
		# Should be executed in correct order (dependencies first)
		initial_index = result["executed_migrations"].index("001_initial_schema")
		advanced_index = result["executed_migrations"].index("002_advanced_features")
		assert initial_index < advanced_index
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_migration_idempotency(self, test_database):
		"""Test that migrations are idempotent"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Execute migration once
		result1 = await manager.migrate(["001_initial_schema"])
		assert result1["status"] == "completed"
		
		# Execute same migration again - should be no-op
		result2 = await manager.migrate(["001_initial_schema"])
		assert result2["status"] == "completed"
		assert len(result2["executed_migrations"]) == 0  # Nothing to execute
		
		await manager.shutdown()


@pytest.mark.integration
class TestMigrationRollback:
	"""Test migration rollback functionality"""
	
	@pytest.mark.asyncio
	async def test_single_migration_rollback(self, test_database):
		"""Test rolling back a single migration"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Execute migration first
		await manager.migrate(["001_initial_schema"])
		
		# Verify tables exist
		async with manager.pool.acquire() as conn:
			table_exists = await conn.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.tables 
					WHERE table_name = 'crm_contacts'
				)
			""")
			assert table_exists
		
		# Rollback migration
		result = await manager.rollback("001_initial_schema")
		
		assert result["status"] == "completed"
		assert "001_initial_schema" in result["rolled_back_migrations"]
		
		# Verify tables were dropped
		async with manager.pool.acquire() as conn:
			table_exists = await conn.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.tables 
					WHERE table_name = 'crm_contacts'
				)
			""")
			assert not table_exists
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_rollback_last_migration(self, test_database):
		"""Test rolling back the last migration"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Execute multiple migrations
		await manager.migrate(["001_initial_schema", "002_advanced_features"])
		
		original_history_length = len(manager.migration_history)
		
		# Rollback last migration (no target specified)
		result = await manager.rollback()
		
		assert result["status"] == "completed"
		assert len(result["rolled_back_migrations"]) == 1
		assert len(manager.migration_history) == original_history_length - 1
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_rollback_with_dependencies(self, test_database):
		"""Test rollback with dependent migrations"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Execute both migrations
		await manager.migrate(["002_advanced_features"])
		
		# Rollback to initial schema (should rollback advanced features)
		result = await manager.rollback("001_initial_schema")
		
		assert result["status"] == "completed"
		assert "002_advanced_features" in result["rolled_back_migrations"]
		
		# Only initial schema should remain
		assert len(manager.migration_history) == 1
		assert manager.migration_history[0] == "001_initial_schema"
		
		await manager.shutdown()


@pytest.mark.unit
class TestMigrationValidation:
	"""Test migration validation functionality"""
	
	@pytest.mark.asyncio
	async def test_schema_validation(self, test_database):
		"""Test database schema validation"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Execute migrations
		await manager.migrate_to_latest()
		
		# Validate schema
		validation = await manager.validate_schema()
		
		assert validation["status"] == "valid"
		assert len(validation["errors"]) == 0
		assert len(validation["migration_validations"]) > 0
		
		# All migrations should be valid
		for migration_id, result in validation["migration_validations"].items():
			assert result["valid"] is True
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_invalid_schema_detection(self, test_database):
		"""Test detection of invalid schema state"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Execute initial migration
		await manager.migrate(["001_initial_schema"])
		
		# Manually break schema (drop a table)
		async with manager.pool.acquire() as conn:
			await conn.execute("DROP TABLE crm_contacts")
		
		# Validation should detect the problem
		validation = await manager.validate_schema()
		
		assert validation["status"] == "invalid"
		assert len(validation["errors"]) > 0
		
		await manager.shutdown()


@pytest.mark.unit
class TestMigrationStatus:
	"""Test migration status reporting"""
	
	@pytest.mark.asyncio
	async def test_migration_status_reporting(self, test_database):
		"""Test comprehensive migration status reporting"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Get initial status
		status = await manager.get_migration_status()
		
		assert status["total_migrations"] >= 2
		assert status["applied_migrations"] == 0
		assert status["pending_migrations"] >= 2
		assert status["database_version"] is None
		assert status["status"] == "pending_migrations"
		
		# Execute some migrations
		await manager.migrate(["001_initial_schema"])
		
		# Get updated status
		updated_status = await manager.get_migration_status()
		
		assert updated_status["applied_migrations"] == 1
		assert updated_status["pending_migrations"] >= 1
		assert updated_status["database_version"] == "001_initial_schema"
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_health_check(self, test_database):
		"""Test migration manager health check"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		health = await manager.health_check()
		
		assert health["status"] == "healthy"
		assert health["initialized"] is True
		assert health["database_connected"] is True
		assert health["migration_table_exists"] is True
		assert health["available_migrations"] >= 2
		assert "timestamp" in health
		
		await manager.shutdown()


@pytest.mark.unit
class TestMigrationErrorHandling:
	"""Test migration error handling"""
	
	@pytest.mark.asyncio
	async def test_invalid_database_config(self):
		"""Test handling of invalid database configuration"""
		invalid_config = {
			"host": "nonexistent-host",
			"port": 9999,
			"database": "nonexistent-db",
			"user": "invalid-user",
			"password": "invalid-password"
		}
		
		manager = MigrationManager(invalid_config)
		
		with pytest.raises(Exception):
			await manager.initialize()
	
	@pytest.mark.asyncio
	async def test_migration_execution_failure(self, test_database):
		"""Test handling of migration execution failures"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Create a migration that will fail
		class FailingMigration(BaseMigration):
			def _get_migration_id(self):
				return "999_failing"
			def _get_version(self):
				return "999"
			def _get_description(self):
				return "A migration that fails"
			async def up(self, connection):
				raise Exception("Intentional failure")
		
		# Add failing migration
		failing_migration = FailingMigration()
		manager.migrations["999_failing"] = failing_migration
		
		# Execute should handle the failure
		result = await manager.migrate(["999_failing"])
		
		assert result["status"] == "failed"
		assert "999_failing" in result["failed_migrations"]
		assert len(result["executed_migrations"]) == 0
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_rollback_of_non_reversible_migration(self, test_database):
		"""Test rollback failure for non-reversible migrations"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Create non-reversible migration
		class NonReversibleMigration(BaseMigration):
			def _get_migration_id(self):
				return "998_non_reversible"
			def _get_version(self):
				return "998"
			def _get_description(self):
				return "Non-reversible migration"
			def _is_reversible(self):
				return False
			async def up(self, connection):
				pass  # Do nothing for test
		
		# Add and execute migration
		non_reversible = NonReversibleMigration()
		manager.migrations["998_non_reversible"] = non_reversible
		await manager.migrate(["998_non_reversible"])
		
		# Rollback should fail
		with pytest.raises(MigrationError):
			await manager.rollback()
		
		await manager.shutdown()


@pytest.mark.performance
class TestMigrationPerformance:
	"""Test migration performance characteristics"""
	
	@pytest.mark.asyncio
	async def test_migration_execution_performance(self, test_database, performance_timer):
		"""Test migration execution performance"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		performance_timer.start()
		
		# Execute all migrations
		result = await manager.migrate_to_latest()
		
		performance_timer.stop()
		
		assert result["status"] == "completed"
		
		# Migrations should complete within reasonable time
		assert performance_timer.elapsed < 30.0  # 30 seconds max
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_large_dataset_migration_performance(self, test_database):
		"""Test migration performance with large dataset"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# Execute initial schema
		await manager.migrate(["001_initial_schema"])
		
		# Insert large dataset
		async with manager.pool.acquire() as conn:
			# Insert 1000 test contacts
			for i in range(1000):
				await conn.execute("""
					INSERT INTO crm_contacts 
					(id, tenant_id, first_name, last_name, email, created_by, updated_by)
					VALUES ($1, $2, $3, $4, $5, $6, $7)
				""", f"contact_{i:04d}", "test_tenant", f"First{i:04d}", f"Last{i:04d}", 
					f"contact{i:04d}@example.com", "system", "system")
		
		# Execute advanced features migration on large dataset
		start_time = datetime.utcnow()
		result = await manager.migrate(["002_advanced_features"])
		end_time = datetime.utcnow()
		
		duration = (end_time - start_time).total_seconds()
		
		assert result["status"] == "completed"
		assert duration < 60.0  # Should complete within 1 minute even with large dataset
		
		await manager.shutdown()


@pytest.mark.integration
class TestMigrationIntegration:
	"""Integration tests for migration system"""
	
	@pytest.mark.asyncio
	async def test_complete_migration_lifecycle(self, test_database):
		"""Test complete migration lifecycle"""
		manager = MigrationManager(test_database)
		await manager.initialize()
		
		# 1. Check initial state
		initial_status = await manager.get_migration_status()
		assert initial_status["applied_migrations"] == 0
		
		# 2. Execute migrations
		migrate_result = await manager.migrate_to_latest()
		assert migrate_result["status"] == "completed"
		
		# 3. Validate schema
		validation = await manager.validate_schema()
		assert validation["status"] == "valid"
		
		# 4. Check final state
		final_status = await manager.get_migration_status()
		assert final_status["status"] == "up_to_date"
		assert final_status["applied_migrations"] > 0
		assert final_status["pending_migrations"] == 0
		
		# 5. Test rollback
		rollback_result = await manager.rollback()
		assert rollback_result["status"] == "completed"
		
		# 6. Verify rollback
		post_rollback_status = await manager.get_migration_status()
		assert post_rollback_status["applied_migrations"] < final_status["applied_migrations"]
		
		await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_concurrent_migration_safety(self, test_database):
		"""Test migration safety with concurrent access"""
		# Create two managers for the same database
		manager1 = MigrationManager(test_database)
		manager2 = MigrationManager(test_database)
		
		await manager1.initialize()
		await manager2.initialize()
		
		try:
			# Try to run migrations concurrently (should be safe due to transactions)
			task1 = manager1.migrate(["001_initial_schema"])
			task2 = manager2.migrate(["001_initial_schema"])
			
			results = await asyncio.gather(task1, task2, return_exceptions=True)
			
			# At least one should succeed
			successful_results = [r for r in results if not isinstance(r, Exception)]
			assert len(successful_results) >= 1
			
			# Both managers should see the migration as applied
			status1 = await manager1.get_migration_status()
			status2 = await manager2.get_migration_status()
			
			assert "001_initial_schema" in [m["migration_id"] for m in status1["migrations"] if m["is_applied"]]
			assert "001_initial_schema" in [m["migration_id"] for m in status2["migrations"] if m["is_applied"]]
			
		finally:
			await manager1.shutdown()
			await manager2.shutdown()