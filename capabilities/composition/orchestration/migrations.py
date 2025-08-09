#!/usr/bin/env python3
"""
APG Workflow Orchestration Database Migrations

Database schema migrations and version management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from apg.framework.database import APGDatabase
from apg.framework.audit_compliance import APGAuditLogger

from .config import get_config


logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
	"""Migration execution status."""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
	"""Database migration definition."""
	version: str
	name: str
	description: str
	up_sql: str
	down_sql: str
	dependencies: List[str] = None
	is_reversible: bool = True
	checksum: Optional[str] = None
	
	def __post_init__(self):
		if self.dependencies is None:
			self.dependencies = []


class MigrationManager:
	"""Database migration management service."""
	
	def __init__(self):
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
		self.migrations: Dict[str, Migration] = {}
		self.applied_migrations: List[str] = []
	
	async def start(self):
		"""Start migration manager."""
		await self.database.start()
		await self._ensure_migration_table()
		await self._load_applied_migrations()
		await self._register_migrations()
	
	async def _ensure_migration_table(self):
		"""Ensure migration tracking table exists."""
		create_table_sql = """
		CREATE TABLE IF NOT EXISTS wo_migrations (
			version VARCHAR(50) PRIMARY KEY,
			name VARCHAR(255) NOT NULL,
			description TEXT,
			applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			checksum VARCHAR(64),
			status VARCHAR(20) DEFAULT 'completed',
			execution_time_ms INTEGER,
			rollback_sql TEXT
		);
		
		CREATE INDEX IF NOT EXISTS idx_migrations_applied_at ON wo_migrations(applied_at);
		CREATE INDEX IF NOT EXISTS idx_migrations_status ON wo_migrations(status);
		"""
		
		await self.database.execute(create_table_sql)
	
	async def _load_applied_migrations(self):
		"""Load list of applied migrations from database."""
		result = await self.database.fetch_all(
			"SELECT version FROM wo_migrations WHERE status = 'completed' ORDER BY applied_at"
		)
		self.applied_migrations = [row['version'] for row in result]
	
	async def _register_migrations(self):
		"""Register all available migrations."""
		migrations = [
			# Version 1.0.0 - Initial schema
			Migration(
				version="1.0.0",
				name="initial_schema",
				description="Create initial workflow orchestration schema",
				up_sql="""
				-- Workflows table
				CREATE TABLE wo_workflows (
					id VARCHAR(50) PRIMARY KEY,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					definition JSONB NOT NULL,
					version VARCHAR(20) DEFAULT '1.0.0',
					tenant_id VARCHAR(50) NOT NULL,
					created_by VARCHAR(50),
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					is_active BOOLEAN DEFAULT true,
					is_system BOOLEAN DEFAULT false,
					tags JSONB DEFAULT '[]'::jsonb,
					total_executions INTEGER DEFAULT 0,
					successful_executions INTEGER DEFAULT 0,
					failed_executions INTEGER DEFAULT 0,
					avg_duration_seconds DECIMAL(10,2) DEFAULT 0
				);
				
				-- Workflow instances table
				CREATE TABLE wo_workflow_instances (
					id VARCHAR(50) PRIMARY KEY,
					workflow_id VARCHAR(50) NOT NULL REFERENCES wo_workflows(id) ON DELETE CASCADE,
					status VARCHAR(20) DEFAULT 'pending',
					execution_context JSONB DEFAULT '{}'::jsonb,
					result JSONB,
					error_details TEXT,
					priority INTEGER DEFAULT 0,
					tenant_id VARCHAR(50) NOT NULL,
					created_by VARCHAR(50),
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					started_at TIMESTAMP,
					completed_at TIMESTAMP,
					duration_seconds INTEGER,
					retry_count INTEGER DEFAULT 0,
					parent_instance_id VARCHAR(50),
					correlation_id VARCHAR(50)
				);
				
				-- Task executions table
				CREATE TABLE wo_task_executions (
					id VARCHAR(50) PRIMARY KEY,
					workflow_instance_id VARCHAR(50) NOT NULL REFERENCES wo_workflow_instances(id) ON DELETE CASCADE,
					task_id VARCHAR(100) NOT NULL,
					task_name VARCHAR(255),
					status VARCHAR(20) DEFAULT 'pending',
					input_data JSONB,
					output_data JSONB,
					error_details JSONB,
					started_at TIMESTAMP,
					completed_at TIMESTAMP,
					duration_seconds INTEGER,
					retry_count INTEGER DEFAULT 0,
					checkpoint_data JSONB
				);
				
				-- Connectors table
				CREATE TABLE wo_connectors (
					id VARCHAR(50) PRIMARY KEY,
					name VARCHAR(255) NOT NULL,
					connector_type VARCHAR(50) NOT NULL,
					version VARCHAR(20) DEFAULT '1.0.0',
					endpoint VARCHAR(500),
					configuration JSONB DEFAULT '{}'::jsonb,
					auth_config JSONB DEFAULT '{}'::jsonb,
					tenant_id VARCHAR(50) NOT NULL,
					created_by VARCHAR(50),
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					is_active BOOLEAN DEFAULT true,
					is_system BOOLEAN DEFAULT false,
					last_health_check TIMESTAMP,
					health_status VARCHAR(20) DEFAULT 'unknown'
				);
				""",
				down_sql="""
				DROP TABLE IF EXISTS wo_task_executions;
				DROP TABLE IF EXISTS wo_workflow_instances;
				DROP TABLE IF EXISTS wo_connectors;
				DROP TABLE IF EXISTS wo_workflows;
				"""
			),
			
			# Version 1.1.0 - Add indexes and constraints
			Migration(
				version="1.1.0",
				name="add_indexes_constraints",
				description="Add performance indexes and data constraints",
				dependencies=["1.0.0"],
				up_sql="""
				-- Performance indexes
				CREATE INDEX idx_workflows_tenant_id ON wo_workflows(tenant_id);
				CREATE INDEX idx_workflows_status ON wo_workflows(is_active);
				CREATE INDEX idx_workflows_created_at ON wo_workflows(created_at);
				CREATE INDEX idx_workflows_name ON wo_workflows(name);
				
				CREATE INDEX idx_instances_workflow_id ON wo_workflow_instances(workflow_id);
				CREATE INDEX idx_instances_status ON wo_workflow_instances(status);
				CREATE INDEX idx_instances_tenant_id ON wo_workflow_instances(tenant_id);
				CREATE INDEX idx_instances_created_at ON wo_workflow_instances(created_at);
				CREATE INDEX idx_instances_completed_at ON wo_workflow_instances(completed_at);
				CREATE INDEX idx_instances_correlation_id ON wo_workflow_instances(correlation_id);
				
				CREATE INDEX idx_tasks_instance_id ON wo_task_executions(workflow_instance_id);
				CREATE INDEX idx_tasks_status ON wo_task_executions(status);
				CREATE INDEX idx_tasks_task_id ON wo_task_executions(task_id);
				CREATE INDEX idx_tasks_started_at ON wo_task_executions(started_at);
				
				CREATE INDEX idx_connectors_type ON wo_connectors(connector_type);
				CREATE INDEX idx_connectors_tenant_id ON wo_connectors(tenant_id);
				CREATE INDEX idx_connectors_active ON wo_connectors(is_active);
				
				-- Add constraints
				ALTER TABLE wo_workflows ADD CONSTRAINT chk_workflow_version CHECK (version ~ '^[0-9]+\.[0-9]+\.[0-9]+$');
				ALTER TABLE wo_workflow_instances ADD CONSTRAINT chk_instance_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'paused'));
				ALTER TABLE wo_task_executions ADD CONSTRAINT chk_task_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'skipped'));
				ALTER TABLE wo_connectors ADD CONSTRAINT chk_health_status CHECK (health_status IN ('healthy', 'unhealthy', 'unknown', 'error'));
				""",
				down_sql="""
				-- Remove constraints
				ALTER TABLE wo_workflows DROP CONSTRAINT IF EXISTS chk_workflow_version;
				ALTER TABLE wo_workflow_instances DROP CONSTRAINT IF EXISTS chk_instance_status;
				ALTER TABLE wo_task_executions DROP CONSTRAINT IF EXISTS chk_task_status;
				ALTER TABLE wo_connectors DROP CONSTRAINT IF EXISTS chk_health_status;
				
				-- Remove indexes
				DROP INDEX IF EXISTS idx_workflows_tenant_id;
				DROP INDEX IF EXISTS idx_workflows_status;
				DROP INDEX IF EXISTS idx_workflows_created_at;
				DROP INDEX IF EXISTS idx_workflows_name;
				DROP INDEX IF EXISTS idx_instances_workflow_id;
				DROP INDEX IF EXISTS idx_instances_status;
				DROP INDEX IF EXISTS idx_instances_tenant_id;
				DROP INDEX IF EXISTS idx_instances_created_at;
				DROP INDEX IF EXISTS idx_instances_completed_at;
				DROP INDEX IF EXISTS idx_instances_correlation_id;
				DROP INDEX IF EXISTS idx_tasks_instance_id;
				DROP INDEX IF EXISTS idx_tasks_status;
				DROP INDEX IF EXISTS idx_tasks_task_id;
				DROP INDEX IF EXISTS idx_tasks_started_at;
				DROP INDEX IF EXISTS idx_connectors_type;
				DROP INDEX IF EXISTS idx_connectors_tenant_id;
				DROP INDEX IF EXISTS idx_connectors_active;
				"""
			),
			
			# Version 1.2.0 - Add templates and scheduling
			Migration(
				version="1.2.0",
				name="add_templates_scheduling",
				description="Add workflow templates and scheduling tables",
				dependencies=["1.1.0"],
				up_sql="""
				-- Workflow templates table
				CREATE TABLE wo_workflow_templates (
					id VARCHAR(50) PRIMARY KEY,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					category VARCHAR(100),
					tags JSONB DEFAULT '[]'::jsonb,
					template_definition JSONB NOT NULL,
					configuration_schema JSONB,
					author VARCHAR(255),
					organization VARCHAR(255),
					version VARCHAR(20) DEFAULT '1.0.0',
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					is_verified BOOLEAN DEFAULT false,
					is_featured BOOLEAN DEFAULT false,
					usage_count INTEGER DEFAULT 0,
					rating DECIMAL(3,2) DEFAULT 0,
					documentation TEXT
				);
				
				-- Workflow schedules table
				CREATE TABLE wo_workflow_schedules (
					id VARCHAR(50) PRIMARY KEY,
					workflow_id VARCHAR(50) NOT NULL REFERENCES wo_workflows(id) ON DELETE CASCADE,
					name VARCHAR(255) NOT NULL,
					cron_expression VARCHAR(100),
					timezone VARCHAR(50) DEFAULT 'UTC',
					is_active BOOLEAN DEFAULT true,
					tenant_id VARCHAR(50) NOT NULL,
					created_by VARCHAR(50),
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					last_execution TIMESTAMP,
					next_execution TIMESTAMP,
					execution_count INTEGER DEFAULT 0,
					max_executions INTEGER,
					execution_context JSONB DEFAULT '{}'::jsonb
				);
				
				-- Workflow events table for audit trail
				CREATE TABLE wo_workflow_events (
					id VARCHAR(50) PRIMARY KEY,
					workflow_id VARCHAR(50),
					workflow_instance_id VARCHAR(50),
					event_type VARCHAR(50) NOT NULL,
					event_data JSONB DEFAULT '{}'::jsonb,
					tenant_id VARCHAR(50) NOT NULL,
					user_id VARCHAR(50),
					timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					correlation_id VARCHAR(50)
				);
				
				-- Add indexes
				CREATE INDEX idx_templates_category ON wo_workflow_templates(category);
				CREATE INDEX idx_templates_verified ON wo_workflow_templates(is_verified);
				CREATE INDEX idx_templates_featured ON wo_workflow_templates(is_featured);
				CREATE INDEX idx_schedules_workflow_id ON wo_workflow_schedules(workflow_id);
				CREATE INDEX idx_schedules_active ON wo_workflow_schedules(is_active);
				CREATE INDEX idx_schedules_next_execution ON wo_workflow_schedules(next_execution);
				CREATE INDEX idx_events_workflow_id ON wo_workflow_events(workflow_id);
				CREATE INDEX idx_events_instance_id ON wo_workflow_events(workflow_instance_id);
				CREATE INDEX idx_events_type ON wo_workflow_events(event_type);
				CREATE INDEX idx_events_timestamp ON wo_workflow_events(timestamp);
				""",
				down_sql="""
				DROP TABLE IF EXISTS wo_workflow_events;
				DROP TABLE IF EXISTS wo_workflow_schedules;
				DROP TABLE IF EXISTS wo_workflow_templates;
				"""
			),
			
			# Version 1.3.0 - Add monitoring and metrics
			Migration(
				version="1.3.0",
				name="add_monitoring_metrics",
				description="Add monitoring, metrics, and performance tracking",
				dependencies=["1.2.0"],
				up_sql="""
				-- Workflow metrics table
				CREATE TABLE wo_workflow_metrics (
					id VARCHAR(50) PRIMARY KEY,
					workflow_id VARCHAR(50) REFERENCES wo_workflows(id) ON DELETE CASCADE,
					metric_name VARCHAR(100) NOT NULL,
					metric_value DECIMAL(15,4),
					metric_type VARCHAR(20) DEFAULT 'gauge',
					labels JSONB DEFAULT '{}'::jsonb,
					tenant_id VARCHAR(50) NOT NULL,
					timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					aggregation_period VARCHAR(20) DEFAULT 'instant'
				);
				
				-- System health checks table
				CREATE TABLE wo_health_checks (
					id VARCHAR(50) PRIMARY KEY,
					check_name VARCHAR(100) NOT NULL,
					check_type VARCHAR(50) NOT NULL,
					status VARCHAR(20) NOT NULL,
					response_time_ms INTEGER,
					error_message TEXT,
					metadata JSONB DEFAULT '{}'::jsonb,
					tenant_id VARCHAR(50),
					timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				);
				
				-- Connector execution logs
				CREATE TABLE wo_connector_executions (
					id VARCHAR(50) PRIMARY KEY,
					connector_id VARCHAR(50) NOT NULL REFERENCES wo_connectors(id) ON DELETE CASCADE,
					workflow_instance_id VARCHAR(50) REFERENCES wo_workflow_instances(id) ON DELETE CASCADE,
					task_execution_id VARCHAR(50) REFERENCES wo_task_executions(id) ON DELETE CASCADE,
					operation VARCHAR(100) NOT NULL,
					request_data JSONB,
					response_data JSONB,
					status VARCHAR(20) NOT NULL,
					execution_time_ms INTEGER,
					error_details JSONB,
					tenant_id VARCHAR(50) NOT NULL,
					timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				);
				
				-- Add performance indexes
				CREATE INDEX idx_metrics_workflow_id ON wo_workflow_metrics(workflow_id);
				CREATE INDEX idx_metrics_name ON wo_workflow_metrics(metric_name);
				CREATE INDEX idx_metrics_timestamp ON wo_workflow_metrics(timestamp);
				CREATE INDEX idx_metrics_tenant_id ON wo_workflow_metrics(tenant_id);
				
				CREATE INDEX idx_health_checks_name ON wo_health_checks(check_name);
				CREATE INDEX idx_health_checks_status ON wo_health_checks(status);
				CREATE INDEX idx_health_checks_timestamp ON wo_health_checks(timestamp);
				
				CREATE INDEX idx_connector_executions_connector_id ON wo_connector_executions(connector_id);
				CREATE INDEX idx_connector_executions_instance_id ON wo_connector_executions(workflow_instance_id);
				CREATE INDEX idx_connector_executions_status ON wo_connector_executions(status);
				CREATE INDEX idx_connector_executions_timestamp ON wo_connector_executions(timestamp);
				
				-- Add workflow statistics columns
				ALTER TABLE wo_workflows ADD COLUMN last_executed TIMESTAMP;
				ALTER TABLE wo_workflows ADD COLUMN avg_execution_time_ms INTEGER DEFAULT 0;
				ALTER TABLE wo_workflows ADD COLUMN success_rate DECIMAL(5,4) DEFAULT 0;
				""",
				down_sql="""
				ALTER TABLE wo_workflows DROP COLUMN IF EXISTS last_executed;
				ALTER TABLE wo_workflows DROP COLUMN IF EXISTS avg_execution_time_ms;
				ALTER TABLE wo_workflows DROP COLUMN IF EXISTS success_rate;
				
				DROP TABLE IF EXISTS wo_connector_executions;
				DROP TABLE IF EXISTS wo_health_checks;
				DROP TABLE IF EXISTS wo_workflow_metrics;
				"""
			),
			
			# Version 1.4.0 - Add advanced features
			Migration(
				version="1.4.0",
				name="add_advanced_features",
				description="Add workflow versioning, approvals, and collaboration features",
				dependencies=["1.3.0"],
				up_sql="""
				-- Workflow versions table
				CREATE TABLE wo_workflow_versions (
					id VARCHAR(50) PRIMARY KEY,
					workflow_id VARCHAR(50) NOT NULL REFERENCES wo_workflows(id) ON DELETE CASCADE,
					version VARCHAR(20) NOT NULL,
					definition JSONB NOT NULL,
					change_description TEXT,
					created_by VARCHAR(50),
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					is_current BOOLEAN DEFAULT false,
					checksum VARCHAR(64)
				);
				
				-- Workflow approvals table
				CREATE TABLE wo_workflow_approvals (
					id VARCHAR(50) PRIMARY KEY,
					workflow_id VARCHAR(50) NOT NULL REFERENCES wo_workflows(id) ON DELETE CASCADE,
					approval_type VARCHAR(50) NOT NULL,
					status VARCHAR(20) DEFAULT 'pending',
					requested_by VARCHAR(50) NOT NULL,
					approved_by VARCHAR(50),
					rejection_reason TEXT,
					approval_data JSONB DEFAULT '{}'::jsonb,
					tenant_id VARCHAR(50) NOT NULL,
					requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					responded_at TIMESTAMP
				);
				
				-- Workflow collaborations table
				CREATE TABLE wo_workflow_collaborations (
					id VARCHAR(50) PRIMARY KEY,
					workflow_id VARCHAR(50) NOT NULL REFERENCES wo_workflows(id) ON DELETE CASCADE,
					user_id VARCHAR(50) NOT NULL,
					role VARCHAR(50) NOT NULL,
					permissions JSONB DEFAULT '[]'::jsonb,
					invited_by VARCHAR(50),
					invited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					accepted_at TIMESTAMP,
					tenant_id VARCHAR(50) NOT NULL
				);
				
				-- Workflow comments/annotations table
				CREATE TABLE wo_workflow_comments (
					id VARCHAR(50) PRIMARY KEY,
					workflow_id VARCHAR(50) NOT NULL REFERENCES wo_workflows(id) ON DELETE CASCADE,
					user_id VARCHAR(50) NOT NULL,
					comment_text TEXT NOT NULL,
					comment_type VARCHAR(50) DEFAULT 'general',
					parent_comment_id VARCHAR(50),
					metadata JSONB DEFAULT '{}'::jsonb,
					tenant_id VARCHAR(50) NOT NULL,
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				);
				
				-- Add indexes for new tables
				CREATE INDEX idx_versions_workflow_id ON wo_workflow_versions(workflow_id);
				CREATE INDEX idx_versions_current ON wo_workflow_versions(is_current);
				CREATE INDEX idx_approvals_workflow_id ON wo_workflow_approvals(workflow_id);
				CREATE INDEX idx_approvals_status ON wo_workflow_approvals(status);
				CREATE INDEX idx_collaborations_workflow_id ON wo_workflow_collaborations(workflow_id);
				CREATE INDEX idx_collaborations_user_id ON wo_workflow_collaborations(user_id);
				CREATE INDEX idx_comments_workflow_id ON wo_workflow_comments(workflow_id);
				CREATE INDEX idx_comments_user_id ON wo_workflow_comments(user_id);
				
				-- Add unique constraints
				CREATE UNIQUE INDEX idx_versions_workflow_version ON wo_workflow_versions(workflow_id, version);
				CREATE UNIQUE INDEX idx_collaborations_workflow_user ON wo_workflow_collaborations(workflow_id, user_id);
				""",
				down_sql="""
				DROP TABLE IF EXISTS wo_workflow_comments;
				DROP TABLE IF EXISTS wo_workflow_collaborations;
				DROP TABLE IF EXISTS wo_workflow_approvals;
				DROP TABLE IF EXISTS wo_workflow_versions;
				"""
			)
		]
		
		# Register all migrations
		for migration in migrations:
			self.migrations[migration.version] = migration
	
	async def get_pending_migrations(self) -> List[Migration]:
		"""Get list of pending migrations."""
		pending = []
		for version, migration in self.migrations.items():
			if version not in self.applied_migrations:
				# Check if dependencies are satisfied
				if all(dep in self.applied_migrations for dep in migration.dependencies):
					pending.append(migration)
		
		# Sort by version
		pending.sort(key=lambda m: m.version)
		return pending
	
	async def apply_migration(self, migration: Migration) -> bool:
		"""Apply a single migration."""
		logger.info(f"Applying migration {migration.version}: {migration.name}")
		
		start_time = datetime.utcnow()
		
		try:
			# Mark migration as running
			await self._record_migration_status(migration, MigrationStatus.RUNNING)
			
			# Execute migration SQL
			await self.database.execute(migration.up_sql)
			
			# Calculate execution time
			execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			
			# Mark migration as completed
			await self._record_migration_completion(migration, execution_time)
			
			# Add to applied migrations list
			self.applied_migrations.append(migration.version)
			
			# Audit log
			await self.audit.log_event({
				'event_type': 'migration_applied',
				'migration_version': migration.version,
				'migration_name': migration.name,
				'execution_time_ms': execution_time
			})
			
			logger.info(f"✅ Migration {migration.version} applied successfully ({execution_time:.2f}ms)")
			return True
			
		except Exception as e:
			# Mark migration as failed
			await self._record_migration_failure(migration, str(e))
			
			# Audit log
			await self.audit.log_event({
				'event_type': 'migration_failed',
				'migration_version': migration.version,
				'migration_name': migration.name,
				'error': str(e)
			})
			
			logger.error(f"❌ Migration {migration.version} failed: {e}")
			raise
	
	async def rollback_migration(self, version: str) -> bool:
		"""Rollback a specific migration."""
		if version not in self.migrations:
			raise ValueError(f"Migration {version} not found")
		
		migration = self.migrations[version]
		
		if not migration.is_reversible:
			raise ValueError(f"Migration {version} is not reversible")
		
		if version not in self.applied_migrations:
			raise ValueError(f"Migration {version} is not applied")
		
		logger.info(f"Rolling back migration {version}: {migration.name}")
		
		try:
			# Execute rollback SQL
			await self.database.execute(migration.down_sql)
			
			# Mark migration as rolled back
			await self._record_migration_rollback(migration)
			
			# Remove from applied migrations
			self.applied_migrations.remove(version)
			
			# Audit log
			await self.audit.log_event({
				'event_type': 'migration_rolled_back',
				'migration_version': version,
				'migration_name': migration.name
			})
			
			logger.info(f"✅ Migration {version} rolled back successfully")
			return True
			
		except Exception as e:
			logger.error(f"❌ Migration {version} rollback failed: {e}")
			raise
	
	async def apply_all_migrations(self) -> int:
		"""Apply all pending migrations."""
		pending = await self.get_pending_migrations()
		
		if not pending:
			logger.info("No pending migrations to apply")
			return 0
		
		logger.info(f"Applying {len(pending)} pending migrations...")
		
		applied_count = 0
		for migration in pending:
			await self.apply_migration(migration)
			applied_count += 1
		
		logger.info(f"✅ Applied {applied_count} migrations successfully")
		return applied_count
	
	async def get_migration_status(self) -> Dict[str, Any]:
		"""Get overall migration status."""
		pending = await self.get_pending_migrations()
		
		# Get migration history
		history = await self.database.fetch_all("""
			SELECT version, name, applied_at, status, execution_time_ms
			FROM wo_migrations 
			ORDER BY applied_at DESC
			LIMIT 10
		""")
		
		return {
			'total_migrations': len(self.migrations),
			'applied_migrations': len(self.applied_migrations),
			'pending_migrations': len(pending),
			'pending_versions': [m.version for m in pending],
			'recent_history': [dict(row) for row in history]
		}
	
	async def _record_migration_status(self, migration: Migration, status: MigrationStatus):
		"""Record migration status in database."""
		await self.database.execute("""
			INSERT INTO wo_migrations (version, name, description, status, checksum, rollback_sql)
			VALUES ($1, $2, $3, $4, $5, $6)
			ON CONFLICT (version) DO UPDATE SET
				status = EXCLUDED.status,
				applied_at = CURRENT_TIMESTAMP
		""", migration.version, migration.name, migration.description, 
			status.value, migration.checksum, migration.down_sql)
	
	async def _record_migration_completion(self, migration: Migration, execution_time_ms: float):
		"""Record successful migration completion."""
		await self.database.execute("""
			UPDATE wo_migrations 
			SET status = 'completed', 
				applied_at = CURRENT_TIMESTAMP,
				execution_time_ms = $2
			WHERE version = $1
		""", migration.version, int(execution_time_ms))
	
	async def _record_migration_failure(self, migration: Migration, error_message: str):
		"""Record migration failure."""
		await self.database.execute("""
			UPDATE wo_migrations 
			SET status = 'failed'
			WHERE version = $1
		""", migration.version)
	
	async def _record_migration_rollback(self, migration: Migration):
		"""Record migration rollback."""
		await self.database.execute("""
			UPDATE wo_migrations 
			SET status = 'rolled_back'
			WHERE version = $1
		""", migration.version)
	
	async def validate_database_schema(self) -> bool:
		"""Validate that database schema is up to date."""
		pending = await self.get_pending_migrations()
		return len(pending) == 0
	
	async def generate_migration_report(self) -> str:
		"""Generate a detailed migration report."""
		status = await self.get_migration_status()
		
		report = [
			"# Database Migration Report",
			f"Generated at: {datetime.utcnow().isoformat()}",
			"",
			"## Summary",
			f"- Total migrations: {status['total_migrations']}",
			f"- Applied migrations: {status['applied_migrations']}",
			f"- Pending migrations: {status['pending_migrations']}",
			""
		]
		
		if status['pending_migrations'] > 0:
			report.extend([
				"## Pending Migrations",
				""
			])
			for version in status['pending_versions']:
				migration = self.migrations[version]
				report.append(f"- {version}: {migration.name}")
			report.append("")
		
		if status['recent_history']:
			report.extend([
				"## Recent Migration History",
				""
			])
			for entry in status['recent_history']:
				report.append(f"- {entry['version']}: {entry['name']} ({entry['status']}) - {entry['applied_at']}")
		
		return "\n".join(report)


# Global migration manager
migration_manager = MigrationManager()


async def apply_migrations() -> int:
	"""Apply all pending migrations."""
	await migration_manager.start()
	return await migration_manager.apply_all_migrations()


async def get_migration_status() -> Dict[str, Any]:
	"""Get migration status."""
	await migration_manager.start()
	return await migration_manager.get_migration_status()


async def rollback_migration(version: str) -> bool:
	"""Rollback a migration."""
	await migration_manager.start()
	return await migration_manager.rollback_migration(version)