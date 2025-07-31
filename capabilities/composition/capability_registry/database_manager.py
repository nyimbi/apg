"""
Database Manager

Manages database schema composition and migration for composed APG applications.
Handles table creation, relationship mapping, migration planning, and schema versioning.
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from enum import Enum
import hashlib

from pydantic import BaseModel, Field, ConfigDict, validator
from sqlalchemy import (
	create_engine, MetaData, Table, Column, String, Integer, 
	Boolean, DateTime, Text, JSON, ForeignKey, Index, inspect,
	text
)
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.engine import Engine
from flask_appbuilder import Model
from uuid_extensions import uuid7str

from .registry import CapabilityRegistry, CapabilityMetadata, SubCapabilityMetadata, get_registry

logger = logging.getLogger(__name__)

class MigrationType(Enum):
	"""Types of database migrations."""
	CREATE_TABLE = "create_table"
	ALTER_TABLE = "alter_table"
	DROP_TABLE = "drop_table"
	ADD_COLUMN = "add_column"
	DROP_COLUMN = "drop_column"
	ADD_INDEX = "add_index"
	DROP_INDEX = "drop_index"
	ADD_CONSTRAINT = "add_constraint"
	DROP_CONSTRAINT = "drop_constraint"
	CREATE_VIEW = "create_view"
	DROP_VIEW = "drop_view"

class SchemaVersioningStrategy(Enum):
	"""Schema versioning strategies."""
	CAPABILITY_BASED = "capability_based"  # Version per capability
	GLOBAL = "global"  # Single global version
	TIMESTAMP = "timestamp"  # Timestamp-based versioning
	SEMANTIC = "semantic"  # Semantic versioning

class TableMetadata(BaseModel):
	"""Metadata for a database table."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	# Table identity
	table_name: str = Field(..., description="Database table name")
	model_class: str = Field(..., description="SQLAlchemy model class name")
	capability: str = Field(..., description="Owning capability")
	subcapability: Optional[str] = Field(default=None, description="Owning sub-capability")
	
	# Table structure
	columns: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Column definitions")
	indexes: list[dict[str, Any]] = Field(default_factory=list, description="Index definitions")
	constraints: list[dict[str, Any]] = Field(default_factory=list, description="Constraint definitions")
	
	# Relationships
	foreign_keys: list[dict[str, Any]] = Field(default_factory=list, description="Foreign key relationships")
	relationships: list[dict[str, Any]] = Field(default_factory=list, description="SQLAlchemy relationships")
	
	# Metadata
	module_path: str = Field(..., description="Python module path")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	version: str = Field(default="1.0.0", description="Table schema version")
	
	# Options
	is_core_table: bool = Field(default=False, description="Whether this is a core system table")
	supports_multi_tenancy: bool = Field(default=True, description="Whether table supports multi-tenancy")
	audit_enabled: bool = Field(default=True, description="Whether audit logging is enabled")

class MigrationStep(BaseModel):
	"""A single migration step."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str, description="Step ID")
	migration_type: MigrationType = Field(..., description="Type of migration")
	table_name: str = Field(..., description="Target table name")
	
	# Migration details
	sql_statement: str = Field(..., description="SQL statement to execute")
	rollback_statement: Optional[str] = Field(default=None, description="Rollback SQL statement")
	
	# Dependencies
	depends_on: list[str] = Field(default_factory=list, description="Step dependencies")
	capability: str = Field(..., description="Capability that owns this migration")
	
	# Metadata
	description: str = Field(default="", description="Human-readable description")
	estimated_duration_ms: int = Field(default=0, description="Estimated execution time")
	reversible: bool = Field(default=True, description="Whether migration is reversible")
	
	# Validation
	pre_conditions: list[str] = Field(default_factory=list, description="Pre-conditions to check")
	post_conditions: list[str] = Field(default_factory=list, description="Post-conditions to verify")

class MigrationPlan(BaseModel):
	"""Complete migration plan for schema composition."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str, description="Migration plan ID")
	name: str = Field(..., description="Migration plan name")
	description: str = Field(default="", description="Plan description")
	
	# Migration steps
	steps: list[MigrationStep] = Field(default_factory=list, description="Migration steps in order")
	
	# Target schema
	target_capabilities: list[str] = Field(..., description="Target capabilities")
	target_tables: list[str] = Field(default_factory=list, description="Target tables")
	
	# Versioning
	from_version: Optional[str] = Field(default=None, description="Source schema version")
	to_version: str = Field(..., description="Target schema version")
	versioning_strategy: SchemaVersioningStrategy = Field(default=SchemaVersioningStrategy.CAPABILITY_BASED)
	
	# Execution metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Plan creation time")
	estimated_duration_ms: int = Field(default=0, description="Estimated total execution time")
	
	# Safety and validation
	requires_downtime: bool = Field(default=False, description="Whether downtime is required")
	backup_required: bool = Field(default=True, description="Whether backup is required")
	validation_queries: list[str] = Field(default_factory=list, description="Validation queries")

class SchemaComposition(BaseModel):
	"""Complete schema composition for an APG application."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str, description="Schema composition ID")
	tenant_id: str = Field(..., description="Tenant ID")
	application_name: str = Field(..., description="Application name")
	
	# Composition details
	capabilities: list[str] = Field(..., description="Included capabilities")
	tables: dict[str, TableMetadata] = Field(default_factory=dict, description="Table metadata")
	
	# Schema information
	total_tables: int = Field(default=0, description="Total number of tables")
	total_columns: int = Field(default=0, description="Total number of columns")
	total_indexes: int = Field(default=0, description="Total number of indexes")
	total_constraints: int = Field(default=0, description="Total number of constraints")
	
	# Versioning and tracking
	schema_version: str = Field(..., description="Overall schema version")
	capability_versions: dict[str, str] = Field(default_factory=dict, description="Per-capability versions")
	schema_hash: str = Field(default="", description="Schema structure hash")
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	# Migration history
	applied_migrations: list[str] = Field(default_factory=list, description="Applied migration IDs")
	migration_history: list[dict[str, Any]] = Field(default_factory=list, description="Migration history")

class DatabaseManager:
	"""
	Database schema composition and migration manager.
	
	Handles creation and management of database schemas for composed APG applications,
	including table discovery, relationship mapping, migration planning, and execution.
	"""
	
	def __init__(self, 
				 database_url: Optional[str] = None,
				 registry: Optional[CapabilityRegistry] = None):
		"""Initialize the database manager."""
		self.database_url = database_url
		self.registry = registry or get_registry()
		self.engine: Optional[Engine] = None
		self.metadata = MetaData()
		
		if database_url:
			self.engine = create_engine(database_url)
		
		logger.info("DatabaseManager initialized")
	
	def discover_schema(self, capabilities: List[str]) -> SchemaComposition:
		"""
		Discover database schema for the given capabilities.
		
		Args:
			capabilities: List of capability codes
			
		Returns:
			SchemaComposition with discovered tables and metadata
		"""
		# Ensure registry is populated
		if not self.registry.capabilities:
			self.registry.discover_all()
		
		schema_composition = SchemaComposition(
			tenant_id="default",  # Will be overridden
			application_name="APG_Application",
			capabilities=capabilities,
			schema_version="1.0.0"
		)
		
		logger.info(f"Discovering schema for capabilities: {capabilities}")
		
		for cap_code in capabilities:
			capability = self.registry.get_capability(cap_code)
			if not capability:
				logger.warning(f"Capability {cap_code} not found during schema discovery")
				continue
			
			# Discover tables from sub-capabilities
			for subcap in capability.subcapabilities.values():
				if subcap.has_models:
					tables = self._discover_tables_from_subcapability(capability, subcap)
					for table_name, table_meta in tables.items():
						schema_composition.tables[table_name] = table_meta
		
		# Calculate totals
		self._calculate_schema_statistics(schema_composition)
		
		# Generate schema hash
		schema_composition.schema_hash = self._generate_schema_hash(schema_composition)
		
		logger.info(f"Discovered {schema_composition.total_tables} tables from {len(capabilities)} capabilities")
		return schema_composition
	
	def _discover_tables_from_subcapability(self, 
											capability: CapabilityMetadata,
											subcapability: SubCapabilityMetadata) -> Dict[str, TableMetadata]:
		"""Discover tables from a sub-capability's models."""
		tables = {}
		
		try:
			# Import the models module
			models_module_path = f"{subcapability.module_path}.models"
			models_module = importlib.import_module(models_module_path)
			
			# Find all SQLAlchemy model classes
			for name, obj in vars(models_module).items():
				if (isinstance(obj, type) and 
					hasattr(obj, '__tablename__') and
					issubclass(obj, Model)):
					
					table_meta = self._extract_table_metadata(obj, capability, subcapability)
					tables[table_meta.table_name] = table_meta
		
		except ImportError as e:
			logger.debug(f"Could not import models for {subcapability.code}: {e}")
		except Exception as e:
			logger.error(f"Error discovering tables from {subcapability.code}: {e}")
		
		return tables
	
	def _extract_table_metadata(self, 
								model_class: type,
								capability: CapabilityMetadata,
								subcapability: SubCapabilityMetadata) -> TableMetadata:
		"""Extract metadata from a SQLAlchemy model class."""
		table_name = model_class.__tablename__
		
		# Extract column information
		columns = {}
		indexes = []
		constraints = []
		foreign_keys = []
		relationships = []
		
		if hasattr(model_class, '__table__'):
			table = model_class.__table__
			
			# Extract columns
			for column in table.columns:
				columns[column.name] = {
					'type': str(column.type),
					'nullable': column.nullable,
					'primary_key': column.primary_key,
					'default': str(column.default) if column.default else None,
					'autoincrement': column.autoincrement,
					'unique': column.unique
				}
			
			# Extract indexes
			for index in table.indexes:
				indexes.append({
					'name': index.name,
					'columns': [col.name for col in index.columns],
					'unique': index.unique
				})
			
			# Extract constraints
			for constraint in table.constraints:
				constraints.append({
					'name': constraint.name,
					'type': type(constraint).__name__,
					'columns': [col.name for col in constraint.columns] if hasattr(constraint, 'columns') else []
				})
			
			# Extract foreign keys
			for fk in table.foreign_keys:
				foreign_keys.append({
					'column': fk.parent.name,
					'references_table': fk.column.table.name,
					'references_column': fk.column.name,
					'constraint_name': fk.constraint.name if fk.constraint else None
				})
		
		# Extract SQLAlchemy relationships
		if hasattr(model_class, '__mapper__'):
			for rel_name, rel in model_class.__mapper__.relationships.items():
				relationships.append({
					'name': rel_name,
					'target_class': rel.mapper.class_.__name__,
					'direction': str(rel.direction),
					'cascade': rel.cascade.save_update if hasattr(rel.cascade, 'save_update') else None
				})
		
		# Determine table characteristics
		is_core_table = 'core' in capability.code.lower() or 'auth' in capability.code.lower()
		supports_multi_tenancy = 'tenant_id' in columns
		audit_enabled = any(col in columns for col in ['created_on', 'changed_on', 'created_by', 'changed_by'])
		
		return TableMetadata(
			table_name=table_name,
			model_class=model_class.__name__,
			capability=capability.code,
			subcapability=subcapability.code,
			columns=columns,
			indexes=indexes,
			constraints=constraints,
			foreign_keys=foreign_keys,
			relationships=relationships,
			module_path=f"{subcapability.module_path}.models",
			is_core_table=is_core_table,
			supports_multi_tenancy=supports_multi_tenancy,
			audit_enabled=audit_enabled
		)
	
	def _calculate_schema_statistics(self, schema: SchemaComposition) -> None:
		"""Calculate statistics for the schema composition."""
		schema.total_tables = len(schema.tables)
		schema.total_columns = sum(len(table.columns) for table in schema.tables.values())
		schema.total_indexes = sum(len(table.indexes) for table in schema.tables.values())
		schema.total_constraints = sum(len(table.constraints) for table in schema.tables.values())
	
	def _generate_schema_hash(self, schema: SchemaComposition) -> str:
		"""Generate a hash for the schema structure."""
		schema_data = {
			'tables': {
				name: {
					'columns': table.columns,
					'indexes': table.indexes,
					'constraints': table.constraints
				}
				for name, table in schema.tables.items()
			}
		}
		
		schema_json = json.dumps(schema_data, sort_keys=True)
		return hashlib.sha256(schema_json.encode()).hexdigest()[:16]
	
	def create_migration_plan(self, 
							 target_schema: SchemaComposition,
							 current_schema: Optional[SchemaComposition] = None) -> MigrationPlan:
		"""
		Create a migration plan to reach the target schema.
		
		Args:
			target_schema: Desired schema composition
			current_schema: Current schema (None for initial creation)
			
		Returns:
			MigrationPlan with ordered migration steps
		"""
		plan = MigrationPlan(
			name=f"Migration to {target_schema.application_name}",
			description=f"Migration plan for {len(target_schema.capabilities)} capabilities",
			target_capabilities=target_schema.capabilities,
			target_tables=list(target_schema.tables.keys()),
			to_version=target_schema.schema_version
		)
		
		if current_schema:
			plan.from_version = current_schema.schema_version
			plan = self._create_update_migration_plan(plan, current_schema, target_schema)
		else:
			plan = self._create_initial_migration_plan(plan, target_schema)
		
		# Calculate estimated duration
		plan.estimated_duration_ms = sum(step.estimated_duration_ms for step in plan.steps)
		
		logger.info(f"Created migration plan with {len(plan.steps)} steps")
		return plan
	
	def _create_initial_migration_plan(self, 
									   plan: MigrationPlan,
									   schema: SchemaComposition) -> MigrationPlan:
		"""Create migration plan for initial database creation."""
		
		# Sort tables by dependencies (foreign keys)
		sorted_tables = self._sort_tables_by_dependencies(schema.tables)
		
		for table_name in sorted_tables:
			table_meta = schema.tables[table_name]
			
			# Create table creation step
			create_sql = self._generate_create_table_sql(table_meta)
			
			step = MigrationStep(
				migration_type=MigrationType.CREATE_TABLE,
				table_name=table_name,
				sql_statement=create_sql,
				capability=table_meta.capability,
				description=f"Create table {table_name}",
				estimated_duration_ms=100,  # Base estimate
				pre_conditions=[],
				post_conditions=[f"Table {table_name} exists"]
			)
			
			plan.steps.append(step)
			
			# Create index creation steps
			for index in table_meta.indexes:
				index_sql = self._generate_create_index_sql(table_name, index)
				
				index_step = MigrationStep(
					migration_type=MigrationType.ADD_INDEX,
					table_name=table_name,
					sql_statement=index_sql,
					capability=table_meta.capability,
					description=f"Create index {index['name']} on {table_name}",
					estimated_duration_ms=50,
					depends_on=[step.id]
				)
				
				plan.steps.append(index_step)
		
		return plan
	
	def _create_update_migration_plan(self, 
									  plan: MigrationPlan,
									  current_schema: SchemaComposition,
									  target_schema: SchemaComposition) -> MigrationPlan:
		"""Create migration plan for schema updates."""
		
		current_tables = set(current_schema.tables.keys())
		target_tables = set(target_schema.tables.keys())
		
		# Tables to create
		new_tables = target_tables - current_tables
		for table_name in new_tables:
			table_meta = target_schema.tables[table_name]
			create_sql = self._generate_create_table_sql(table_meta)
			
			step = MigrationStep(
				migration_type=MigrationType.CREATE_TABLE,
				table_name=table_name,
				sql_statement=create_sql,
				capability=table_meta.capability,
				description=f"Create new table {table_name}",
				estimated_duration_ms=100
			)
			
			plan.steps.append(step)
		
		# Tables to drop
		dropped_tables = current_tables - target_tables
		for table_name in dropped_tables:
			drop_sql = f"DROP TABLE IF EXISTS {table_name};"
			
			step = MigrationStep(
				migration_type=MigrationType.DROP_TABLE,
				table_name=table_name,
				sql_statement=drop_sql,
				capability="UNKNOWN",
				description=f"Drop table {table_name}",
				estimated_duration_ms=50,
				reversible=False
			)
			
			plan.steps.append(step)
		
		# Tables to modify
		common_tables = current_tables & target_tables
		for table_name in common_tables:
			current_table = current_schema.tables[table_name]
			target_table = target_schema.tables[table_name]
			
			modify_steps = self._create_table_modification_steps(current_table, target_table)
			plan.steps.extend(modify_steps)
		
		return plan
	
	def _sort_tables_by_dependencies(self, tables: Dict[str, TableMetadata]) -> List[str]:
		"""Sort tables by foreign key dependencies."""
		# Build dependency graph
		graph = {}
		in_degree = {}
		
		for table_name in tables:
			graph[table_name] = []
			in_degree[table_name] = 0
		
		for table_name, table_meta in tables.items():
			for fk in table_meta.foreign_keys:
				ref_table = fk['references_table']
				if ref_table in tables and ref_table != table_name:
					graph[ref_table].append(table_name)
					in_degree[table_name] += 1
		
		# Topological sort
		queue = [table for table, degree in in_degree.items() if degree == 0]
		result = []
		
		while queue:
			table = queue.pop(0)
			result.append(table)
			
			for dependent in graph[table]:
				in_degree[dependent] -= 1
				if in_degree[dependent] == 0:
					queue.append(dependent)
		
		# Add any remaining tables (in case of circular dependencies)
		remaining = set(tables.keys()) - set(result)
		result.extend(remaining)
		
		return result
	
	def _generate_create_table_sql(self, table_meta: TableMetadata) -> str:
		"""Generate CREATE TABLE SQL statement."""
		columns_sql = []
		
		for col_name, col_info in table_meta.columns.items():
			col_sql = f"{col_name} {col_info['type']}"
			
			if not col_info['nullable']:
				col_sql += " NOT NULL"
			
			if col_info['primary_key']:
				col_sql += " PRIMARY KEY"
			
			if col_info['unique']:
				col_sql += " UNIQUE"
			
			if col_info['default']:
				col_sql += f" DEFAULT {col_info['default']}"
			
			columns_sql.append(col_sql)
		
		# Add foreign key constraints
		for fk in table_meta.foreign_keys:
			fk_sql = f"FOREIGN KEY ({fk['column']}) REFERENCES {fk['references_table']}({fk['references_column']})"
			columns_sql.append(fk_sql)
		
		create_sql = f"CREATE TABLE {table_meta.table_name} (\n  "
		create_sql += ",\n  ".join(columns_sql)
		create_sql += "\n);"
		
		return create_sql
	
	def _generate_create_index_sql(self, table_name: str, index_info: Dict[str, Any]) -> str:
		"""Generate CREATE INDEX SQL statement."""
		index_type = "UNIQUE INDEX" if index_info['unique'] else "INDEX"
		columns = ", ".join(index_info['columns'])
		
		return f"CREATE {index_type} {index_info['name']} ON {table_name} ({columns});"
	
	def _create_table_modification_steps(self, 
										 current_table: TableMetadata,
										 target_table: TableMetadata) -> List[MigrationStep]:
		"""Create migration steps for table modifications."""
		steps = []
		
		current_columns = set(current_table.columns.keys())
		target_columns = set(target_table.columns.keys())
		
		# Add new columns
		new_columns = target_columns - current_columns
		for col_name in new_columns:
			col_info = target_table.columns[col_name]
			add_sql = f"ALTER TABLE {target_table.table_name} ADD COLUMN {col_name} {col_info['type']}"
			
			if not col_info['nullable']:
				add_sql += " NOT NULL"
			
			if col_info['default']:
				add_sql += f" DEFAULT {col_info['default']}"
			
			add_sql += ";"
			
			step = MigrationStep(
				migration_type=MigrationType.ADD_COLUMN,
				table_name=target_table.table_name,
				sql_statement=add_sql,
				rollback_statement=f"ALTER TABLE {target_table.table_name} DROP COLUMN {col_name};",
				capability=target_table.capability,
				description=f"Add column {col_name} to {target_table.table_name}",
				estimated_duration_ms=30
			)
			
			steps.append(step)
		
		# Drop removed columns
		dropped_columns = current_columns - target_columns
		for col_name in dropped_columns:
			drop_sql = f"ALTER TABLE {target_table.table_name} DROP COLUMN {col_name};"
			
			step = MigrationStep(
				migration_type=MigrationType.DROP_COLUMN,
				table_name=target_table.table_name,
				sql_statement=drop_sql,
				capability=target_table.capability,
				description=f"Drop column {col_name} from {target_table.table_name}",
				estimated_duration_ms=30,
				reversible=False
			)
			
			steps.append(step)
		
		return steps
	
	def execute_migration_plan(self, 
							   plan: MigrationPlan,
							   dry_run: bool = False) -> Dict[str, Any]:
		"""
		Execute a migration plan.
		
		Args:
			plan: Migration plan to execute
			dry_run: If True, only validate without executing
			
		Returns:
			Execution result with status and details
		"""
		if not self.engine:
			raise ValueError("Database engine not configured")
		
		execution_result = {
			"success": False,
			"dry_run": dry_run,
			"plan_id": plan.id,
			"executed_steps": 0,
			"total_steps": len(plan.steps),
			"execution_time_ms": 0,
			"errors": [],
			"warnings": []
		}
		
		start_time = datetime.utcnow()
		
		try:
			with self.engine.connect() as connection:
				transaction = connection.begin()
				
				try:
					for i, step in enumerate(plan.steps):
						logger.info(f"Executing step {i+1}/{len(plan.steps)}: {step.description}")
						
						# Validate pre-conditions
						for condition in step.pre_conditions:
							if not self._validate_condition(connection, condition):
								raise Exception(f"Pre-condition failed: {condition}")
						
						if not dry_run:
							# Execute the migration step
							connection.execute(text(step.sql_statement))
						
						# Validate post-conditions
						for condition in step.post_conditions:
							if not dry_run and not self._validate_condition(connection, condition):
								raise Exception(f"Post-condition failed: {condition}")
						
						execution_result["executed_steps"] += 1
					
					if not dry_run:
						transaction.commit()
						logger.info("Migration plan executed successfully")
					else:
						transaction.rollback()
						logger.info("Migration plan validated successfully (dry run)")
					
					execution_result["success"] = True
					
				except Exception as e:
					transaction.rollback()
					error_msg = f"Migration failed at step {execution_result['executed_steps'] + 1}: {str(e)}"
					logger.error(error_msg)
					execution_result["errors"].append(error_msg)
		
		except Exception as e:
			error_msg = f"Failed to execute migration plan: {str(e)}"
			logger.error(error_msg)
			execution_result["errors"].append(error_msg)
		
		end_time = datetime.utcnow()
		execution_result["execution_time_ms"] = int((end_time - start_time).total_seconds() * 1000)
		
		return execution_result
	
	def _validate_condition(self, connection, condition: str) -> bool:
		"""Validate a condition against the database."""
		try:
			# Simple condition validation - can be extended
			if condition.startswith("Table ") and condition.endswith(" exists"):
				table_name = condition.replace("Table ", "").replace(" exists", "")
				inspector = inspect(connection)
				return inspector.has_table(table_name)
			
			return True  # Default to true for unknown conditions
			
		except Exception as e:
			logger.warning(f"Failed to validate condition '{condition}': {e}")
			return False
	
	def validate_schema_composition(self, schema: SchemaComposition) -> Dict[str, Any]:
		"""Validate a schema composition for consistency and conflicts."""
		validation_result = {
			"valid": True,
			"errors": [],
			"warnings": []
		}
		
		# Check for table name conflicts
		table_names = set()
		for table_name in schema.tables:
			if table_name in table_names:
				validation_result["errors"].append(f"Duplicate table name: {table_name}")
				validation_result["valid"] = False
			table_names.add(table_name)
		
		# Check for foreign key integrity
		for table_name, table_meta in schema.tables.items():
			for fk in table_meta.foreign_keys:
				ref_table = fk['references_table']
				if ref_table not in schema.tables:
					validation_result["errors"].append(
						f"Foreign key in {table_name} references non-existent table {ref_table}"
					)
					validation_result["valid"] = False
		
		# Check for missing tenant_id columns in multi-tenant tables
		for table_name, table_meta in schema.tables.items():
			if table_meta.supports_multi_tenancy and 'tenant_id' not in table_meta.columns:
				validation_result["warnings"].append(
					f"Table {table_name} claims multi-tenancy support but lacks tenant_id column"
				)
		
		return validation_result
	
	def get_current_schema_version(self) -> Optional[str]:
		"""Get the current database schema version."""
		if not self.engine:
			return None
		
		try:
			with self.engine.connect() as connection:
				# Check if version table exists
				inspector = inspect(connection)
				if not inspector.has_table('apg_schema_version'):
					return None
				
				# Get current version
				result = connection.execute(text(
					"SELECT version FROM apg_schema_version ORDER BY applied_at DESC LIMIT 1"
				))
				row = result.fetchone()
				return row[0] if row else None
				
		except Exception as e:
			logger.error(f"Failed to get current schema version: {e}")
			return None
	
	def backup_database(self, backup_path: Optional[str] = None) -> str:
		"""Create a database backup before migration."""
		if not backup_path:
			timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
			backup_path = f"backup_apg_{timestamp}.sql"
		
		# This is a placeholder - actual implementation would depend on database type
		logger.info(f"Database backup would be created at: {backup_path}")
		return backup_path

# Global database manager instance
_database_manager_instance: Optional[DatabaseManager] = None

def get_database_manager(database_url: Optional[str] = None) -> DatabaseManager:
	"""Get the global database manager instance."""
	global _database_manager_instance
	if _database_manager_instance is None:
		_database_manager_instance = DatabaseManager(database_url)
	return _database_manager_instance