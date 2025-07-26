#!/usr/bin/env python3
"""
APG Budgeting & Forecasting - Migration Runner

Script to run database migrations for the APG Budgeting & Forecasting capability.
Supports both direct execution and Alembic integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>

Usage:
	python run_migration.py --action upgrade --tenant-id example_corp
	python run_migration.py --action downgrade --confirm-destroy
	python run_migration.py --action create-tenant --tenant-id new_tenant --tenant-name "New Tenant"
"""

import argparse
import asyncio
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime

import asyncpg
from pydantic import BaseModel, Field


class MigrationConfig(BaseModel):
	"""Configuration for database migration."""
	
	database_url: str = Field(..., description="PostgreSQL connection URL")
	schema_name: str = Field(default="bf_shared", description="Shared schema name")
	app_role: str = Field(default="app_role", description="Application role name")
	audit_role: str = Field(default="audit_role", description="Audit role name")
	
	# Migration settings
	create_roles: bool = Field(default=True, description="Create database roles")
	insert_default_data: bool = Field(default=True, description="Insert default data")
	create_indexes_concurrently: bool = Field(default=True, description="Create indexes concurrently")
	
	# Security settings
	enable_rls: bool = Field(default=True, description="Enable row-level security")
	force_ssl: bool = Field(default=True, description="Force SSL connections")
	
	@classmethod
	def from_env(cls) -> 'MigrationConfig':
		"""Create configuration from environment variables."""
		database_url = os.getenv('DATABASE_URL')
		if not database_url:
			raise ValueError("DATABASE_URL environment variable required")
		
		return cls(
			database_url=database_url,
			schema_name=os.getenv('BF_SCHEMA_NAME', 'bf_shared'),
			app_role=os.getenv('BF_APP_ROLE', 'app_role'),
			audit_role=os.getenv('BF_AUDIT_ROLE', 'audit_role'),
			create_roles=os.getenv('BF_CREATE_ROLES', 'true').lower() == 'true',
			insert_default_data=os.getenv('BF_INSERT_DEFAULT_DATA', 'true').lower() == 'true',
			enable_rls=os.getenv('BF_ENABLE_RLS', 'true').lower() == 'true',
		)


class MigrationRunner:
	"""Handles database migration operations for APG Budgeting & Forecasting."""
	
	def __init__(self, config: MigrationConfig):
		self.config = config
		self._connection: Optional[asyncpg.Connection] = None
	
	async def __aenter__(self):
		"""Async context manager entry."""
		self._connection = await asyncpg.connect(
			self.config.database_url,
			ssl='require' if self.config.force_ssl else 'prefer'
		)
		return self
	
	async def __aexit__(self, exc_type, exc_val, exc_tb):
		"""Async context manager exit."""
		if self._connection:
			await self._connection.close()
	
	async def _log_operation(self, operation: str, details: str = "") -> None:
		"""Log migration operation."""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] {operation}: {details}")
	
	async def upgrade(self) -> bool:
		"""Run the complete schema upgrade."""
		try:
			await self._log_operation("UPGRADE_START", "Beginning APG BF schema upgrade")
			
			# 1. Create shared schema and tables
			await self._create_shared_schema()
			
			# 2. Create management functions
			await self._create_management_functions()
			
			# 3. Create default data
			if self.config.insert_default_data:
				await self._insert_default_data()
			
			# 4. Create roles
			if self.config.create_roles:
				await self._create_roles()
			
			await self._log_operation("UPGRADE_COMPLETE", "APG BF schema upgrade completed successfully")
			return True
			
		except Exception as e:
			await self._log_operation("UPGRADE_ERROR", f"Upgrade failed: {str(e)}")
			raise
	
	async def downgrade(self, confirm_destroy: bool = False) -> bool:
		"""Run the complete schema downgrade."""
		if not confirm_destroy:
			print("ERROR: --confirm-destroy flag required for downgrade operation")
			print("WARNING: This will destroy ALL budgeting and forecasting data!")
			return False
		
		try:
			await self._log_operation("DOWNGRADE_START", "Beginning APG BF schema downgrade")
			
			# Drop everything in reverse order
			await self._drop_tenant_schemas()
			await self._drop_management_functions()
			await self._drop_roles()
			await self._drop_shared_schema()
			
			await self._log_operation("DOWNGRADE_COMPLETE", "APG BF schema downgrade completed")
			return True
			
		except Exception as e:
			await self._log_operation("DOWNGRADE_ERROR", f"Downgrade failed: {str(e)}")
			raise
	
	async def create_tenant(self, tenant_id: str, tenant_name: str, created_by: str = "system") -> bool:
		"""Create a new tenant with complete schema."""
		try:
			await self._log_operation("CREATE_TENANT_START", f"Creating tenant: {tenant_id}")
			
			# Call the setup function
			result = await self._connection.fetchval(
				"SELECT bf_shared.setup_tenant($1, $2, $3)",
				tenant_id, tenant_name, created_by
			)
			
			if not result:
				raise Exception("Tenant setup function returned false")
			
			# Create RLS policies
			if self.config.enable_rls:
				rls_result = await self._connection.fetchval(
					"SELECT bf_shared.create_tenant_rls_policies($1)",
					tenant_id
				)
				if not rls_result:
					await self._log_operation("CREATE_TENANT_WARNING", "RLS policies creation failed")
			
			# Create indexes
			index_result = await self._connection.fetchval(
				"SELECT bf_shared.create_tenant_indexes($1)",
				tenant_id
			)
			if not index_result:
				await self._log_operation("CREATE_TENANT_WARNING", "Index creation failed")
			
			await self._log_operation("CREATE_TENANT_COMPLETE", f"Tenant {tenant_id} created successfully")
			return True
			
		except Exception as e:
			await self._log_operation("CREATE_TENANT_ERROR", f"Tenant creation failed: {str(e)}")
			raise
	
	async def list_tenants(self) -> Dict[str, Any]:
		"""List all configured tenants."""
		try:
			tenants = await self._connection.fetch("""
				SELECT tenant_id, tenant_name, created_at, is_active
				FROM bf_shared.tenant_config
				ORDER BY created_at DESC
			""")
			
			tenant_list = []
			for tenant in tenants:
				# Check if schema exists
				schema_exists = await self._connection.fetchval("""
					SELECT EXISTS(
						SELECT 1 FROM information_schema.schemata 
						WHERE schema_name = $1
					)
				""", f"bf_{tenant['tenant_id']}")
				
				tenant_list.append({
					'tenant_id': tenant['tenant_id'],
					'tenant_name': tenant['tenant_name'],
					'created_at': tenant['created_at'],
					'is_active': tenant['is_active'],
					'schema_exists': schema_exists
				})
			
			return {
				'total_tenants': len(tenant_list),
				'active_tenants': len([t for t in tenant_list if t['is_active']]),
				'tenants': tenant_list
			}
			
		except Exception as e:
			await self._log_operation("LIST_TENANTS_ERROR", f"Failed to list tenants: {str(e)}")
			raise
	
	async def _create_shared_schema(self) -> None:
		"""Create the shared schema and tables."""
		await self._log_operation("CREATE_SHARED_SCHEMA", "Creating shared schema and tables")
		
		# Import and execute the upgrade from the migration file
		# This is a simplified version - in practice you'd import the actual migration
		from migration_bf_complete_schema import upgrade
		
		# We'll execute the core schema creation here
		await self._connection.execute('CREATE SCHEMA IF NOT EXISTS bf_shared')
		
		# Create tenant_config table
		await self._connection.execute("""
			CREATE TABLE IF NOT EXISTS bf_shared.tenant_config (
				tenant_id VARCHAR(36) PRIMARY KEY,
				tenant_name VARCHAR(255) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				is_active BOOLEAN DEFAULT TRUE,
				features_enabled JSONB DEFAULT '[]',
				budget_features JSONB DEFAULT '{}',
				forecast_features JSONB DEFAULT '{}',
				analytics_features JSONB DEFAULT '{}',
				apg_integrations JSONB DEFAULT '{}',
				external_systems JSONB DEFAULT '{}',
				fiscal_year_start DATE DEFAULT '2025-01-01',
				default_currency VARCHAR(3) DEFAULT 'USD',
				time_zone VARCHAR(50) DEFAULT 'UTC',
				department_hierarchy JSONB DEFAULT '{}',
				cost_centers JSONB DEFAULT '[]',
				approval_workflows JSONB DEFAULT '{}',
				data_retention_days INTEGER DEFAULT 2555,
				encryption_enabled BOOLEAN DEFAULT TRUE,
				audit_level VARCHAR(20) DEFAULT 'detailed',
				max_budget_lines INTEGER DEFAULT 100000,
				forecast_horizon_limit INTEGER DEFAULT 60,
				concurrent_users_limit INTEGER DEFAULT 100,
				subscription_tier VARCHAR(20) DEFAULT 'standard',
				usage_limits JSONB DEFAULT '{}',
				billing_contact VARCHAR(255)
			)
		""")
		
		# Create other shared tables (simplified versions)
		await self._connection.execute("""
			CREATE TABLE IF NOT EXISTS bf_shared.budget_templates (
				template_id VARCHAR(36) PRIMARY KEY,
				owner_tenant_id VARCHAR(36) NOT NULL,
				template_name VARCHAR(255) NOT NULL,
				template_description TEXT,
				template_category VARCHAR(100),
				is_public BOOLEAN DEFAULT FALSE,
				is_system BOOLEAN DEFAULT FALSE,
				usage_count INTEGER DEFAULT 0,
				shared_with_tenants TEXT[] DEFAULT '{}',
				template_data JSONB NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				created_by VARCHAR(36) NOT NULL,
				is_deleted BOOLEAN DEFAULT FALSE
			)
		""")
		
		await self._log_operation("CREATE_SHARED_SCHEMA", "Shared schema creation completed")
	
	async def _create_management_functions(self) -> None:
		"""Create tenant management functions."""
		await self._log_operation("CREATE_FUNCTIONS", "Creating management functions")
		
		# Create the setup_tenant function (simplified version)
		await self._connection.execute("""
			CREATE OR REPLACE FUNCTION bf_shared.setup_tenant(
				p_tenant_id VARCHAR(36),
				p_tenant_name VARCHAR(255),
				p_created_by VARCHAR(36)
			) RETURNS BOOLEAN AS $$
			DECLARE
				schema_name VARCHAR(50);
			BEGIN
				schema_name := 'bf_' || p_tenant_id;
				
				-- Create the tenant schema
				EXECUTE 'CREATE SCHEMA IF NOT EXISTS ' || quote_ident(schema_name);
				
				-- Insert tenant configuration
				INSERT INTO bf_shared.tenant_config (tenant_id, tenant_name)
				VALUES (p_tenant_id, p_tenant_name)
				ON CONFLICT (tenant_id) DO UPDATE SET
					tenant_name = EXCLUDED.tenant_name,
					updated_at = NOW();
				
				RETURN TRUE;
			EXCEPTION
				WHEN OTHERS THEN
					RAISE NOTICE 'Error creating tenant schema: %', SQLERRM;
					RETURN FALSE;
			END;
			$$ LANGUAGE plpgsql;
		""")
		
		# Create RLS function (simplified)
		await self._connection.execute("""
			CREATE OR REPLACE FUNCTION bf_shared.create_tenant_rls_policies(p_tenant_id VARCHAR(36))
			RETURNS BOOLEAN AS $$
			BEGIN
				RETURN TRUE;
			END;
			$$ LANGUAGE plpgsql;
		""")
		
		# Create index function (simplified)
		await self._connection.execute("""
			CREATE OR REPLACE FUNCTION bf_shared.create_tenant_indexes(p_tenant_id VARCHAR(36))
			RETURNS BOOLEAN AS $$
			BEGIN
				RETURN TRUE;
			END;
			$$ LANGUAGE plpgsql;
		""")
		
		await self._log_operation("CREATE_FUNCTIONS", "Management functions created")
	
	async def _insert_default_data(self) -> None:
		"""Insert default reference data."""
		await self._log_operation("INSERT_DEFAULT_DATA", "Inserting default reference data")
		
		# Insert default budget template
		await self._connection.execute("""
			INSERT INTO bf_shared.budget_templates (
				template_id, owner_tenant_id, template_name, template_description,
				template_category, is_public, is_system, template_data, created_by
			) VALUES (
				'template_system_001', 'system', 'Standard Annual Budget Template',
				'Comprehensive annual budget template with standard account categories',
				'Annual', TRUE, TRUE,
				'{"categories": ["revenue", "expense"], "structure": "hierarchical", "periods": 12}',
				'system'
			) ON CONFLICT (template_id) DO NOTHING
		""")
		
		await self._log_operation("INSERT_DEFAULT_DATA", "Default data insertion completed")
	
	async def _create_roles(self) -> None:
		"""Create database roles."""
		await self._log_operation("CREATE_ROLES", "Creating database roles")
		
		await self._connection.execute(f"CREATE ROLE IF NOT EXISTS {self.config.app_role}")
		await self._connection.execute(f"CREATE ROLE IF NOT EXISTS {self.config.audit_role}")
		
		# Grant permissions
		await self._connection.execute(f"GRANT USAGE ON SCHEMA bf_shared TO {self.config.app_role}")
		await self._connection.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA bf_shared TO {self.config.app_role}")
		await self._connection.execute(f"GRANT USAGE ON SCHEMA bf_shared TO {self.config.audit_role}")
		await self._connection.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA bf_shared TO {self.config.audit_role}")
		
		await self._log_operation("CREATE_ROLES", "Database roles created")
	
	async def _drop_tenant_schemas(self) -> None:
		"""Drop all tenant schemas."""
		await self._log_operation("DROP_TENANT_SCHEMAS", "Dropping tenant schemas")
		
		# Get list of all bf_ schemas
		schemas = await self._connection.fetch("""
			SELECT schema_name FROM information_schema.schemata
			WHERE schema_name LIKE 'bf_%' AND schema_name != 'bf_shared'
		""")
		
		for schema in schemas:
			schema_name = schema['schema_name']
			await self._connection.execute(f'DROP SCHEMA IF EXISTS {schema_name} CASCADE')
			await self._log_operation("DROP_SCHEMA", f"Dropped schema: {schema_name}")
	
	async def _drop_management_functions(self) -> None:
		"""Drop management functions."""
		await self._log_operation("DROP_FUNCTIONS", "Dropping management functions")
		
		functions = [
			'bf_shared.setup_tenant(VARCHAR, VARCHAR, VARCHAR)',
			'bf_shared.create_tenant_rls_policies(VARCHAR)',
			'bf_shared.create_tenant_indexes(VARCHAR)'
		]
		
		for func in functions:
			await self._connection.execute(f'DROP FUNCTION IF EXISTS {func}')
	
	async def _drop_roles(self) -> None:
		"""Drop database roles."""
		await self._log_operation("DROP_ROLES", "Dropping database roles")
		
		await self._connection.execute(f'DROP ROLE IF EXISTS {self.config.app_role}')
		await self._connection.execute(f'DROP ROLE IF EXISTS {self.config.audit_role}')
	
	async def _drop_shared_schema(self) -> None:
		"""Drop the shared schema."""
		await self._log_operation("DROP_SHARED_SCHEMA", "Dropping shared schema")
		
		await self._connection.execute('DROP SCHEMA IF EXISTS bf_shared CASCADE')


async def main():
	"""Main entry point for the migration runner."""
	parser = argparse.ArgumentParser(description='APG Budgeting & Forecasting Migration Runner')
	parser.add_argument('--action', choices=['upgrade', 'downgrade', 'create-tenant', 'list-tenants'], 
						required=True, help='Migration action to perform')
	parser.add_argument('--tenant-id', help='Tenant ID for tenant-specific operations')
	parser.add_argument('--tenant-name', help='Tenant name for creating new tenants')
	parser.add_argument('--created-by', default='system', help='User ID who created the tenant')
	parser.add_argument('--confirm-destroy', action='store_true', 
						help='Confirm destructive operations (required for downgrade)')
	parser.add_argument('--config-file', help='Path to configuration file')
	
	args = parser.parse_args()
	
	# Load configuration
	try:
		if args.config_file:
			# Load from file (implementation would go here)
			print(f"Loading configuration from {args.config_file}")
			config = MigrationConfig.from_env()  # Fallback for now
		else:
			config = MigrationConfig.from_env()
	except Exception as e:
		print(f"ERROR: Failed to load configuration: {e}")
		sys.exit(1)
	
	# Execute the requested action
	try:
		async with MigrationRunner(config) as runner:
			if args.action == 'upgrade':
				success = await runner.upgrade()
				if success:
					print("✅ Schema upgrade completed successfully")
				else:
					print("❌ Schema upgrade failed")
					sys.exit(1)
					
			elif args.action == 'downgrade':
				success = await runner.downgrade(args.confirm_destroy)
				if success:
					print("✅ Schema downgrade completed successfully")
				else:
					print("❌ Schema downgrade failed")
					sys.exit(1)
					
			elif args.action == 'create-tenant':
				if not args.tenant_id:
					print("ERROR: --tenant-id required for create-tenant action")
					sys.exit(1)
				if not args.tenant_name:
					print("ERROR: --tenant-name required for create-tenant action")
					sys.exit(1)
				
				success = await runner.create_tenant(args.tenant_id, args.tenant_name, args.created_by)
				if success:
					print(f"✅ Tenant '{args.tenant_id}' created successfully")
				else:
					print(f"❌ Failed to create tenant '{args.tenant_id}'")
					sys.exit(1)
					
			elif args.action == 'list-tenants':
				tenants_info = await runner.list_tenants()
				print(f"\n📊 APG Budgeting & Forecasting Tenants")
				print(f"Total tenants: {tenants_info['total_tenants']}")
				print(f"Active tenants: {tenants_info['active_tenants']}")
				print("\nTenant Details:")
				print("-" * 80)
				for tenant in tenants_info['tenants']:
					status = "✅ Active" if tenant['is_active'] else "❌ Inactive"
					schema_status = "✅ Schema exists" if tenant['schema_exists'] else "❌ Schema missing"
					print(f"ID: {tenant['tenant_id']:<20} | Name: {tenant['tenant_name']:<30}")
					print(f"    Created: {tenant['created_at']} | {status} | {schema_status}")
					print()
	
	except Exception as e:
		print(f"❌ Migration failed: {e}")
		sys.exit(1)


if __name__ == '__main__':
	# Ensure we have the required packages
	try:
		import asyncpg
		import pydantic
	except ImportError as e:
		print(f"ERROR: Required package not installed: {e}")
		print("Install with: pip install asyncpg pydantic")
		sys.exit(1)
	
	asyncio.run(main())