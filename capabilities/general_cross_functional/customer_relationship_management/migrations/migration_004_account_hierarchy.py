"""
APG Customer Relationship Management - Account Hierarchy Migration

Database migration to create account hierarchy audit log and supporting structures
for advanced hierarchy management and relationship tracking.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from .base_migration import BaseMigration, MigrationDirection


logger = logging.getLogger(__name__)


class AccountHierarchyMigration(BaseMigration):
	"""Migration for account hierarchy functionality"""
	
	def _get_migration_id(self) -> str:
		return "004_account_hierarchy"
	
	def _get_version(self) -> str:
		return "004"
	
	def _get_description(self) -> str:
		return "Create account hierarchy audit log and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating account hierarchy structures...")
			
			# Create hierarchy relationship type enum
			await connection.execute("""
				CREATE TYPE crm_hierarchy_relationship_type AS ENUM (
					'parent_child', 'subsidiary', 'division', 'branch', 
					'affiliate', 'joint_venture', 'partnership', 'acquisition'
				)
			""")
			
			# Create hierarchy audit log table
			await connection.execute("""
				CREATE TABLE crm_hierarchy_audit_log (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					account_id TEXT NOT NULL,
					old_parent_id TEXT,
					new_parent_id TEXT,
					relationship_type crm_hierarchy_relationship_type DEFAULT 'parent_child',
					change_notes TEXT,
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					
					-- Foreign key constraints
					FOREIGN KEY (account_id) REFERENCES crm_accounts(id) ON DELETE CASCADE,
					FOREIGN KEY (old_parent_id) REFERENCES crm_accounts(id) ON DELETE SET NULL,
					FOREIGN KEY (new_parent_id) REFERENCES crm_accounts(id) ON DELETE SET NULL,
					
					-- Constraints
					CONSTRAINT check_change_notes_length CHECK (char_length(change_notes) <= 1000)
				)
			""")
			
			# Create indexes for hierarchy audit log
			await connection.execute("""
				CREATE INDEX idx_crm_hierarchy_audit_tenant 
				ON crm_hierarchy_audit_log(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_hierarchy_audit_account 
				ON crm_hierarchy_audit_log(account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_hierarchy_audit_created_at 
				ON crm_hierarchy_audit_log(created_at DESC)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_hierarchy_audit_tenant_account 
				ON crm_hierarchy_audit_log(tenant_id, account_id)
			""")
			
			# Add additional indexes to accounts table for efficient hierarchy queries
			await connection.execute("""
				CREATE INDEX idx_crm_accounts_parent_tenant 
				ON crm_accounts(parent_account_id, tenant_id)
				WHERE parent_account_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_accounts_hierarchy_path 
				ON crm_accounts(tenant_id, parent_account_id, id)
			""")
			
			# Create materialized view for hierarchy statistics (optional optimization)
			await connection.execute("""
				CREATE MATERIALIZED VIEW crm_account_hierarchy_stats AS
				SELECT 
					a.id as account_id,
					a.tenant_id,
					a.account_name,
					a.parent_account_id,
					
					-- Direct children count
					(SELECT COUNT(*) FROM crm_accounts c 
					 WHERE c.parent_account_id = a.id AND c.tenant_id = a.tenant_id 
					 AND c.status = 'active') as direct_children_count,
					
					-- Hierarchy level (simplified calculation for root and level 1)
					CASE 
						WHEN a.parent_account_id IS NULL THEN 0
						ELSE 1
					END as hierarchy_level,
					
					-- Is leaf node
					CASE 
						WHEN NOT EXISTS (
							SELECT 1 FROM crm_accounts c 
							WHERE c.parent_account_id = a.id AND c.tenant_id = a.tenant_id 
							AND c.status = 'active'
						) THEN TRUE
						ELSE FALSE
					END as is_leaf,
					
					-- Aggregated metrics
					COALESCE(a.annual_revenue, 0) as own_revenue,
					COALESCE(a.employee_count, 0) as own_employee_count,
					
					-- Timestamps
					a.created_at,
					a.updated_at
					
				FROM crm_accounts a
				WHERE a.status = 'active'
			""")
			
			# Create index on materialized view
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_hierarchy_stats_account 
				ON crm_account_hierarchy_stats(account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_hierarchy_stats_tenant 
				ON crm_account_hierarchy_stats(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_hierarchy_stats_parent 
				ON crm_account_hierarchy_stats(parent_account_id)
				WHERE parent_account_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_hierarchy_stats_level 
				ON crm_account_hierarchy_stats(hierarchy_level)
			""")
			
			# Create function to refresh hierarchy stats
			await connection.execute("""
				CREATE OR REPLACE FUNCTION refresh_crm_hierarchy_stats()
				RETURNS VOID AS $$
				BEGIN
					REFRESH MATERIALIZED VIEW CONCURRENTLY crm_account_hierarchy_stats;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function to get account hierarchy path
			await connection.execute("""
				CREATE OR REPLACE FUNCTION get_account_hierarchy_path(
					account_id TEXT,
					tenant_filter TEXT DEFAULT NULL
				)
				RETURNS TABLE(
					level INTEGER,
					account_id TEXT,
					account_name TEXT,
					parent_account_id TEXT
				) AS $$
				BEGIN
					RETURN QUERY
					WITH RECURSIVE hierarchy_path AS (
						-- Base case: start with the given account
						SELECT 
							0 as level,
							a.id as account_id,
							a.account_name,
							a.parent_account_id
						FROM crm_accounts a
						WHERE a.id = account_id
						AND (tenant_filter IS NULL OR a.tenant_id = tenant_filter)
						
						UNION ALL
						
						-- Recursive case: get parent accounts
						SELECT 
							hp.level + 1,
							a.id,
							a.account_name,
							a.parent_account_id
						FROM crm_accounts a
						JOIN hierarchy_path hp ON a.id = hp.parent_account_id
						WHERE hp.level < 10 -- Prevent infinite recursion
					)
					SELECT * FROM hierarchy_path
					ORDER BY level DESC;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function to get account subtree
			await connection.execute("""
				CREATE OR REPLACE FUNCTION get_account_subtree(
					root_account_id TEXT,
					tenant_filter TEXT DEFAULT NULL,
					max_depth INTEGER DEFAULT 5
				)
				RETURNS TABLE(
					level INTEGER,
					account_id TEXT,
					account_name TEXT,
					parent_account_id TEXT,
					path TEXT[]
				) AS $$
				BEGIN
					RETURN QUERY
					WITH RECURSIVE account_subtree AS (
						-- Base case: start with root account
						SELECT 
							0 as level,
							a.id as account_id,
							a.account_name,
							a.parent_account_id,
							ARRAY[a.id] as path
						FROM crm_accounts a
						WHERE a.id = root_account_id
						AND (tenant_filter IS NULL OR a.tenant_id = tenant_filter)
						
						UNION ALL
						
						-- Recursive case: get child accounts
						SELECT 
							ast.level + 1,
							a.id,
							a.account_name,
							a.parent_account_id,
							ast.path || a.id
						FROM crm_accounts a
						JOIN account_subtree ast ON a.parent_account_id = ast.account_id
						WHERE ast.level < max_depth
						AND NOT (a.id = ANY(ast.path)) -- Prevent cycles
					)
					SELECT * FROM account_subtree
					ORDER BY level, account_name;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Account hierarchy structures created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create account hierarchy structures: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back account hierarchy migration...")
			
			# Drop functions
			await connection.execute("DROP FUNCTION IF EXISTS get_account_subtree CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS get_account_hierarchy_path CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS refresh_crm_hierarchy_stats CASCADE")
			
			# Drop materialized view
			await connection.execute("DROP MATERIALIZED VIEW IF EXISTS crm_account_hierarchy_stats CASCADE")
			
			# Drop indexes on accounts table
			await connection.execute("DROP INDEX IF EXISTS idx_crm_accounts_hierarchy_path")
			await connection.execute("DROP INDEX IF EXISTS idx_crm_accounts_parent_tenant")
			
			# Drop audit log table (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_hierarchy_audit_log CASCADE")
			
			# Drop enum
			await connection.execute("DROP TYPE IF EXISTS crm_hierarchy_relationship_type CASCADE")
			
			logger.info("✅ Account hierarchy migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback account hierarchy migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if audit log table exists
			table_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.tables 
					WHERE table_name = 'crm_hierarchy_audit_log'
				)
			""")
			
			if not table_exists:
				return False
			
			# Check if enum exists
			enum_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM pg_type 
					WHERE typname = 'crm_hierarchy_relationship_type'
				)
			""")
			
			if not enum_exists:
				return False
			
			# Check if materialized view exists
			view_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.tables 
					WHERE table_name = 'crm_account_hierarchy_stats'
					AND table_type = 'BASE TABLE'
				)
			""")
			
			if not view_exists:
				return False
			
			# Check if functions exist
			function_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.routines 
				WHERE routine_name IN (
					'get_account_hierarchy_path',
					'get_account_subtree',
					'refresh_crm_hierarchy_stats'
				)
			""")
			
			if function_count < 3:
				return False
			
			# Check key indexes
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename = 'crm_hierarchy_audit_log'
				AND indexname IN (
					'idx_crm_hierarchy_audit_tenant',
					'idx_crm_hierarchy_audit_account'
				)
			""")
			
			if index_count < 2:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False