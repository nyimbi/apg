"""
APG Customer Relationship Management - Territory Management Migration

Database migration to create territory management tables and supporting structures
for sales territory assignment and geographic coverage analysis.

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


class TerritoryManagementMigration(BaseMigration):
	"""Migration for territory management functionality"""
	
	def _get_migration_id(self) -> str:
		return "006_territory_management"
	
	def _get_version(self) -> str:
		return "006"
	
	def _get_description(self) -> str:
		return "Create territory management tables and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating territory management structures...")
			
			# Create territory type enum
			await connection.execute("""
				CREATE TYPE crm_territory_type AS ENUM (
					'geographic', 'industry', 'account_size', 'product', 'channel', 'hybrid'
				)
			""")
			
			# Create territory status enum
			await connection.execute("""
				CREATE TYPE crm_territory_status AS ENUM (
					'active', 'inactive', 'planning', 'archived'
				)
			""")
			
			# Create assignment type enum
			await connection.execute("""
				CREATE TYPE crm_assignment_type AS ENUM (
					'primary', 'secondary', 'overlay', 'shared'
				)
			""")
			
			# Create territories table
			await connection.execute("""
				CREATE TABLE crm_territories (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					territory_name TEXT NOT NULL,
					territory_code TEXT,
					territory_type crm_territory_type NOT NULL,
					status crm_territory_status DEFAULT 'active',
					
					-- Assignment
					owner_id TEXT NOT NULL,
					sales_rep_ids TEXT[] DEFAULT '{}',
					
					-- Geographic criteria
					countries TEXT[] DEFAULT '{}',
					states_provinces TEXT[] DEFAULT '{}',
					cities TEXT[] DEFAULT '{}',
					postal_codes TEXT[] DEFAULT '{}',
					
					-- Business criteria
					industries TEXT[] DEFAULT '{}',
					company_size_min INTEGER,
					company_size_max INTEGER,
					revenue_min DECIMAL(15,2),
					revenue_max DECIMAL(15,2),
					
					-- Product/service criteria
					product_lines TEXT[] DEFAULT '{}',
					service_types TEXT[] DEFAULT '{}',
					
					-- Goals and metrics
					annual_quota DECIMAL(15,2),
					account_target INTEGER,
					
					-- Metadata
					description TEXT,
					notes TEXT,
					rules JSONB DEFAULT '{}',
					metadata JSONB DEFAULT '{}',
					
					-- Performance tracking (cached values)
					current_accounts INTEGER DEFAULT 0,
					current_revenue DECIMAL(15,2) DEFAULT 0.0,
					quota_achievement DECIMAL(5,2) DEFAULT 0.0,
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Constraints
					CONSTRAINT check_territory_name_length CHECK (char_length(territory_name) >= 1 AND char_length(territory_name) <= 200),
					CONSTRAINT check_territory_code_length CHECK (char_length(territory_code) <= 50),
					CONSTRAINT check_description_length CHECK (char_length(description) <= 2000),
					CONSTRAINT check_notes_length CHECK (char_length(notes) <= 2000),
					CONSTRAINT check_company_size_range CHECK (company_size_min IS NULL OR company_size_max IS NULL OR company_size_min <= company_size_max),
					CONSTRAINT check_revenue_range CHECK (revenue_min IS NULL OR revenue_max IS NULL OR revenue_min <= revenue_max),
					CONSTRAINT check_quotas_positive CHECK (annual_quota IS NULL OR annual_quota >= 0),
					CONSTRAINT check_account_target_positive CHECK (account_target IS NULL OR account_target >= 0)
				)
			""")
			
			# Create account territory assignments table
			await connection.execute("""
				CREATE TABLE crm_account_territory_assignments (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					account_id TEXT NOT NULL,
					territory_id TEXT NOT NULL,
					assignment_type crm_assignment_type DEFAULT 'primary',
					
					-- Assignment details
					assigned_by TEXT NOT NULL,
					assignment_reason TEXT,
					effective_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expiry_date TIMESTAMP WITH TIME ZONE,
					
					-- Performance tracking
					assignment_score DECIMAL(3,2),
					
					-- Metadata
					notes TEXT,
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Foreign key constraints
					FOREIGN KEY (account_id) REFERENCES crm_accounts(id) ON DELETE CASCADE,
					FOREIGN KEY (territory_id) REFERENCES crm_territories(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_assignment_score CHECK (assignment_score IS NULL OR (assignment_score >= 0.0 AND assignment_score <= 1.0)),
					CONSTRAINT check_assignment_notes_length CHECK (char_length(notes) <= 1000),
					CONSTRAINT check_effective_before_expiry CHECK (expiry_date IS NULL OR effective_date <= expiry_date)
				)
			""")
			
			# Create indexes for territories table
			await connection.execute("""
				CREATE INDEX idx_crm_territories_tenant 
				ON crm_territories(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_name 
				ON crm_territories(territory_name)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_code 
				ON crm_territories(territory_code)
				WHERE territory_code IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_type 
				ON crm_territories(territory_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_status 
				ON crm_territories(status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_owner 
				ON crm_territories(owner_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_created_at 
				ON crm_territories(created_at)
			""")
			
			# Create GIN indexes for array and JSONB fields
			await connection.execute("""
				CREATE INDEX idx_crm_territories_sales_reps 
				ON crm_territories USING GIN (sales_rep_ids)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_countries 
				ON crm_territories USING GIN (countries)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_states 
				ON crm_territories USING GIN (states_provinces)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_cities 
				ON crm_territories USING GIN (cities)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_postal_codes 
				ON crm_territories USING GIN (postal_codes)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_industries 
				ON crm_territories USING GIN (industries)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_product_lines 
				ON crm_territories USING GIN (product_lines)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_rules 
				ON crm_territories USING GIN (rules)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_metadata 
				ON crm_territories USING GIN (metadata)
			""")
			
			# Create composite indexes for common queries
			await connection.execute("""
				CREATE INDEX idx_crm_territories_tenant_status 
				ON crm_territories(tenant_id, status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_tenant_type 
				ON crm_territories(tenant_id, territory_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territories_tenant_owner 
				ON crm_territories(tenant_id, owner_id)
			""")
			
			# Create unique constraint for territory names within tenant
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_territories_unique_name 
				ON crm_territories(tenant_id, territory_name)
			""")
			
			# Create indexes for assignments table
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_tenant 
				ON crm_account_territory_assignments(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_account 
				ON crm_account_territory_assignments(account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_territory 
				ON crm_account_territory_assignments(territory_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_type 
				ON crm_account_territory_assignments(assignment_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_effective_date 
				ON crm_account_territory_assignments(effective_date)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_expiry_date 
				ON crm_account_territory_assignments(expiry_date)
				WHERE expiry_date IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_score 
				ON crm_account_territory_assignments(assignment_score)
				WHERE assignment_score IS NOT NULL
			""")
			
			# Create GIN index for assignment metadata
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_metadata 
				ON crm_account_territory_assignments USING GIN (metadata)
			""")
			
			# Create composite indexes for common assignment queries
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_tenant_account 
				ON crm_account_territory_assignments(tenant_id, account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_tenant_territory 
				ON crm_account_territory_assignments(tenant_id, territory_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_territory_assignments_territory_type 
				ON crm_account_territory_assignments(territory_id, assignment_type)
			""")
			
			# Create unique constraint to prevent duplicate primary assignments
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_territory_assignments_unique_primary 
				ON crm_account_territory_assignments(tenant_id, account_id)
				WHERE assignment_type = 'primary'
			""")
			
			# Create triggers for updating updated_at timestamps
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_territories_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_territories_updated_at
					BEFORE UPDATE ON crm_territories
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_territories_updated_at()
			""")
			
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_territory_assignments_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_territory_assignments_updated_at
					BEFORE UPDATE ON crm_account_territory_assignments
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_territory_assignments_updated_at()
			""")
			
			# Create view for territory performance analytics
			await connection.execute("""
				CREATE VIEW crm_territory_performance AS
				SELECT 
					t.id as territory_id,
					t.tenant_id,
					t.territory_name,
					t.territory_type,
					t.status,
					t.owner_id,
					t.annual_quota,
					t.account_target,
					t.current_accounts,
					t.current_revenue,
					t.quota_achievement,
					
					-- Assignment statistics
					(SELECT COUNT(*) FROM crm_account_territory_assignments a 
					 WHERE a.territory_id = t.id AND a.assignment_type = 'primary') as primary_accounts,
					
					(SELECT COUNT(*) FROM crm_account_territory_assignments a 
					 WHERE a.territory_id = t.id AND a.assignment_type = 'secondary') as secondary_accounts,
					
					(SELECT COUNT(*) FROM crm_account_territory_assignments a 
					 WHERE a.territory_id = t.id AND a.assignment_type = 'overlay') as overlay_accounts,
					
					-- Performance metrics
					CASE 
						WHEN t.annual_quota > 0 THEN (t.current_revenue / t.annual_quota) * 100
						ELSE 0
					END as quota_percentage,
					
					CASE 
						WHEN t.account_target > 0 THEN (t.current_accounts::DECIMAL / t.account_target) * 100
						ELSE 0
					END as account_target_percentage,
					
					-- Average assignment score
					(SELECT AVG(a.assignment_score) FROM crm_account_territory_assignments a 
					 WHERE a.territory_id = t.id AND a.assignment_score IS NOT NULL) as avg_assignment_score
					
				FROM crm_territories t
				WHERE t.status = 'active'
			""")
			
			# Create function for territory coverage analysis
			await connection.execute("""
				CREATE OR REPLACE FUNCTION analyze_territory_coverage(
					territory_id_param TEXT,
					tenant_filter TEXT DEFAULT NULL
				)
				RETURNS TABLE(
					coverage_type TEXT,
					total_accounts BIGINT,
					covered_accounts BIGINT,
					coverage_percentage DECIMAL,
					uncovered_accounts BIGINT
				) AS $$
				BEGIN
					RETURN QUERY
					WITH territory_accounts AS (
						SELECT DISTINCT a.account_id
						FROM crm_account_territory_assignments a
						WHERE a.territory_id = territory_id_param
						AND (tenant_filter IS NULL OR a.tenant_id = tenant_filter)
					),
					all_accounts AS (
						SELECT acc.id, acc.account_type, acc.industry, acc.annual_revenue, acc.employee_count
						FROM crm_accounts acc
						WHERE (tenant_filter IS NULL OR acc.tenant_id = tenant_filter)
						AND acc.status = 'active'
					)
					SELECT 
						'overall'::TEXT as coverage_type,
						COUNT(*) as total_accounts,
						COUNT(ta.account_id) as covered_accounts,
						CASE 
							WHEN COUNT(*) > 0 THEN (COUNT(ta.account_id)::DECIMAL / COUNT(*)) * 100
							ELSE 0
						END as coverage_percentage,
						COUNT(*) - COUNT(ta.account_id) as uncovered_accounts
					FROM all_accounts aa
					LEFT JOIN territory_accounts ta ON aa.id = ta.account_id;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for territory assignment recommendations
			await connection.execute("""
				CREATE OR REPLACE FUNCTION get_territory_assignment_candidates(
					territory_id_param TEXT,
					tenant_filter TEXT DEFAULT NULL,
					limit_results INTEGER DEFAULT 10
				)
				RETURNS TABLE(
					account_id TEXT,
					account_name TEXT,
					account_type TEXT,
					match_score DECIMAL,
					match_reasons TEXT[]
				) AS $$
				BEGIN
					RETURN QUERY
					WITH territory_criteria AS (
						SELECT t.industries, t.company_size_min, t.company_size_max, 
							   t.revenue_min, t.revenue_max, t.countries, t.states_provinces, t.cities
						FROM crm_territories t
						WHERE t.id = territory_id_param
					),
					unassigned_accounts AS (
						SELECT acc.id, acc.account_name, acc.account_type, acc.industry, 
							   acc.employee_count, acc.annual_revenue
						FROM crm_accounts acc
						WHERE (tenant_filter IS NULL OR acc.tenant_id = tenant_filter)
						AND acc.status = 'active'
						AND NOT EXISTS (
							SELECT 1 FROM crm_account_territory_assignments a
							WHERE a.account_id = acc.id AND a.territory_id = territory_id_param
						)
					)
					SELECT 
						ua.id as account_id,
						ua.account_name,
						ua.account_type,
						COALESCE(
							-- Industry match (40% weight)
							CASE 
								WHEN tc.industries IS NOT NULL AND ua.industry = ANY(tc.industries) THEN 0.4
								ELSE 0.0
							END +
							-- Company size match (30% weight)
							CASE 
								WHEN (tc.company_size_min IS NULL OR ua.employee_count >= tc.company_size_min)
								 AND (tc.company_size_max IS NULL OR ua.employee_count <= tc.company_size_max)
								THEN 0.3
								ELSE 0.0
							END +
							-- Revenue match (30% weight)
							CASE 
								WHEN (tc.revenue_min IS NULL OR ua.annual_revenue >= tc.revenue_min)
								 AND (tc.revenue_max IS NULL OR ua.annual_revenue <= tc.revenue_max)
								THEN 0.3
								ELSE 0.0
							END,
							0.0
						) as match_score,
						ARRAY[]::TEXT[] as match_reasons  -- Simplified for now
					FROM unassigned_accounts ua
					CROSS JOIN territory_criteria tc
					ORDER BY match_score DESC
					LIMIT limit_results;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Territory management structures created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create territory management structures: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back territory management migration...")
			
			# Drop functions
			await connection.execute("DROP FUNCTION IF EXISTS get_territory_assignment_candidates CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS analyze_territory_coverage CASCADE")
			
			# Drop view
			await connection.execute("DROP VIEW IF EXISTS crm_territory_performance CASCADE")
			
			# Drop triggers and functions
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_territory_assignments_updated_at ON crm_account_territory_assignments")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_territory_assignments_updated_at CASCADE")
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_territories_updated_at ON crm_territories")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_territories_updated_at CASCADE")
			
			# Drop tables (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_account_territory_assignments CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_territories CASCADE")
			
			# Drop enums
			await connection.execute("DROP TYPE IF EXISTS crm_assignment_type CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_territory_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_territory_type CASCADE")
			
			logger.info("✅ Territory management migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback territory management migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if main tables exist
			tables_exist = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.tables 
				WHERE table_name IN ('crm_territories', 'crm_account_territory_assignments')
			""")
			
			if tables_exist != 2:
				return False
			
			# Check if enums exist
			enum_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_type 
				WHERE typname IN (
					'crm_territory_type',
					'crm_territory_status',
					'crm_assignment_type'
				)
			""")
			
			if enum_count != 3:
				return False
			
			# Check if view exists
			view_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.views 
					WHERE table_name = 'crm_territory_performance'
				)
			""")
			
			if not view_exists:
				return False
			
			# Check if functions exist
			function_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.routines 
				WHERE routine_name IN (
					'analyze_territory_coverage',
					'get_territory_assignment_candidates',
					'update_crm_territories_updated_at',
					'update_crm_territory_assignments_updated_at'
				)
			""")
			
			if function_count < 4:
				return False
			
			# Check key indexes
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename IN ('crm_territories', 'crm_account_territory_assignments')
				AND indexname IN (
					'idx_crm_territories_tenant',
					'idx_crm_territory_assignments_tenant',
					'idx_crm_territories_unique_name',
					'idx_crm_territory_assignments_unique_primary'
				)
			""")
			
			if index_count < 4:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False