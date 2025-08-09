"""
APG Customer Relationship Management - Account Relationships Migration

Database migration to create account relationships table and supporting structures
for advanced business relationship management and partner network tracking.

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


class AccountRelationshipsMigration(BaseMigration):
	"""Migration for account relationships functionality"""
	
	def _get_migration_id(self) -> str:
		return "005_account_relationships"
	
	def _get_version(self) -> str:
		return "005"
	
	def _get_description(self) -> str:
		return "Create account relationships table and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema", "004_account_hierarchy"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating account relationships structures...")
			
			# Create account relationship type enum
			await connection.execute("""
				CREATE TYPE crm_account_relationship_type AS ENUM (
					'customer', 'vendor', 'partner', 'competitor', 'subsidiary',
					'parent_company', 'joint_venture', 'strategic_alliance',
					'reseller', 'distributor', 'supplier', 'service_provider',
					'integration_partner', 'referral_source', 'acquisition_target',
					'investor', 'board_member', 'consultant', 'legal_counsel', 'other'
				)
			""")
			
			# Create relationship strength enum
			await connection.execute("""
				CREATE TYPE crm_relationship_strength AS ENUM (
					'weak', 'moderate', 'strong', 'strategic'
				)
			""")
			
			# Create relationship status enum
			await connection.execute("""
				CREATE TYPE crm_relationship_status AS ENUM (
					'active', 'inactive', 'pending', 'terminated', 'suspended'
				)
			""")
			
			# Create relationship direction enum
			await connection.execute("""
				CREATE TYPE crm_relationship_direction AS ENUM (
					'outbound', 'inbound', 'bidirectional'
				)
			""")
			
			# Create account relationships table
			await connection.execute("""
				CREATE TABLE crm_account_relationships (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					from_account_id TEXT NOT NULL,
					to_account_id TEXT NOT NULL,
					relationship_type crm_account_relationship_type NOT NULL,
					relationship_strength crm_relationship_strength DEFAULT 'moderate',
					relationship_status crm_relationship_status DEFAULT 'active',
					direction crm_relationship_direction DEFAULT 'outbound',
					
					-- Financial aspects
					annual_value DECIMAL(15,2),
					contract_start_date TIMESTAMP WITH TIME ZONE,
					contract_end_date TIMESTAMP WITH TIME ZONE,
					renewal_date TIMESTAMP WITH TIME ZONE,
					
					-- Relationship details
					key_contact_id TEXT,
					relationship_owner_id TEXT NOT NULL,
					
					-- Risk and compliance
					risk_level TEXT,
					compliance_status TEXT,
					
					-- Metadata
					description TEXT,
					notes TEXT,
					tags TEXT[] DEFAULT '{}',
					metadata JSONB DEFAULT '{}',
					
					-- Source tracking
					source TEXT,
					verified_at TIMESTAMP WITH TIME ZONE,
					verified_by TEXT,
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Foreign key constraints
					FOREIGN KEY (from_account_id) REFERENCES crm_accounts(id) ON DELETE CASCADE,
					FOREIGN KEY (to_account_id) REFERENCES crm_accounts(id) ON DELETE CASCADE,
					FOREIGN KEY (key_contact_id) REFERENCES crm_contacts(id) ON DELETE SET NULL,
					
					-- Constraints
					CONSTRAINT check_different_accounts CHECK (from_account_id != to_account_id),
					CONSTRAINT check_description_length CHECK (char_length(description) <= 2000),
					CONSTRAINT check_notes_length CHECK (char_length(notes) <= 2000),
					CONSTRAINT check_annual_value CHECK (annual_value >= 0)
				)
			""")
			
			# Create indexes for performance
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_tenant 
				ON crm_account_relationships(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_from_account 
				ON crm_account_relationships(from_account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_to_account 
				ON crm_account_relationships(to_account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_type 
				ON crm_account_relationships(relationship_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_strength 
				ON crm_account_relationships(relationship_strength)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_status 
				ON crm_account_relationships(relationship_status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_direction 
				ON crm_account_relationships(direction)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_created_at 
				ON crm_account_relationships(created_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_annual_value 
				ON crm_account_relationships(annual_value)
				WHERE annual_value IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_contract_end 
				ON crm_account_relationships(contract_end_date)
				WHERE contract_end_date IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_renewal_date 
				ON crm_account_relationships(renewal_date)
				WHERE renewal_date IS NOT NULL
			""")
			
			# Create composite indexes for common queries
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_tenant_from 
				ON crm_account_relationships(tenant_id, from_account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_tenant_to 
				ON crm_account_relationships(tenant_id, to_account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_accounts 
				ON crm_account_relationships(from_account_id, to_account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_type_status 
				ON crm_account_relationships(relationship_type, relationship_status)
			""")
			
			# Create unique constraint to prevent duplicate relationships
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_account_relationships_unique 
				ON crm_account_relationships(
					tenant_id, from_account_id, to_account_id, relationship_type
				)
			""")
			
			# Create GIN indexes for metadata and tags
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_metadata 
				ON crm_account_relationships USING GIN (metadata)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_account_relationships_tags 
				ON crm_account_relationships USING GIN (tags)
			""")
			
			# Create trigger for updating updated_at timestamp
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_account_relationships_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_account_relationships_updated_at
					BEFORE UPDATE ON crm_account_relationships
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_account_relationships_updated_at()
			""")
			
			# Create view for relationship analytics
			await connection.execute("""
				CREATE VIEW crm_account_relationship_analytics AS
				SELECT 
					a.id as account_id,
					a.tenant_id,
					a.account_name,
					
					-- Outgoing relationships
					(SELECT COUNT(*) FROM crm_account_relationships r 
					 WHERE r.from_account_id = a.id AND r.tenant_id = a.tenant_id 
					 AND r.relationship_status = 'active') as outgoing_relationships,
					
					-- Incoming relationships
					(SELECT COUNT(*) FROM crm_account_relationships r 
					 WHERE r.to_account_id = a.id AND r.tenant_id = a.tenant_id 
					 AND r.relationship_status = 'active') as incoming_relationships,
					
					-- Total relationships
					(SELECT COUNT(*) FROM crm_account_relationships r 
					 WHERE (r.from_account_id = a.id OR r.to_account_id = a.id) 
					 AND r.tenant_id = a.tenant_id 
					 AND r.relationship_status = 'active') as total_relationships,
					
					-- Customer relationships
					(SELECT COUNT(*) FROM crm_account_relationships r 
					 WHERE r.from_account_id = a.id AND r.tenant_id = a.tenant_id 
					 AND r.relationship_type = 'customer' 
					 AND r.relationship_status = 'active') as customer_relationships,
					
					-- Vendor relationships
					(SELECT COUNT(*) FROM crm_account_relationships r 
					 WHERE r.from_account_id = a.id AND r.tenant_id = a.tenant_id 
					 AND r.relationship_type = 'vendor' 
					 AND r.relationship_status = 'active') as vendor_relationships,
					
					-- Partner relationships
					(SELECT COUNT(*) FROM crm_account_relationships r 
					 WHERE (r.from_account_id = a.id OR r.to_account_id = a.id) 
					 AND r.tenant_id = a.tenant_id 
					 AND r.relationship_type = 'partner' 
					 AND r.relationship_status = 'active') as partner_relationships,
					
					-- Total annual value (outgoing)
					(SELECT COALESCE(SUM(r.annual_value), 0) FROM crm_account_relationships r 
					 WHERE r.from_account_id = a.id AND r.tenant_id = a.tenant_id 
					 AND r.relationship_status = 'active' 
					 AND r.annual_value IS NOT NULL) as total_relationship_value,
					
					-- Strategic relationships
					(SELECT COUNT(*) FROM crm_account_relationships r 
					 WHERE (r.from_account_id = a.id OR r.to_account_id = a.id) 
					 AND r.tenant_id = a.tenant_id 
					 AND r.relationship_strength = 'strategic' 
					 AND r.relationship_status = 'active') as strategic_relationships,
					
					-- Contracts expiring soon (next 90 days)
					(SELECT COUNT(*) FROM crm_account_relationships r 
					 WHERE (r.from_account_id = a.id OR r.to_account_id = a.id) 
					 AND r.tenant_id = a.tenant_id 
					 AND r.contract_end_date IS NOT NULL
					 AND r.contract_end_date BETWEEN NOW() AND NOW() + INTERVAL '90 days'
					 AND r.relationship_status = 'active') as expiring_contracts
					
				FROM crm_accounts a
				WHERE a.status = 'active'
			""")
			
			# Create function for relationship network analysis
			await connection.execute("""
				CREATE OR REPLACE FUNCTION get_relationship_network(
					center_account_id TEXT,
					tenant_filter TEXT DEFAULT NULL,
					max_depth INTEGER DEFAULT 2,
					relationship_types TEXT[] DEFAULT NULL
				)
				RETURNS TABLE(
					account_id TEXT,
					account_name TEXT,
					relationship_type TEXT,
					relationship_strength TEXT,
					depth INTEGER,
					path TEXT[]
				) AS $$
				BEGIN
					RETURN QUERY
					WITH RECURSIVE relationship_network AS (
						-- Base case: start with center account
						SELECT 
							a.id as account_id,
							a.account_name,
							CAST(NULL AS TEXT) as relationship_type,
							CAST(NULL AS TEXT) as relationship_strength,
							0 as depth,
							ARRAY[a.id] as path
						FROM crm_accounts a
						WHERE a.id = center_account_id
						AND (tenant_filter IS NULL OR a.tenant_id = tenant_filter)
						
						UNION ALL
						
						-- Recursive case: get connected accounts
						SELECT 
							CASE 
								WHEN r.from_account_id = rn.account_id THEN a_to.id
								ELSE a_from.id
							END as account_id,
							CASE 
								WHEN r.from_account_id = rn.account_id THEN a_to.account_name
								ELSE a_from.account_name
							END as account_name,
							r.relationship_type::TEXT,
							r.relationship_strength::TEXT,
							rn.depth + 1,
							CASE 
								WHEN r.from_account_id = rn.account_id THEN rn.path || a_to.id
								ELSE rn.path || a_from.id
							END as path
						FROM crm_account_relationships r
						JOIN relationship_network rn ON (
							r.from_account_id = rn.account_id OR r.to_account_id = rn.account_id
						)
						JOIN crm_accounts a_from ON r.from_account_id = a_from.id
						JOIN crm_accounts a_to ON r.to_account_id = a_to.id
						WHERE rn.depth < max_depth
						AND r.relationship_status = 'active'
						AND (relationship_types IS NULL OR r.relationship_type = ANY(relationship_types))
						AND (
							CASE 
								WHEN r.from_account_id = rn.account_id THEN a_to.id
								ELSE a_from.id
							END
						) != ALL(rn.path) -- Prevent cycles
					)
					SELECT * FROM relationship_network
					WHERE depth > 0  -- Exclude the center account
					ORDER BY depth, account_name;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for relationship strength analysis
			await connection.execute("""
				CREATE OR REPLACE FUNCTION analyze_relationship_strength(
					account_id_param TEXT,
					tenant_filter TEXT DEFAULT NULL
				)
				RETURNS TABLE(
					relationship_type TEXT,
					total_count BIGINT,
					weak_count BIGINT,
					moderate_count BIGINT,
					strong_count BIGINT,
					strategic_count BIGINT,
					total_value DECIMAL,
					avg_value DECIMAL
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						r.relationship_type::TEXT,
						COUNT(*) as total_count,
						COUNT(*) FILTER (WHERE r.relationship_strength = 'weak') as weak_count,
						COUNT(*) FILTER (WHERE r.relationship_strength = 'moderate') as moderate_count,
						COUNT(*) FILTER (WHERE r.relationship_strength = 'strong') as strong_count,
						COUNT(*) FILTER (WHERE r.relationship_strength = 'strategic') as strategic_count,
						COALESCE(SUM(r.annual_value), 0) as total_value,
						COALESCE(AVG(r.annual_value), 0) as avg_value
					FROM crm_account_relationships r
					WHERE (r.from_account_id = account_id_param OR r.to_account_id = account_id_param)
					AND (tenant_filter IS NULL OR r.tenant_id = tenant_filter)
					AND r.relationship_status = 'active'
					GROUP BY r.relationship_type
					ORDER BY total_count DESC;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Account relationships structures created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create account relationships structures: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back account relationships migration...")
			
			# Drop functions
			await connection.execute("DROP FUNCTION IF EXISTS analyze_relationship_strength CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS get_relationship_network CASCADE")
			
			# Drop view
			await connection.execute("DROP VIEW IF EXISTS crm_account_relationship_analytics CASCADE")
			
			# Drop trigger and function
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_account_relationships_updated_at ON crm_account_relationships")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_account_relationships_updated_at CASCADE")
			
			# Drop table (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_account_relationships CASCADE")
			
			# Drop enums
			await connection.execute("DROP TYPE IF EXISTS crm_relationship_direction CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_relationship_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_relationship_strength CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_account_relationship_type CASCADE")
			
			logger.info("✅ Account relationships migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback account relationships migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if table exists
			table_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.tables 
					WHERE table_name = 'crm_account_relationships'
				)
			""")
			
			if not table_exists:
				return False
			
			# Check if enums exist
			enum_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_type 
				WHERE typname IN (
					'crm_account_relationship_type',
					'crm_relationship_strength',
					'crm_relationship_status',
					'crm_relationship_direction'
				)
			""")
			
			if enum_count != 4:
				return False
			
			# Check if view exists
			view_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.views 
					WHERE table_name = 'crm_account_relationship_analytics'
				)
			""")
			
			if not view_exists:
				return False
			
			# Check if functions exist
			function_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.routines 
				WHERE routine_name IN (
					'get_relationship_network',
					'analyze_relationship_strength',
					'update_crm_account_relationships_updated_at'
				)
			""")
			
			if function_count < 3:
				return False
			
			# Check key indexes
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename = 'crm_account_relationships'
				AND indexname IN (
					'idx_crm_account_relationships_tenant',
					'idx_crm_account_relationships_from_account',
					'idx_crm_account_relationships_unique'
				)
			""")
			
			if index_count < 3:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False