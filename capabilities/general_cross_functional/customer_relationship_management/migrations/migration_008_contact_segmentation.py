"""
APG Customer Relationship Management - Contact Segmentation Migration

Database migration to create contact segmentation tables and supporting structures
for advanced contact segmentation and targeting capabilities.

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


class ContactSegmentationMigration(BaseMigration):
	"""Migration for contact segmentation functionality"""
	
	def _get_migration_id(self) -> str:
		return "008_contact_segmentation"
	
	def _get_version(self) -> str:
		return "008"
	
	def _get_description(self) -> str:
		return "Create contact segmentation tables and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating contact segmentation structures...")
			
			# Create segment type enum
			await connection.execute("""
				CREATE TYPE crm_segment_type AS ENUM (
					'static', 'dynamic', 'smart'
				)
			""")
			
			# Create segment status enum
			await connection.execute("""
				CREATE TYPE crm_segment_status AS ENUM (
					'active', 'inactive', 'archived'
				)
			""")
			
			# Create criteria operator enum
			await connection.execute("""
				CREATE TYPE crm_criteria_operator AS ENUM (
					'equals', 'not_equals', 'contains', 'not_contains',
					'starts_with', 'ends_with', 'greater_than', 'less_than',
					'greater_equal', 'less_equal', 'in', 'not_in',
					'is_null', 'is_not_null', 'between', 'not_between', 'regex'
				)
			""")
			
			# Create logical operator enum
			await connection.execute("""
				CREATE TYPE crm_logical_operator AS ENUM (
					'and', 'or', 'not'
				)
			""")
			
			# Create contact segments table
			await connection.execute("""
				CREATE TABLE crm_contact_segments (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Basic information
					name TEXT NOT NULL,
					description TEXT,
					segment_type crm_segment_type DEFAULT 'dynamic',
					status crm_segment_status DEFAULT 'active',
					
					-- Segmentation rules (stored as JSONB for flexibility)
					rules JSONB DEFAULT '[]',
					
					-- Static contact list (for static segments)
					contact_ids TEXT[] DEFAULT '{}',
					
					-- Metadata and settings
					auto_refresh BOOLEAN DEFAULT true,
					refresh_frequency_hours INTEGER DEFAULT 24,
					last_refreshed_at TIMESTAMP WITH TIME ZONE,
					
					-- Performance tracking
					contact_count INTEGER DEFAULT 0,
					estimated_count INTEGER,
					
					-- Usage tracking
					usage_count INTEGER DEFAULT 0,
					last_used_at TIMESTAMP WITH TIME ZONE,
					
					-- Categorization
					category TEXT,
					tags TEXT[] DEFAULT '{}',
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Constraints
					CONSTRAINT check_segment_name_length CHECK (char_length(name) >= 1 AND char_length(name) <= 200),
					CONSTRAINT check_segment_description_length CHECK (char_length(description) <= 2000),
					CONSTRAINT check_refresh_frequency_positive CHECK (refresh_frequency_hours > 0),
					CONSTRAINT check_contact_count_positive CHECK (contact_count >= 0),
					CONSTRAINT check_usage_count_positive CHECK (usage_count >= 0),
					CONSTRAINT check_category_length CHECK (char_length(category) <= 100)
				)
			""")
			
			# Create segment memberships table for tracking segment membership history
			await connection.execute("""
				CREATE TABLE crm_segment_memberships (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					segment_id TEXT NOT NULL,
					contact_id TEXT NOT NULL,
					
					-- Membership details
					joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					left_at TIMESTAMP WITH TIME ZONE,
					is_active BOOLEAN DEFAULT true,
					
					-- Membership source
					source TEXT DEFAULT 'system', -- system, manual, import, etc.
					source_details JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT DEFAULT 'system',
					updated_by TEXT DEFAULT 'system',
					
					-- Foreign key constraints
					FOREIGN KEY (segment_id) REFERENCES crm_contact_segments(id) ON DELETE CASCADE,
					FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_membership_dates CHECK (left_at IS NULL OR joined_at <= left_at)
				)
			""")
			
			# Create segment usage log table for tracking segment usage
			await connection.execute("""
				CREATE TABLE crm_segment_usage_log (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					segment_id TEXT NOT NULL,
					
					-- Usage details
					usage_type TEXT NOT NULL, -- 'view', 'export', 'campaign', 'email', etc.
					usage_context TEXT, -- Additional context about the usage
					contact_count INTEGER,
					
					-- User information
					used_by TEXT NOT NULL,
					used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Foreign key constraints
					FOREIGN KEY (segment_id) REFERENCES crm_contact_segments(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_usage_type_length CHECK (char_length(usage_type) >= 1 AND char_length(usage_type) <= 50),
					CONSTRAINT check_usage_context_length CHECK (char_length(usage_context) <= 500),
					CONSTRAINT check_usage_contact_count_positive CHECK (contact_count IS NULL OR contact_count >= 0)
				)
			""")
			
			# Create indexes for segments table
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_tenant 
				ON crm_contact_segments(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_name 
				ON crm_contact_segments(name)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_type 
				ON crm_contact_segments(segment_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_status 
				ON crm_contact_segments(status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_category 
				ON crm_contact_segments(category)
				WHERE category IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_created_at 
				ON crm_contact_segments(created_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_updated_at 
				ON crm_contact_segments(updated_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_last_used_at 
				ON crm_contact_segments(last_used_at)
				WHERE last_used_at IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_last_refreshed_at 
				ON crm_contact_segments(last_refreshed_at)
				WHERE last_refreshed_at IS NOT NULL
			""")
			
			# Create GIN indexes for array and JSONB fields
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_contact_ids 
				ON crm_contact_segments USING GIN (contact_ids)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_tags 
				ON crm_contact_segments USING GIN (tags)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_rules 
				ON crm_contact_segments USING GIN (rules)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_metadata 
				ON crm_contact_segments USING GIN (metadata)
			""")
			
			# Create composite indexes for common queries
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_tenant_type 
				ON crm_contact_segments(tenant_id, segment_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_tenant_status 
				ON crm_contact_segments(tenant_id, status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_contact_segments_tenant_category 
				ON crm_contact_segments(tenant_id, category)
				WHERE category IS NOT NULL
			""")
			
			# Create unique constraint for segment names within tenant
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_contact_segments_unique_name 
				ON crm_contact_segments(tenant_id, name)
				WHERE status != 'archived'
			""")
			
			# Create indexes for memberships table
			await connection.execute("""
				CREATE INDEX idx_crm_segment_memberships_tenant 
				ON crm_segment_memberships(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_memberships_segment 
				ON crm_segment_memberships(segment_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_memberships_contact 
				ON crm_segment_memberships(contact_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_memberships_active 
				ON crm_segment_memberships(is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_memberships_joined_at 
				ON crm_segment_memberships(joined_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_memberships_left_at 
				ON crm_segment_memberships(left_at)
				WHERE left_at IS NOT NULL
			""")
			
			# Create composite indexes for memberships
			await connection.execute("""
				CREATE INDEX idx_crm_segment_memberships_tenant_segment 
				ON crm_segment_memberships(tenant_id, segment_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_memberships_tenant_contact 
				ON crm_segment_memberships(tenant_id, contact_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_memberships_segment_active 
				ON crm_segment_memberships(segment_id, is_active)
			""")
			
			# Create unique constraint to prevent duplicate active memberships
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_segment_memberships_unique_active 
				ON crm_segment_memberships(tenant_id, segment_id, contact_id)
				WHERE is_active = true
			""")
			
			# Create indexes for usage log table
			await connection.execute("""
				CREATE INDEX idx_crm_segment_usage_log_tenant 
				ON crm_segment_usage_log(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_usage_log_segment 
				ON crm_segment_usage_log(segment_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_usage_log_type 
				ON crm_segment_usage_log(usage_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_usage_log_used_by 
				ON crm_segment_usage_log(used_by)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_usage_log_used_at 
				ON crm_segment_usage_log(used_at)
			""")
			
			# Create composite indexes for usage log
			await connection.execute("""
				CREATE INDEX idx_crm_segment_usage_log_tenant_segment 
				ON crm_segment_usage_log(tenant_id, segment_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_segment_usage_log_segment_type 
				ON crm_segment_usage_log(segment_id, usage_type)
			""")
			
			# Create triggers for updating updated_at timestamps
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_contact_segments_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_contact_segments_updated_at
					BEFORE UPDATE ON crm_contact_segments
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_contact_segments_updated_at()
			""")
			
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_segment_memberships_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_segment_memberships_updated_at
					BEFORE UPDATE ON crm_segment_memberships
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_segment_memberships_updated_at()
			""")
			
			# Create view for segment analytics
			await connection.execute("""
				CREATE VIEW crm_segment_analytics AS
				SELECT 
					s.id as segment_id,
					s.tenant_id,
					s.name as segment_name,
					s.segment_type,
					s.status,
					s.contact_count,
					s.usage_count,
					s.last_used_at,
					s.last_refreshed_at,
					s.created_at,
					
					-- Membership statistics
					(SELECT COUNT(*) FROM crm_segment_memberships m 
					 WHERE m.segment_id = s.id AND m.is_active = true) as active_memberships,
					
					(SELECT COUNT(*) FROM crm_segment_memberships m 
					 WHERE m.segment_id = s.id AND m.is_active = false) as inactive_memberships,
					
					-- Recent activity
					(SELECT COUNT(*) FROM crm_segment_usage_log u 
					 WHERE u.segment_id = s.id AND u.used_at >= NOW() - INTERVAL '30 days') as usage_last_30_days,
					
					(SELECT COUNT(*) FROM crm_segment_usage_log u 
					 WHERE u.segment_id = s.id AND u.used_at >= NOW() - INTERVAL '7 days') as usage_last_7_days,
					
					-- Growth metrics
					(SELECT COUNT(*) FROM crm_segment_memberships m 
					 WHERE m.segment_id = s.id AND m.joined_at >= NOW() - INTERVAL '30 days') as contacts_added_last_30_days,
					
					(SELECT COUNT(*) FROM crm_segment_memberships m 
					 WHERE m.segment_id = s.id AND m.left_at >= NOW() - INTERVAL '30 days') as contacts_removed_last_30_days
					
				FROM crm_contact_segments s
			""")
			
			# Create function for segment refresh recommendations
			await connection.execute("""
				CREATE OR REPLACE FUNCTION get_segments_needing_refresh(
					tenant_filter TEXT DEFAULT NULL,
					hours_threshold INTEGER DEFAULT 24
				)
				RETURNS TABLE(
					segment_id TEXT,
					segment_name TEXT,
					last_refreshed_at TIMESTAMP WITH TIME ZONE,
					hours_since_refresh INTERVAL,
					refresh_frequency_hours INTEGER
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						s.id as segment_id,
						s.name as segment_name,
						s.last_refreshed_at,
						CASE 
							WHEN s.last_refreshed_at IS NOT NULL THEN NOW() - s.last_refreshed_at
							ELSE NULL
						END as hours_since_refresh,
						s.refresh_frequency_hours
					FROM crm_contact_segments s
					WHERE s.segment_type = 'dynamic'
					AND s.status = 'active'
					AND s.auto_refresh = true
					AND (tenant_filter IS NULL OR s.tenant_id = tenant_filter)
					AND (
						s.last_refreshed_at IS NULL OR 
						s.last_refreshed_at < NOW() - (s.refresh_frequency_hours || ' hours')::INTERVAL
					)
					ORDER BY 
						CASE WHEN s.last_refreshed_at IS NULL THEN 1 ELSE 0 END,
						s.last_refreshed_at ASC;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for segment overlap analysis
			await connection.execute("""
				CREATE OR REPLACE FUNCTION analyze_segment_overlap(
					segment_id_1 TEXT,
					segment_id_2 TEXT,
					tenant_filter TEXT DEFAULT NULL
				)
				RETURNS TABLE(
					segment_1_contacts BIGINT,
					segment_2_contacts BIGINT,
					overlapping_contacts BIGINT,
					overlap_percentage DECIMAL
				) AS $$
				BEGIN
					RETURN QUERY
					WITH segment_1_members AS (
						SELECT DISTINCT m.contact_id
						FROM crm_segment_memberships m
						WHERE m.segment_id = segment_id_1
						AND m.is_active = true
						AND (tenant_filter IS NULL OR m.tenant_id = tenant_filter)
					),
					segment_2_members AS (
						SELECT DISTINCT m.contact_id
						FROM crm_segment_memberships m
						WHERE m.segment_id = segment_id_2
						AND m.is_active = true
						AND (tenant_filter IS NULL OR m.tenant_id = tenant_filter)
					),
					overlap AS (
						SELECT s1.contact_id
						FROM segment_1_members s1
						INNER JOIN segment_2_members s2 ON s1.contact_id = s2.contact_id
					)
					SELECT 
						(SELECT COUNT(*) FROM segment_1_members) as segment_1_contacts,
						(SELECT COUNT(*) FROM segment_2_members) as segment_2_contacts,
						(SELECT COUNT(*) FROM overlap) as overlapping_contacts,
						CASE 
							WHEN (SELECT COUNT(*) FROM segment_1_members) > 0 THEN
								((SELECT COUNT(*) FROM overlap)::DECIMAL / (SELECT COUNT(*) FROM segment_1_members)) * 100
							ELSE 0
						END as overlap_percentage;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Contact segmentation structures created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create contact segmentation structures: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back contact segmentation migration...")
			
			# Drop functions
			await connection.execute("DROP FUNCTION IF EXISTS analyze_segment_overlap CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS get_segments_needing_refresh CASCADE")
			
			# Drop view
			await connection.execute("DROP VIEW IF EXISTS crm_segment_analytics CASCADE")
			
			# Drop triggers and functions
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_segment_memberships_updated_at ON crm_segment_memberships")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_segment_memberships_updated_at CASCADE")
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_contact_segments_updated_at ON crm_contact_segments")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_contact_segments_updated_at CASCADE")
			
			# Drop tables (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_segment_usage_log CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_segment_memberships CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_contact_segments CASCADE")
			
			# Drop enums
			await connection.execute("DROP TYPE IF EXISTS crm_logical_operator CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_criteria_operator CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_segment_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_segment_type CASCADE")
			
			logger.info("✅ Contact segmentation migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback contact segmentation migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if main tables exist
			tables_exist = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.tables 
				WHERE table_name IN ('crm_contact_segments', 'crm_segment_memberships', 'crm_segment_usage_log')
			""")
			
			if tables_exist != 3:
				return False
			
			# Check if enums exist
			enum_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_type 
				WHERE typname IN (
					'crm_segment_type',
					'crm_segment_status',
					'crm_criteria_operator',
					'crm_logical_operator'
				)
			""")
			
			if enum_count != 4:
				return False
			
			# Check if view exists
			view_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.views 
					WHERE table_name = 'crm_segment_analytics'
				)
			""")
			
			if not view_exists:
				return False
			
			# Check if functions exist
			function_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.routines 
				WHERE routine_name IN (
					'get_segments_needing_refresh',
					'analyze_segment_overlap',
					'update_crm_contact_segments_updated_at',
					'update_crm_segment_memberships_updated_at'
				)
			""")
			
			if function_count < 4:
				return False
			
			# Check key indexes
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename IN ('crm_contact_segments', 'crm_segment_memberships', 'crm_segment_usage_log')
				AND indexname IN (
					'idx_crm_contact_segments_tenant',
					'idx_crm_segment_memberships_tenant',
					'idx_crm_segment_usage_log_tenant',
					'idx_crm_contact_segments_unique_name'
				)
			""")
			
			if index_count < 4:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False