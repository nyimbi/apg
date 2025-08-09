"""
APG Customer Relationship Management - Communication History Migration

Database migration to create communication tracking tables and supporting structures
for comprehensive interaction history management.

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


class CommunicationHistoryMigration(BaseMigration):
	"""Migration for communication history functionality"""
	
	def _get_migration_id(self) -> str:
		return "007_communication_history"
	
	def _get_version(self) -> str:
		return "007"
	
	def _get_description(self) -> str:
		return "Create communication history tables and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating communication history structures...")
			
			# Create communication type enum
			await connection.execute("""
				CREATE TYPE crm_communication_type AS ENUM (
					'email', 'phone_call', 'video_call', 'meeting', 'sms', 'chat',
					'social_media', 'letter', 'fax', 'website_visit', 'webinar', 'event', 'other'
				)
			""")
			
			# Create communication direction enum
			await connection.execute("""
				CREATE TYPE crm_communication_direction AS ENUM (
					'inbound', 'outbound', 'internal'
				)
			""")
			
			# Create communication status enum
			await connection.execute("""
				CREATE TYPE crm_communication_status AS ENUM (
					'scheduled', 'completed', 'cancelled', 'no_show', 'rescheduled'
				)
			""")
			
			# Create communication priority enum
			await connection.execute("""
				CREATE TYPE crm_communication_priority AS ENUM (
					'low', 'normal', 'high', 'urgent'
				)
			""")
			
			# Create communication outcome enum
			await connection.execute("""
				CREATE TYPE crm_communication_outcome AS ENUM (
					'successful', 'unsuccessful', 'follow_up_required', 'meeting_scheduled',
					'proposal_requested', 'decision_pending', 'closed_won', 'closed_lost', 'no_interest'
				)
			""")
			
			# Create communications table
			await connection.execute("""
				CREATE TABLE crm_communications (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Related entities
					contact_id TEXT,
					account_id TEXT,
					lead_id TEXT,
					opportunity_id TEXT,
					
					-- Communication details
					communication_type crm_communication_type NOT NULL,
					direction crm_communication_direction NOT NULL,
					status crm_communication_status DEFAULT 'completed',
					priority crm_communication_priority DEFAULT 'normal',
					
					-- Content
					subject TEXT NOT NULL,
					content TEXT,
					summary TEXT,
					
					-- Participants
					from_address TEXT,
					to_addresses TEXT[] DEFAULT '{}',
					cc_addresses TEXT[] DEFAULT '{}',
					bcc_addresses TEXT[] DEFAULT '{}',
					participants TEXT[] DEFAULT '{}',
					
					-- Timing
					scheduled_at TIMESTAMP WITH TIME ZONE,
					started_at TIMESTAMP WITH TIME ZONE,
					ended_at TIMESTAMP WITH TIME ZONE,
					duration_minutes INTEGER,
					
					-- Outcome and follow-up
					outcome crm_communication_outcome,
					outcome_notes TEXT,
					follow_up_required BOOLEAN DEFAULT false,
					follow_up_date TIMESTAMP WITH TIME ZONE,
					follow_up_notes TEXT,
					
					-- Attachments and references
					attachments JSONB DEFAULT '[]',
					external_id TEXT,
					external_source TEXT,
					
					-- Metadata
					tags TEXT[] DEFAULT '{}',
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Foreign key constraints
					FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE SET NULL,
					FOREIGN KEY (account_id) REFERENCES crm_accounts(id) ON DELETE SET NULL,
					
					-- Constraints
					CONSTRAINT check_subject_length CHECK (char_length(subject) >= 1 AND char_length(subject) <= 500),
					CONSTRAINT check_content_length CHECK (char_length(content) <= 10000),
					CONSTRAINT check_summary_length CHECK (char_length(summary) <= 2000),
					CONSTRAINT check_outcome_notes_length CHECK (char_length(outcome_notes) <= 2000),
					CONSTRAINT check_follow_up_notes_length CHECK (char_length(follow_up_notes) <= 2000),
					CONSTRAINT check_duration_positive CHECK (duration_minutes >= 0),
					CONSTRAINT check_timing_order CHECK (
						(started_at IS NULL OR ended_at IS NULL OR started_at <= ended_at) AND
						(scheduled_at IS NULL OR started_at IS NULL OR scheduled_at <= started_at)
					),
					CONSTRAINT check_has_related_entity CHECK (
						contact_id IS NOT NULL OR account_id IS NOT NULL OR 
						lead_id IS NOT NULL OR opportunity_id IS NOT NULL
					)
				)
			""")
			
			# Create communication templates table
			await connection.execute("""
				CREATE TABLE crm_communication_templates (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Template details
					name TEXT NOT NULL,
					description TEXT,
					communication_type crm_communication_type NOT NULL,
					
					-- Template content
					subject_template TEXT NOT NULL,
					content_template TEXT NOT NULL,
					
					-- Template variables
					variables TEXT[] DEFAULT '{}',
					
					-- Usage tracking
					usage_count INTEGER DEFAULT 0,
					last_used_at TIMESTAMP WITH TIME ZONE,
					
					-- Categorization
					category TEXT,
					tags TEXT[] DEFAULT '{}',
					
					-- Status
					is_active BOOLEAN DEFAULT true,
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					
					-- Constraints
					CONSTRAINT check_template_name_length CHECK (char_length(name) >= 1 AND char_length(name) <= 200),
					CONSTRAINT check_template_description_length CHECK (char_length(description) <= 1000),
					CONSTRAINT check_subject_template_length CHECK (char_length(subject_template) >= 1 AND char_length(subject_template) <= 500),
					CONSTRAINT check_content_template_length CHECK (char_length(content_template) >= 1),
					CONSTRAINT check_usage_count_positive CHECK (usage_count >= 0)
				)
			""")
			
			# Create indexes for communications table
			await connection.execute("""
				CREATE INDEX idx_crm_communications_tenant 
				ON crm_communications(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_contact 
				ON crm_communications(contact_id)
				WHERE contact_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_account 
				ON crm_communications(account_id)
				WHERE account_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_lead 
				ON crm_communications(lead_id)
				WHERE lead_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_opportunity 
				ON crm_communications(opportunity_id)
				WHERE opportunity_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_type 
				ON crm_communications(communication_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_direction 
				ON crm_communications(direction)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_status 
				ON crm_communications(status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_priority 
				ON crm_communications(priority)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_outcome 
				ON crm_communications(outcome)
				WHERE outcome IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_created_at 
				ON crm_communications(created_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_scheduled_at 
				ON crm_communications(scheduled_at)
				WHERE scheduled_at IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_follow_up_date 
				ON crm_communications(follow_up_date)
				WHERE follow_up_required = true AND follow_up_date IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_external_id 
				ON crm_communications(external_id, external_source)
				WHERE external_id IS NOT NULL
			""")
			
			# Create GIN indexes for array and JSONB fields
			await connection.execute("""
				CREATE INDEX idx_crm_communications_tags 
				ON crm_communications USING GIN (tags)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_to_addresses 
				ON crm_communications USING GIN (to_addresses)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_participants 
				ON crm_communications USING GIN (participants)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_metadata 
				ON crm_communications USING GIN (metadata)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_attachments 
				ON crm_communications USING GIN (attachments)
			""")
			
			# Create composite indexes for common queries
			await connection.execute("""
				CREATE INDEX idx_crm_communications_tenant_contact 
				ON crm_communications(tenant_id, contact_id)
				WHERE contact_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_tenant_account 
				ON crm_communications(tenant_id, account_id)
				WHERE account_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_tenant_type 
				ON crm_communications(tenant_id, communication_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_tenant_direction 
				ON crm_communications(tenant_id, direction)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_tenant_created_at 
				ON crm_communications(tenant_id, created_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communications_follow_up_tenant 
				ON crm_communications(tenant_id, follow_up_required, follow_up_date)
				WHERE follow_up_required = true
			""")
			
			# Create indexes for templates table
			await connection.execute("""
				CREATE INDEX idx_crm_communication_templates_tenant 
				ON crm_communication_templates(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communication_templates_type 
				ON crm_communication_templates(communication_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communication_templates_category 
				ON crm_communication_templates(category)
				WHERE category IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communication_templates_active 
				ON crm_communication_templates(is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_communication_templates_usage 
				ON crm_communication_templates(usage_count DESC)
			""")
			
			# Create GIN index for template tags
			await connection.execute("""
				CREATE INDEX idx_crm_communication_templates_tags 
				ON crm_communication_templates USING GIN (tags)
			""")
			
			# Create unique constraint for template names within tenant
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_communication_templates_unique_name 
				ON crm_communication_templates(tenant_id, name)
				WHERE is_active = true
			""")
			
			# Create triggers for updating updated_at timestamps
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_communications_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_communications_updated_at
					BEFORE UPDATE ON crm_communications
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_communications_updated_at()
			""")
			
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_communication_templates_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_communication_templates_updated_at
					BEFORE UPDATE ON crm_communication_templates
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_communication_templates_updated_at()
			""")
			
			# Create view for communication analytics
			await connection.execute("""
				CREATE VIEW crm_communication_analytics AS
				SELECT 
					c.tenant_id,
					c.contact_id,
					c.account_id,
					
					-- Total communications
					COUNT(*) as total_communications,
					
					-- Communications by type
					COUNT(*) FILTER (WHERE c.communication_type = 'email') as email_count,
					COUNT(*) FILTER (WHERE c.communication_type = 'phone_call') as phone_call_count,
					COUNT(*) FILTER (WHERE c.communication_type = 'meeting') as meeting_count,
					COUNT(*) FILTER (WHERE c.communication_type = 'video_call') as video_call_count,
					
					-- Communications by direction
					COUNT(*) FILTER (WHERE c.direction = 'inbound') as inbound_count,
					COUNT(*) FILTER (WHERE c.direction = 'outbound') as outbound_count,
					
					-- Communications by outcome
					COUNT(*) FILTER (WHERE c.outcome = 'successful') as successful_count,
					COUNT(*) FILTER (WHERE c.outcome = 'follow_up_required') as follow_up_required_count,
					
					-- Timing statistics
					AVG(c.duration_minutes) FILTER (WHERE c.duration_minutes IS NOT NULL) as avg_duration_minutes,
					SUM(c.duration_minutes) FILTER (WHERE c.duration_minutes IS NOT NULL) as total_duration_minutes,
					
					-- Follow-up statistics
					COUNT(*) FILTER (WHERE c.follow_up_required = true) as follow_ups_required,
					COUNT(*) FILTER (WHERE c.follow_up_required = true AND c.follow_up_date <= NOW()) as overdue_follow_ups,
					
					-- Recent activity
					MAX(c.created_at) as last_communication_at,
					COUNT(*) FILTER (WHERE c.created_at >= NOW() - INTERVAL '30 days') as communications_last_30_days,
					COUNT(*) FILTER (WHERE c.created_at >= NOW() - INTERVAL '7 days') as communications_last_7_days
					
				FROM crm_communications c
				GROUP BY c.tenant_id, c.contact_id, c.account_id
			""")
			
			# Create function for communication search with full-text search
			await connection.execute("""
				CREATE OR REPLACE FUNCTION search_communications(
					tenant_filter TEXT,
					search_term TEXT,
					limit_results INTEGER DEFAULT 50
				)
				RETURNS TABLE(
					id TEXT,
					subject TEXT,
					content TEXT,
					communication_type TEXT,
					created_at TIMESTAMP WITH TIME ZONE,
					contact_id TEXT,
					account_id TEXT,
					search_rank REAL
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						c.id,
						c.subject,
						c.content,
						c.communication_type::TEXT,
						c.created_at,
						c.contact_id,
						c.account_id,
						ts_rank(
							to_tsvector('english', COALESCE(c.subject, '') || ' ' || COALESCE(c.content, '') || ' ' || COALESCE(c.summary, '')),
							plainto_tsquery('english', search_term)
						) as search_rank
					FROM crm_communications c
					WHERE c.tenant_id = tenant_filter
					AND (
						to_tsvector('english', COALESCE(c.subject, '') || ' ' || COALESCE(c.content, '') || ' ' || COALESCE(c.summary, ''))
						@@ plainto_tsquery('english', search_term)
					)
					ORDER BY search_rank DESC, c.created_at DESC
					LIMIT limit_results;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for getting communication timeline
			await connection.execute("""
				CREATE OR REPLACE FUNCTION get_communication_timeline(
					tenant_filter TEXT,
					entity_id TEXT,
					entity_type TEXT DEFAULT 'contact',
					limit_results INTEGER DEFAULT 20
				)
				RETURNS TABLE(
					id TEXT,
					communication_type TEXT,
					direction TEXT,
					subject TEXT,
					created_at TIMESTAMP WITH TIME ZONE,
					outcome TEXT,
					follow_up_required BOOLEAN
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						c.id,
						c.communication_type::TEXT,
						c.direction::TEXT,
						c.subject,
						c.created_at,
						c.outcome::TEXT,
						c.follow_up_required
					FROM crm_communications c
					WHERE c.tenant_id = tenant_filter
					AND (
						(entity_type = 'contact' AND c.contact_id = entity_id) OR
						(entity_type = 'account' AND c.account_id = entity_id) OR
						(entity_type = 'lead' AND c.lead_id = entity_id) OR
						(entity_type = 'opportunity' AND c.opportunity_id = entity_id)
					)
					ORDER BY c.created_at DESC
					LIMIT limit_results;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Communication history structures created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create communication history structures: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back communication history migration...")
			
			# Drop functions
			await connection.execute("DROP FUNCTION IF EXISTS get_communication_timeline CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS search_communications CASCADE")
			
			# Drop view
			await connection.execute("DROP VIEW IF EXISTS crm_communication_analytics CASCADE")
			
			# Drop triggers and functions
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_communication_templates_updated_at ON crm_communication_templates")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_communication_templates_updated_at CASCADE")
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_communications_updated_at ON crm_communications")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_communications_updated_at CASCADE")
			
			# Drop tables (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_communication_templates CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_communications CASCADE")
			
			# Drop enums
			await connection.execute("DROP TYPE IF EXISTS crm_communication_outcome CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_communication_priority CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_communication_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_communication_direction CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_communication_type CASCADE")
			
			logger.info("✅ Communication history migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback communication history migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if main tables exist
			tables_exist = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.tables 
				WHERE table_name IN ('crm_communications', 'crm_communication_templates')
			""")
			
			if tables_exist != 2:
				return False
			
			# Check if enums exist
			enum_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_type 
				WHERE typname IN (
					'crm_communication_type',
					'crm_communication_direction',
					'crm_communication_status',
					'crm_communication_priority',
					'crm_communication_outcome'
				)
			""")
			
			if enum_count != 5:
				return False
			
			# Check if view exists
			view_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.views 
					WHERE table_name = 'crm_communication_analytics'
				)
			""")
			
			if not view_exists:
				return False
			
			# Check if functions exist
			function_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.routines 
				WHERE routine_name IN (
					'search_communications',
					'get_communication_timeline',
					'update_crm_communications_updated_at',
					'update_crm_communication_templates_updated_at'
				)
			""")
			
			if function_count < 4:
				return False
			
			# Check key indexes
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename IN ('crm_communications', 'crm_communication_templates')
				AND indexname IN (
					'idx_crm_communications_tenant',
					'idx_crm_communications_contact',
					'idx_crm_communication_templates_tenant',
					'idx_crm_communication_templates_unique_name'
				)
			""")
			
			if index_count < 4:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False