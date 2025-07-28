"""
APG Customer Relationship Management - Lead Scoring Migration

Database migration to create lead scoring tables and supporting structures
for advanced lead scoring and qualification capabilities.

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


class LeadScoringMigration(BaseMigration):
	"""Migration for lead scoring functionality"""
	
	def _get_migration_id(self) -> str:
		return "009_lead_scoring"
	
	def _get_version(self) -> str:
		return "009"
	
	def _get_description(self) -> str:
		return "Create lead scoring tables and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating lead scoring structures...")
			
			# Create score category enum
			await connection.execute("""
				CREATE TYPE crm_score_category AS ENUM (
					'demographic', 'firmographic', 'behavioral', 
					'engagement', 'fit', 'intent'
				)
			""")
			
			# Create score weight enum
			await connection.execute("""
				CREATE TYPE crm_score_weight AS ENUM (
					'critical', 'high', 'medium', 'low'
				)
			""")
			
			# Create lead score rules table
			await connection.execute("""
				CREATE TABLE crm_lead_score_rules (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Rule definition
					name TEXT NOT NULL,
					description TEXT,
					category crm_score_category NOT NULL,
					weight crm_score_weight NOT NULL,
					
					-- Scoring logic
					field TEXT NOT NULL,
					operator TEXT NOT NULL,
					value JSONB NOT NULL,
					score_points INTEGER NOT NULL,
					
					-- Rule conditions
					is_active BOOLEAN DEFAULT true,
					applies_to_lead_sources TEXT[] DEFAULT '{}',
					applies_to_contact_types TEXT[] DEFAULT '{}',
					
					-- Time-based conditions
					valid_from TIMESTAMP WITH TIME ZONE,
					valid_until TIMESTAMP WITH TIME ZONE,
					
					-- Usage tracking
					usage_count INTEGER DEFAULT 0,
					last_used_at TIMESTAMP WITH TIME ZONE,
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Constraints
					CONSTRAINT check_rule_name_length CHECK (char_length(name) >= 1 AND char_length(name) <= 200),
					CONSTRAINT check_rule_description_length CHECK (char_length(description) <= 1000),
					CONSTRAINT check_field_name_length CHECK (char_length(field) >= 1 AND char_length(field) <= 100),
					CONSTRAINT check_operator_length CHECK (char_length(operator) >= 1 AND char_length(operator) <= 50),
					CONSTRAINT check_score_points_range CHECK (score_points >= 0 AND score_points <= 100),
					CONSTRAINT check_usage_count_positive CHECK (usage_count >= 0),
					CONSTRAINT check_valid_date_range CHECK (valid_until IS NULL OR valid_from IS NULL OR valid_from <= valid_until)
				)
			""")
			
			# Create lead scores table
			await connection.execute("""
				CREATE TABLE crm_lead_scores (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					lead_id TEXT NOT NULL,
					contact_id TEXT,
					
					-- Score details
					total_score INTEGER DEFAULT 0,
					max_possible_score INTEGER DEFAULT 0,
					score_percentage DECIMAL(5,2) DEFAULT 0.0,
					grade CHAR(1) DEFAULT 'F',
					
					-- Category breakdown
					demographic_score INTEGER DEFAULT 0,
					firmographic_score INTEGER DEFAULT 0,
					behavioral_score INTEGER DEFAULT 0,
					engagement_score INTEGER DEFAULT 0,
					fit_score INTEGER DEFAULT 0,
					intent_score INTEGER DEFAULT 0,
					
					-- Rule applications
					applied_rules TEXT[] DEFAULT '{}',
					rule_details JSONB DEFAULT '[]',
					
					-- Recommendations
					recommended_action TEXT,
					priority_level TEXT DEFAULT 'medium',
					qualification_status TEXT DEFAULT 'unqualified',
					
					-- Timing and freshness
					calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE,
					calculation_duration_ms INTEGER,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Foreign key constraints
					FOREIGN KEY (lead_id) REFERENCES crm_leads(id) ON DELETE CASCADE,
					FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE SET NULL,
					
					-- Constraints
					CONSTRAINT check_total_score_positive CHECK (total_score >= 0),
					CONSTRAINT check_max_possible_score_positive CHECK (max_possible_score >= 0),
					CONSTRAINT check_score_percentage_range CHECK (score_percentage >= 0 AND score_percentage <= 100),
					CONSTRAINT check_grade_valid CHECK (grade IN ('A', 'B', 'C', 'D', 'F')),
					CONSTRAINT check_category_scores_positive CHECK (
						demographic_score >= 0 AND firmographic_score >= 0 AND 
						behavioral_score >= 0 AND engagement_score >= 0 AND 
						fit_score >= 0 AND intent_score >= 0
					),
					CONSTRAINT check_priority_level_valid CHECK (priority_level IN ('hot', 'warm', 'cold')),
					CONSTRAINT check_qualification_status_valid CHECK (
						qualification_status IN ('qualified', 'marketing_qualified', 'unqualified')
					),
					CONSTRAINT check_calculation_duration_positive CHECK (calculation_duration_ms IS NULL OR calculation_duration_ms >= 0),
					CONSTRAINT check_expires_after_calculated CHECK (expires_at IS NULL OR expires_at > calculated_at)
				)
			""")
			
			# Create lead score history table for tracking score changes
			await connection.execute("""
				CREATE TABLE crm_lead_score_history (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					lead_id TEXT NOT NULL,
					contact_id TEXT,
					
					-- Previous and new scores
					previous_score INTEGER,
					new_score INTEGER,
					previous_grade CHAR(1),
					new_grade CHAR(1),
					score_change INTEGER DEFAULT 0,
					
					-- Change details
					change_reason TEXT,
					changed_rules TEXT[] DEFAULT '{}',
					
					-- Metadata
					changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					changed_by TEXT DEFAULT 'system',
					metadata JSONB DEFAULT '{}',
					
					-- Foreign key constraints
					FOREIGN KEY (lead_id) REFERENCES crm_leads(id) ON DELETE CASCADE,
					FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE SET NULL,
					
					-- Constraints
					CONSTRAINT check_score_change CHECK (score_change = new_score - COALESCE(previous_score, 0)),
					CONSTRAINT check_previous_score_positive CHECK (previous_score IS NULL OR previous_score >= 0),
					CONSTRAINT check_new_score_positive CHECK (new_score >= 0),
					CONSTRAINT check_grades_valid CHECK (
						(previous_grade IS NULL OR previous_grade IN ('A', 'B', 'C', 'D', 'F')) AND
						new_grade IN ('A', 'B', 'C', 'D', 'F')
					)
				)
			""")
			
			# Create indexes for lead score rules table
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_tenant 
				ON crm_lead_score_rules(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_active 
				ON crm_lead_score_rules(is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_category 
				ON crm_lead_score_rules(category)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_weight 
				ON crm_lead_score_rules(weight)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_field 
				ON crm_lead_score_rules(field)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_created_at 
				ON crm_lead_score_rules(created_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_last_used_at 
				ON crm_lead_score_rules(last_used_at)
				WHERE last_used_at IS NOT NULL
			""")
			
			# Create GIN indexes for array and JSONB fields
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_lead_sources 
				ON crm_lead_score_rules USING GIN (applies_to_lead_sources)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_contact_types 
				ON crm_lead_score_rules USING GIN (applies_to_contact_types)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_value 
				ON crm_lead_score_rules USING GIN (value)
			""")
			
			# Create composite indexes for rules
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_tenant_active 
				ON crm_lead_score_rules(tenant_id, is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_tenant_category 
				ON crm_lead_score_rules(tenant_id, category)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_rules_active_valid 
				ON crm_lead_score_rules(is_active, valid_from, valid_until)
			""")
			
			# Create unique constraint for rule names within tenant
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_lead_score_rules_unique_name 
				ON crm_lead_score_rules(tenant_id, name)
				WHERE is_active = true
			""")
			
			# Create indexes for lead scores table
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_tenant 
				ON crm_lead_scores(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_lead 
				ON crm_lead_scores(lead_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_contact 
				ON crm_lead_scores(contact_id)
				WHERE contact_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_total_score 
				ON crm_lead_scores(total_score)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_grade 
				ON crm_lead_scores(grade)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_priority_level 
				ON crm_lead_scores(priority_level)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_qualification_status 
				ON crm_lead_scores(qualification_status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_calculated_at 
				ON crm_lead_scores(calculated_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_expires_at 
				ON crm_lead_scores(expires_at)
				WHERE expires_at IS NOT NULL
			""")
			
			# Create GIN indexes for arrays and JSONB
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_applied_rules 
				ON crm_lead_scores USING GIN (applied_rules)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_rule_details 
				ON crm_lead_scores USING GIN (rule_details)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_metadata 
				ON crm_lead_scores USING GIN (metadata)
			""")
			
			# Create composite indexes for scores
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_tenant_lead 
				ON crm_lead_scores(tenant_id, lead_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_tenant_grade 
				ON crm_lead_scores(tenant_id, grade)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_tenant_priority 
				ON crm_lead_scores(tenant_id, priority_level)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_scores_tenant_qualification 
				ON crm_lead_scores(tenant_id, qualification_status)
			""")
			
			# Create unique constraint to prevent duplicate scores for same lead
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_lead_scores_unique_lead 
				ON crm_lead_scores(tenant_id, lead_id)
			""")
			
			# Create indexes for score history table
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_history_tenant 
				ON crm_lead_score_history(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_history_lead 
				ON crm_lead_score_history(lead_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_history_contact 
				ON crm_lead_score_history(contact_id)
				WHERE contact_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_history_changed_at 
				ON crm_lead_score_history(changed_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_history_changed_by 
				ON crm_lead_score_history(changed_by)
			""")
			
			# Create composite indexes for history
			await connection.execute("""
				CREATE INDEX idx_crm_lead_score_history_tenant_lead 
				ON crm_lead_score_history(tenant_id, lead_id)
			""")
			
			# Create triggers for updating updated_at timestamps
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_lead_score_rules_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_lead_score_rules_updated_at
					BEFORE UPDATE ON crm_lead_score_rules
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_lead_score_rules_updated_at()
			""")
			
			# Create trigger for lead score history tracking
			await connection.execute("""
				CREATE OR REPLACE FUNCTION track_lead_score_changes()
				RETURNS TRIGGER AS $$
				BEGIN
					-- Only track significant score changes (more than 5 points)
					IF TG_OP = 'UPDATE' AND ABS(NEW.total_score - OLD.total_score) >= 5 THEN
						INSERT INTO crm_lead_score_history (
							id, tenant_id, lead_id, contact_id,
							previous_score, new_score, previous_grade, new_grade,
							score_change, change_reason, changed_at, changed_by
						) VALUES (
							gen_random_uuid()::text,
							NEW.tenant_id,
							NEW.lead_id,
							NEW.contact_id,
							OLD.total_score,
							NEW.total_score,
							OLD.grade,
							NEW.grade,
							NEW.total_score - OLD.total_score,
							'Score recalculation',
							NOW(),
							'system'
						);
					END IF;
					
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_lead_scores_history
					AFTER UPDATE ON crm_lead_scores
					FOR EACH ROW
					EXECUTE FUNCTION track_lead_score_changes()
			""")
			
			# Create view for lead scoring analytics
			await connection.execute("""
				CREATE VIEW crm_lead_scoring_analytics AS
				SELECT 
					s.tenant_id,
					
					-- Score statistics
					COUNT(*) as total_leads_scored,
					AVG(s.total_score) as average_score,
					PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.total_score) as median_score,
					MIN(s.total_score) as min_score,
					MAX(s.total_score) as max_score,
					
					-- Grade distribution
					COUNT(*) FILTER (WHERE s.grade = 'A') as grade_a_count,
					COUNT(*) FILTER (WHERE s.grade = 'B') as grade_b_count,
					COUNT(*) FILTER (WHERE s.grade = 'C') as grade_c_count,
					COUNT(*) FILTER (WHERE s.grade = 'D') as grade_d_count,
					COUNT(*) FILTER (WHERE s.grade = 'F') as grade_f_count,
					
					-- Priority distribution
					COUNT(*) FILTER (WHERE s.priority_level = 'hot') as hot_leads,
					COUNT(*) FILTER (WHERE s.priority_level = 'warm') as warm_leads,
					COUNT(*) FILTER (WHERE s.priority_level = 'cold') as cold_leads,
					
					-- Qualification distribution
					COUNT(*) FILTER (WHERE s.qualification_status = 'qualified') as qualified_leads,
					COUNT(*) FILTER (WHERE s.qualification_status = 'marketing_qualified') as mql_leads,
					COUNT(*) FILTER (WHERE s.qualification_status = 'unqualified') as unqualified_leads,
					
					-- Category averages
					AVG(s.demographic_score) as avg_demographic_score,
					AVG(s.firmographic_score) as avg_firmographic_score,
					AVG(s.behavioral_score) as avg_behavioral_score,
					AVG(s.engagement_score) as avg_engagement_score,
					AVG(s.fit_score) as avg_fit_score,
					AVG(s.intent_score) as avg_intent_score,
					
					-- Time-based metrics
					COUNT(*) FILTER (WHERE s.calculated_at >= CURRENT_DATE) as scores_today,
					COUNT(*) FILTER (WHERE s.calculated_at >= DATE_TRUNC('week', NOW())) as scores_this_week,
					COUNT(*) FILTER (WHERE s.calculated_at >= DATE_TRUNC('month', NOW())) as scores_this_month,
					
					-- Performance metrics
					AVG(s.calculation_duration_ms) as avg_calculation_time_ms,
					COUNT(*) FILTER (WHERE s.expires_at < NOW()) as expired_scores
					
				FROM crm_lead_scores s
				GROUP BY s.tenant_id
			""")
			
			# Create function for getting top performing rules
			await connection.execute("""
				CREATE OR REPLACE FUNCTION get_top_performing_rules(
					tenant_filter TEXT DEFAULT NULL,
					limit_count INTEGER DEFAULT 10
				)
				RETURNS TABLE(
					rule_id TEXT,
					rule_name TEXT,
					category TEXT,
					usage_count BIGINT,
					avg_score_contribution DECIMAL,
					effectiveness_score DECIMAL
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						r.id as rule_id,
						r.name as rule_name,
						r.category::TEXT as category,
						r.usage_count::BIGINT,
						AVG(r.score_points)::DECIMAL as avg_score_contribution,
						(r.usage_count * AVG(r.score_points) / GREATEST(EXTRACT(DAYS FROM (NOW() - r.created_at)), 1))::DECIMAL as effectiveness_score
					FROM crm_lead_score_rules r
					WHERE r.is_active = true
					AND (tenant_filter IS NULL OR r.tenant_id = tenant_filter)
					AND r.usage_count > 0
					GROUP BY r.id, r.name, r.category, r.usage_count, r.created_at
					ORDER BY effectiveness_score DESC
					LIMIT limit_count;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for lead score decay (scores get stale over time)
			await connection.execute("""
				CREATE OR REPLACE FUNCTION calculate_score_freshness_factor(
					calculated_at TIMESTAMP WITH TIME ZONE,
					max_age_days INTEGER DEFAULT 7
				)
				RETURNS DECIMAL AS $$
				DECLARE
					age_days DECIMAL;
					freshness_factor DECIMAL;
				BEGIN
					age_days := EXTRACT(DAYS FROM (NOW() - calculated_at));
					
					-- Scores are 100% fresh for first day, then decay linearly
					IF age_days <= 1 THEN
						freshness_factor := 1.0;
					ELSIF age_days >= max_age_days THEN
						freshness_factor := 0.1; -- Minimum 10% freshness
					ELSE
						freshness_factor := 1.0 - (0.9 * (age_days - 1) / (max_age_days - 1));
					END IF;
					
					RETURN freshness_factor;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for batch score expiration cleanup
			await connection.execute("""
				CREATE OR REPLACE FUNCTION cleanup_expired_lead_scores(
					tenant_filter TEXT DEFAULT NULL,
					batch_size INTEGER DEFAULT 1000
				)
				RETURNS TABLE(
					deleted_count BIGINT,
					archived_count BIGINT
				) AS $$
				DECLARE
					deleted_rows BIGINT := 0;
					archived_rows BIGINT := 0;
				BEGIN
					-- Archive scores older than 90 days to history table
					WITH scores_to_archive AS (
						SELECT * FROM crm_lead_scores
						WHERE calculated_at < NOW() - INTERVAL '90 days'
						AND (tenant_filter IS NULL OR tenant_id = tenant_filter)
						LIMIT batch_size
					)
					INSERT INTO crm_lead_score_history (
						id, tenant_id, lead_id, contact_id,
						previous_score, new_score, previous_grade, new_grade,
						score_change, change_reason, changed_at, changed_by, metadata
					)
					SELECT 
						gen_random_uuid()::text,
						tenant_id, lead_id, contact_id,
						NULL, total_score, NULL, grade,
						0, 'Archived due to age',
						NOW(), 'system',
						jsonb_build_object('archived_from_calculated_at', calculated_at)
					FROM scores_to_archive;
					
					GET DIAGNOSTICS archived_rows = ROW_COUNT;
					
					-- Delete the archived scores
					DELETE FROM crm_lead_scores
					WHERE calculated_at < NOW() - INTERVAL '90 days'
					AND (tenant_filter IS NULL OR tenant_id = tenant_filter)
					AND id IN (
						SELECT id FROM crm_lead_scores
						WHERE calculated_at < NOW() - INTERVAL '90 days'
						LIMIT batch_size
					);
					
					GET DIAGNOSTICS deleted_rows = ROW_COUNT;
					
					RETURN QUERY SELECT deleted_rows, archived_rows;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Lead scoring structures created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create lead scoring structures: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back lead scoring migration...")
			
			# Drop functions
			await connection.execute("DROP FUNCTION IF EXISTS cleanup_expired_lead_scores CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS calculate_score_freshness_factor CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS get_top_performing_rules CASCADE")
			
			# Drop view
			await connection.execute("DROP VIEW IF EXISTS crm_lead_scoring_analytics CASCADE")
			
			# Drop triggers and functions
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_lead_scores_history ON crm_lead_scores")
			await connection.execute("DROP FUNCTION IF EXISTS track_lead_score_changes CASCADE")
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_lead_score_rules_updated_at ON crm_lead_score_rules")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_lead_score_rules_updated_at CASCADE")
			
			# Drop tables (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_lead_score_history CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_lead_scores CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_lead_score_rules CASCADE")
			
			# Drop enums
			await connection.execute("DROP TYPE IF EXISTS crm_score_weight CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_score_category CASCADE")
			
			logger.info("✅ Lead scoring migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback lead scoring migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if main tables exist
			tables_exist = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.tables 
				WHERE table_name IN ('crm_lead_score_rules', 'crm_lead_scores', 'crm_lead_score_history')
			""")
			
			if tables_exist != 3:
				return False
			
			# Check if enums exist
			enum_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_type 
				WHERE typname IN ('crm_score_category', 'crm_score_weight')
			""")
			
			if enum_count != 2:
				return False
			
			# Check if view exists
			view_exists = await connection.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM information_schema.views 
					WHERE table_name = 'crm_lead_scoring_analytics'
				)
			""")
			
			if not view_exists:
				return False
			
			# Check if functions exist
			function_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.routines 
				WHERE routine_name IN (
					'get_top_performing_rules',
					'calculate_score_freshness_factor',
					'cleanup_expired_lead_scores',
					'track_lead_score_changes',
					'update_crm_lead_score_rules_updated_at'
				)
			""")
			
			if function_count < 5:
				return False
			
			# Check key indexes
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename IN ('crm_lead_score_rules', 'crm_lead_scores', 'crm_lead_score_history')
				AND indexname IN (
					'idx_crm_lead_score_rules_tenant',
					'idx_crm_lead_scores_tenant',
					'idx_crm_lead_score_history_tenant',
					'idx_crm_lead_scores_unique_lead'
				)
			""")
			
			if index_count < 4:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False