"""
APG Customer Relationship Management - Sales Pipeline Migration

Database migration to create sales pipeline tables and supporting structures
for advanced pipeline management, stage tracking, and opportunity analytics.

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


class SalesPipelineMigration(BaseMigration):
	"""Migration for sales pipeline functionality"""
	
	def _get_migration_id(self) -> str:
		return "010_sales_pipeline"
	
	def _get_version(self) -> str:
		return "010"
	
	def _get_description(self) -> str:
		return "Create sales pipeline tables and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating sales pipeline structures...")
			
			# Create stage type enum
			await connection.execute("""
				CREATE TYPE crm_stage_type AS ENUM (
					'prospecting', 'qualification', 'needs_analysis', 'proposal',
					'negotiation', 'decision', 'closed_won', 'closed_lost', 'on_hold', 'custom'
				)
			""")
			
			# Create stage category enum
			await connection.execute("""
				CREATE TYPE crm_stage_category AS ENUM (
					'early', 'middle', 'late', 'closed', 'inactive'
				)
			""")
			
			# Create automation trigger enum
			await connection.execute("""
				CREATE TYPE crm_automation_trigger AS ENUM (
					'time_based', 'activity_based', 'score_based', 'manual_only'
				)
			""")
			
			# Create sales pipelines table
			await connection.execute("""
				CREATE TABLE crm_sales_pipelines (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Basic information
					name TEXT NOT NULL,
					description TEXT,
					is_default BOOLEAN DEFAULT false,
					is_active BOOLEAN DEFAULT true,
					
					-- Pipeline settings
					currency TEXT DEFAULT 'USD',
					enable_stage_automation BOOLEAN DEFAULT false,
					enable_probability_updates BOOLEAN DEFAULT true,
					enable_forecasting BOOLEAN DEFAULT true,
					
					-- Performance tracking
					total_opportunities INTEGER DEFAULT 0,
					total_value DECIMAL(15,2) DEFAULT 0,
					weighted_value DECIMAL(15,2) DEFAULT 0,
					average_deal_size DECIMAL(15,2) DEFAULT 0,
					average_cycle_time DECIMAL(8,2) DEFAULT 0,
					win_rate DECIMAL(5,2) DEFAULT 0,
					
					-- Team assignments
					assigned_teams TEXT[] DEFAULT '{}',
					assigned_users TEXT[] DEFAULT '{}',
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Constraints
					CONSTRAINT check_pipeline_name_length CHECK (char_length(name) >= 1 AND char_length(name) <= 200),
					CONSTRAINT check_pipeline_description_length CHECK (char_length(description) <= 1000),
					CONSTRAINT check_currency_length CHECK (char_length(currency) = 3),
					CONSTRAINT check_total_opportunities_positive CHECK (total_opportunities >= 0),
					CONSTRAINT check_total_value_positive CHECK (total_value >= 0),
					CONSTRAINT check_weighted_value_positive CHECK (weighted_value >= 0),
					CONSTRAINT check_average_deal_size_positive CHECK (average_deal_size >= 0),
					CONSTRAINT check_average_cycle_time_positive CHECK (average_cycle_time >= 0),
					CONSTRAINT check_win_rate_range CHECK (win_rate >= 0 AND win_rate <= 100)
				)
			""")
			
			# Create pipeline stages table
			await connection.execute("""
				CREATE TABLE crm_pipeline_stages (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					pipeline_id TEXT NOT NULL,
					
					-- Basic information
					name TEXT NOT NULL,
					description TEXT,
					stage_type crm_stage_type DEFAULT 'custom',
					category crm_stage_category DEFAULT 'early',
					
					-- Stage properties
					"order" INTEGER NOT NULL,
					probability DECIMAL(5,2) NOT NULL,
					is_active BOOLEAN DEFAULT true,
					is_closed BOOLEAN DEFAULT false,
					is_won BOOLEAN DEFAULT false,
					
					-- Duration and timing
					expected_duration_days INTEGER,
					max_duration_days INTEGER,
					
					-- Automation settings
					automation_trigger crm_automation_trigger DEFAULT 'manual_only',
					auto_advance_conditions JSONB DEFAULT '{}',
					required_activities TEXT[] DEFAULT '{}',
					required_fields TEXT[] DEFAULT '{}',
					
					-- Stage-specific settings
					allows_forecasting BOOLEAN DEFAULT true,
					weighted_probability DECIMAL(5,2) DEFAULT 0,
					conversion_tracking BOOLEAN DEFAULT true,
					
					-- Notifications and alerts
					enable_alerts BOOLEAN DEFAULT false,
					alert_conditions JSONB DEFAULT '{}',
					notification_emails TEXT[] DEFAULT '{}',
					
					-- Colors and styling
					color TEXT DEFAULT '#007bff',
					icon TEXT DEFAULT 'circle',
					
					-- Performance tracking
					opportunity_count INTEGER DEFAULT 0,
					total_value DECIMAL(15,2) DEFAULT 0,
					average_duration DECIMAL(8,2) DEFAULT 0,
					conversion_rate DECIMAL(5,2) DEFAULT 0,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Foreign key constraints
					FOREIGN KEY (pipeline_id) REFERENCES crm_sales_pipelines(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_stage_name_length CHECK (char_length(name) >= 1 AND char_length(name) <= 100),
					CONSTRAINT check_stage_description_length CHECK (char_length(description) <= 500),
					CONSTRAINT check_stage_order_positive CHECK ("order" > 0),
					CONSTRAINT check_probability_range CHECK (probability >= 0 AND probability <= 100),
					CONSTRAINT check_expected_duration_positive CHECK (expected_duration_days IS NULL OR expected_duration_days > 0),
					CONSTRAINT check_max_duration_positive CHECK (max_duration_days IS NULL OR max_duration_days > 0),
					CONSTRAINT check_duration_relationship CHECK (max_duration_days IS NULL OR expected_duration_days IS NULL OR max_duration_days >= expected_duration_days),
					CONSTRAINT check_weighted_probability_range CHECK (weighted_probability >= 0 AND weighted_probability <= 100),
					CONSTRAINT check_color_format CHECK (color ~ '^#[0-9a-fA-F]{6}$'),
					CONSTRAINT check_opportunity_count_positive CHECK (opportunity_count >= 0),
					CONSTRAINT check_stage_total_value_positive CHECK (total_value >= 0),
					CONSTRAINT check_average_duration_positive CHECK (average_duration >= 0),
					CONSTRAINT check_conversion_rate_range CHECK (conversion_rate >= 0 AND conversion_rate <= 100)
				)
			""")
			
			# Create opportunity stage history table
			await connection.execute("""
				CREATE TABLE crm_opportunity_stage_history (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					opportunity_id TEXT NOT NULL,
					
					-- Stage change details
					from_stage_id TEXT,
					to_stage_id TEXT NOT NULL,
					from_stage_name TEXT,
					to_stage_name TEXT NOT NULL,
					
					-- Change metrics
					previous_probability DECIMAL(5,2),
					new_probability DECIMAL(5,2) NOT NULL,
					previous_value DECIMAL(15,2),
					new_value DECIMAL(15,2) NOT NULL,
					days_in_previous_stage INTEGER,
					
					-- Change context
					change_reason TEXT,
					notes TEXT,
					automated BOOLEAN DEFAULT false,
					
					-- Audit fields
					changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					changed_by TEXT NOT NULL,
					metadata JSONB DEFAULT '{}',
					
					-- Foreign key constraints
					FOREIGN KEY (opportunity_id) REFERENCES crm_opportunities(id) ON DELETE CASCADE,
					FOREIGN KEY (from_stage_id) REFERENCES crm_pipeline_stages(id) ON DELETE SET NULL,
					FOREIGN KEY (to_stage_id) REFERENCES crm_pipeline_stages(id) ON DELETE RESTRICT,
					
					-- Constraints
					CONSTRAINT check_probabilities_range CHECK (
						(previous_probability IS NULL OR (previous_probability >= 0 AND previous_probability <= 100)) AND
						(new_probability >= 0 AND new_probability <= 100)
					),
					CONSTRAINT check_values_positive CHECK (
						(previous_value IS NULL OR previous_value >= 0) AND
						new_value >= 0
					),
					CONSTRAINT check_days_in_stage_positive CHECK (days_in_previous_stage IS NULL OR days_in_previous_stage >= 0),
					CONSTRAINT check_change_reason_length CHECK (char_length(change_reason) <= 500),
					CONSTRAINT check_notes_length CHECK (char_length(notes) <= 2000)
				)
			""")
			
			# Create indexes for pipelines table
			await connection.execute("""
				CREATE INDEX idx_crm_sales_pipelines_tenant 
				ON crm_sales_pipelines(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_sales_pipelines_active 
				ON crm_sales_pipelines(is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_sales_pipelines_default 
				ON crm_sales_pipelines(is_default)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_sales_pipelines_created_at 
				ON crm_sales_pipelines(created_at)
			""")
			
			# Create GIN indexes for arrays and JSONB
			await connection.execute("""
				CREATE INDEX idx_crm_sales_pipelines_assigned_teams 
				ON crm_sales_pipelines USING GIN (assigned_teams)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_sales_pipelines_assigned_users 
				ON crm_sales_pipelines USING GIN (assigned_users)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_sales_pipelines_metadata 
				ON crm_sales_pipelines USING GIN (metadata)
			""")
			
			# Create composite indexes for pipelines
			await connection.execute("""
				CREATE INDEX idx_crm_sales_pipelines_tenant_active 
				ON crm_sales_pipelines(tenant_id, is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_sales_pipelines_tenant_default 
				ON crm_sales_pipelines(tenant_id, is_default)
			""")
			
			# Ensure only one default pipeline per tenant
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_sales_pipelines_unique_default 
				ON crm_sales_pipelines(tenant_id)
				WHERE is_default = true
			""")
			
			# Create indexes for pipeline stages table
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_tenant 
				ON crm_pipeline_stages(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_pipeline 
				ON crm_pipeline_stages(pipeline_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_type 
				ON crm_pipeline_stages(stage_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_category 
				ON crm_pipeline_stages(category)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_active 
				ON crm_pipeline_stages(is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_closed 
				ON crm_pipeline_stages(is_closed)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_order 
				ON crm_pipeline_stages("order")
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_probability 
				ON crm_pipeline_stages(probability)
			""")
			
			# Create GIN indexes for arrays and JSONB in stages
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_auto_advance_conditions 
				ON crm_pipeline_stages USING GIN (auto_advance_conditions)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_required_activities 
				ON crm_pipeline_stages USING GIN (required_activities)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_required_fields 
				ON crm_pipeline_stages USING GIN (required_fields)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_alert_conditions 
				ON crm_pipeline_stages USING GIN (alert_conditions)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_notification_emails 
				ON crm_pipeline_stages USING GIN (notification_emails)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_metadata 
				ON crm_pipeline_stages USING GIN (metadata)
			""")
			
			# Create composite indexes for stages
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_tenant_pipeline 
				ON crm_pipeline_stages(tenant_id, pipeline_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_pipeline_order 
				ON crm_pipeline_stages(pipeline_id, "order")
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_pipeline_stages_pipeline_active 
				ON crm_pipeline_stages(pipeline_id, is_active)
			""")
			
			# Ensure unique stage order within pipeline
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_pipeline_stages_unique_order 
				ON crm_pipeline_stages(pipeline_id, "order")
				WHERE is_active = true
			""")
			
			# Ensure unique stage names within pipeline
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_pipeline_stages_unique_name 
				ON crm_pipeline_stages(pipeline_id, name)
				WHERE is_active = true
			""")
			
			# Create indexes for stage history table
			await connection.execute("""
				CREATE INDEX idx_crm_opportunity_stage_history_tenant 
				ON crm_opportunity_stage_history(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_opportunity_stage_history_opportunity 
				ON crm_opportunity_stage_history(opportunity_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_opportunity_stage_history_from_stage 
				ON crm_opportunity_stage_history(from_stage_id)
				WHERE from_stage_id IS NOT NULL
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_opportunity_stage_history_to_stage 
				ON crm_opportunity_stage_history(to_stage_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_opportunity_stage_history_changed_at 
				ON crm_opportunity_stage_history(changed_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_opportunity_stage_history_changed_by 
				ON crm_opportunity_stage_history(changed_by)
			""")
			
			# Create GIN index for metadata in history
			await connection.execute("""
				CREATE INDEX idx_crm_opportunity_stage_history_metadata 
				ON crm_opportunity_stage_history USING GIN (metadata)
			""")
			
			# Create composite indexes for history
			await connection.execute("""
				CREATE INDEX idx_crm_opportunity_stage_history_tenant_opportunity 
				ON crm_opportunity_stage_history(tenant_id, opportunity_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_opportunity_stage_history_opportunity_changed_at 
				ON crm_opportunity_stage_history(opportunity_id, changed_at)
			""")
			
			# Create triggers for updating updated_at timestamps
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_sales_pipelines_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_sales_pipelines_updated_at
					BEFORE UPDATE ON crm_sales_pipelines
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_sales_pipelines_updated_at()
			""")
			
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_pipeline_stages_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_pipeline_stages_updated_at
					BEFORE UPDATE ON crm_pipeline_stages
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_pipeline_stages_updated_at()
			""")
			
			# Create view for pipeline analytics
			await connection.execute("""
				CREATE VIEW crm_pipeline_analytics AS
				SELECT 
					p.id as pipeline_id,
					p.tenant_id,
					p.name as pipeline_name,
					p.is_default,
					p.is_active,
					
					-- Overall metrics
					COUNT(o.id) as total_opportunities,
					COALESCE(SUM(o.value), 0) as total_value,
					COALESCE(SUM(o.value * ps.probability / 100), 0) as weighted_value,
					COALESCE(AVG(o.value), 0) as average_deal_size,
					
					-- Stage metrics
					COUNT(ps.id) as total_stages,
					COUNT(ps.id) FILTER (WHERE ps.is_active = true) as active_stages,
					COUNT(ps.id) FILTER (WHERE ps.is_closed = true) as closed_stages,
					
					-- Win/loss metrics
					COUNT(o.id) FILTER (WHERE ps.is_won = true) as won_opportunities,
					COUNT(o.id) FILTER (WHERE ps.is_closed = true AND ps.is_won = false) as lost_opportunities,
					COUNT(o.id) FILTER (WHERE ps.is_closed = true) as closed_opportunities,
					
					-- Time-based metrics
					COUNT(o.id) FILTER (WHERE o.created_at >= CURRENT_DATE) as opportunities_today,
					COUNT(o.id) FILTER (WHERE o.created_at >= DATE_TRUNC('week', NOW())) as opportunities_this_week,
					COUNT(o.id) FILTER (WHERE o.created_at >= DATE_TRUNC('month', NOW())) as opportunities_this_month,
					
					-- Velocity metrics
					COUNT(o.id) FILTER (WHERE ps.is_closed = true AND o.updated_at >= DATE_TRUNC('month', NOW())) as closed_this_month,
					COUNT(o.id) FILTER (WHERE ps.is_won = true AND o.updated_at >= DATE_TRUNC('month', NOW())) as won_this_month
					
				FROM crm_sales_pipelines p
				LEFT JOIN crm_pipeline_stages ps ON p.id = ps.pipeline_id
				LEFT JOIN crm_opportunities o ON ps.id = o.stage_id
				GROUP BY p.id, p.tenant_id, p.name, p.is_default, p.is_active
			""")
			
			# Create view for stage performance metrics
			await connection.execute("""
				CREATE VIEW crm_stage_performance AS
				SELECT 
					ps.id as stage_id,
					ps.tenant_id,
					ps.pipeline_id,
					ps.name as stage_name,
					ps.stage_type,
					ps.category,
					ps."order",
					ps.probability,
					ps.is_active,
					ps.is_closed,
					ps.is_won,
					
					-- Opportunity metrics
					COUNT(o.id) as current_opportunities,
					COALESCE(SUM(o.value), 0) as current_value,
					COALESCE(AVG(o.value), 0) as average_opportunity_value,
					
					-- Duration metrics  
					COALESCE(AVG(EXTRACT(DAYS FROM (NOW() - o.stage_updated_at))), 0) as average_days_in_stage,
					COALESCE(MAX(EXTRACT(DAYS FROM (NOW() - o.stage_updated_at))), 0) as max_days_in_stage,
					COALESCE(MIN(EXTRACT(DAYS FROM (NOW() - o.stage_updated_at))), 0) as min_days_in_stage,
					
					-- Historical metrics
					(SELECT COUNT(*) FROM crm_opportunity_stage_history h 
					 WHERE h.to_stage_id = ps.id AND h.changed_at >= NOW() - INTERVAL '30 days') as entries_last_30_days,
					
					(SELECT COUNT(*) FROM crm_opportunity_stage_history h 
					 WHERE h.from_stage_id = ps.id AND h.changed_at >= NOW() - INTERVAL '30 days') as exits_last_30_days,
					
					(SELECT AVG(h.days_in_previous_stage) FROM crm_opportunity_stage_history h 
					 WHERE h.from_stage_id = ps.id AND h.days_in_previous_stage IS NOT NULL 
					 AND h.changed_at >= NOW() - INTERVAL '90 days') as historical_average_duration
					
				FROM crm_pipeline_stages ps
				LEFT JOIN crm_opportunities o ON ps.id = o.stage_id
				GROUP BY ps.id, ps.tenant_id, ps.pipeline_id, ps.name, ps.stage_type, 
						 ps.category, ps."order", ps.probability, ps.is_active, ps.is_closed, ps.is_won
			""")
			
			# Create function for calculating stage conversion rates
			await connection.execute("""
				CREATE OR REPLACE FUNCTION calculate_stage_conversion_rates(
					pipeline_filter TEXT DEFAULT NULL,
					tenant_filter TEXT DEFAULT NULL,
					days_back INTEGER DEFAULT 90
				)
				RETURNS TABLE(
					stage_id TEXT,
					stage_name TEXT,
					entries BIGINT,
					exits BIGINT,
					conversion_rate DECIMAL
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						ps.id as stage_id,
						ps.name as stage_name,
						COALESCE(entries.count, 0) as entries,
						COALESCE(exits.count, 0) as exits,
						CASE 
							WHEN COALESCE(entries.count, 0) > 0 THEN
								(COALESCE(exits.count, 0)::DECIMAL / entries.count) * 100
							ELSE 0
						END as conversion_rate
					FROM crm_pipeline_stages ps
					LEFT JOIN (
						SELECT to_stage_id, COUNT(*) as count
						FROM crm_opportunity_stage_history
						WHERE changed_at >= NOW() - (days_back || ' days')::INTERVAL
						AND (tenant_filter IS NULL OR tenant_id = tenant_filter)
						GROUP BY to_stage_id
					) entries ON ps.id = entries.to_stage_id
					LEFT JOIN (
						SELECT from_stage_id, COUNT(*) as count
						FROM crm_opportunity_stage_history
						WHERE changed_at >= NOW() - (days_back || ' days')::INTERVAL
						AND (tenant_filter IS NULL OR tenant_id = tenant_filter)
						GROUP BY from_stage_id
					) exits ON ps.id = exits.from_stage_id
					WHERE (pipeline_filter IS NULL OR ps.pipeline_id = pipeline_filter)
					AND (tenant_filter IS NULL OR ps.tenant_id = tenant_filter)
					AND ps.is_active = true
					ORDER BY ps."order";
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for pipeline forecasting
			await connection.execute("""
				CREATE OR REPLACE FUNCTION calculate_pipeline_forecast(
					pipeline_filter TEXT,
					tenant_filter TEXT,
					forecast_months INTEGER DEFAULT 3
				)
				RETURNS TABLE(
					month_offset INTEGER,
					forecast_month DATE,
					weighted_forecast DECIMAL,
					conservative_forecast DECIMAL,
					optimistic_forecast DECIMAL,
					confidence_score DECIMAL
				) AS $$
				DECLARE
					base_weighted DECIMAL;
					historical_win_rate DECIMAL;
					opportunity_count INTEGER;
				BEGIN
					-- Get base metrics
					SELECT 
						COALESCE(SUM(o.value * ps.probability / 100), 0),
						COALESCE(COUNT(o.id), 0)
					INTO base_weighted, opportunity_count
					FROM crm_opportunities o
					JOIN crm_pipeline_stages ps ON o.stage_id = ps.id
					WHERE ps.pipeline_id = pipeline_filter
					AND o.tenant_id = tenant_filter
					AND ps.is_closed = false
					AND ps.allows_forecasting = true;
					
					-- Get historical win rate
					SELECT 
						CASE 
							WHEN COUNT(*) FILTER (WHERE ps.is_closed = true) > 0 THEN
								(COUNT(*) FILTER (WHERE ps.is_won = true)::DECIMAL / 
								 COUNT(*) FILTER (WHERE ps.is_closed = true)) * 100
							ELSE 50.0
						END
					INTO historical_win_rate
					FROM crm_opportunities o
					JOIN crm_pipeline_stages ps ON o.stage_id = ps.id
					WHERE ps.pipeline_id = pipeline_filter
					AND o.tenant_id = tenant_filter
					AND o.updated_at >= NOW() - INTERVAL '12 months';
					
					-- Generate forecasts for each month
					FOR i IN 0..forecast_months-1 LOOP
						RETURN QUERY SELECT
							i as month_offset,
							(DATE_TRUNC('month', NOW()) + (i || ' months')::INTERVAL)::DATE as forecast_month,
							(base_weighted * (0.3 + i * 0.2)) as weighted_forecast,
							(base_weighted * (0.2 + i * 0.15)) as conservative_forecast,
							(base_weighted * (0.5 + i * 0.25)) as optimistic_forecast,
							GREATEST(20.0, LEAST(95.0, historical_win_rate + (opportunity_count * 2.0))) as confidence_score;
					END LOOP;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for identifying pipeline bottlenecks
			await connection.execute("""
				CREATE OR REPLACE FUNCTION identify_pipeline_bottlenecks(
					pipeline_filter TEXT,
					tenant_filter TEXT,
					days_back INTEGER DEFAULT 30
				)
				RETURNS TABLE(
					stage_id TEXT,
					stage_name TEXT,
					stage_order INTEGER,
					opportunities_stuck INTEGER,
					average_days_stuck DECIMAL,
					bottleneck_score DECIMAL,
					recommended_action TEXT
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						ps.id as stage_id,
						ps.name as stage_name,
						ps."order" as stage_order,
						COUNT(o.id)::INTEGER as opportunities_stuck,
						AVG(EXTRACT(DAYS FROM (NOW() - o.stage_updated_at)))::DECIMAL as average_days_stuck,
						(COUNT(o.id) * AVG(EXTRACT(DAYS FROM (NOW() - o.stage_updated_at))) / 10)::DECIMAL as bottleneck_score,
						CASE 
							WHEN AVG(EXTRACT(DAYS FROM (NOW() - o.stage_updated_at))) > COALESCE(ps.expected_duration_days, 14) * 2 THEN
								'Review and accelerate stalled opportunities'
							WHEN COUNT(o.id) > 10 THEN
								'High volume stage - consider process optimization'
							WHEN ps.conversion_rate < 50 THEN
								'Low conversion rate - review stage criteria'
							ELSE
								'Monitor stage performance'
						END as recommended_action
					FROM crm_pipeline_stages ps
					LEFT JOIN crm_opportunities o ON ps.id = o.stage_id
						AND o.stage_updated_at < NOW() - INTERVAL '7 days'
					WHERE ps.pipeline_id = pipeline_filter
					AND ps.tenant_id = tenant_filter
					AND ps.is_active = true
					AND ps.is_closed = false
					GROUP BY ps.id, ps.name, ps."order", ps.expected_duration_days, ps.conversion_rate
					HAVING COUNT(o.id) > 0
					ORDER BY bottleneck_score DESC;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Sales pipeline structures created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create sales pipeline structures: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back sales pipeline migration...")
			
			# Drop functions
			await connection.execute("DROP FUNCTION IF EXISTS identify_pipeline_bottlenecks CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS calculate_pipeline_forecast CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS calculate_stage_conversion_rates CASCADE")
			
			# Drop views
			await connection.execute("DROP VIEW IF EXISTS crm_stage_performance CASCADE")
			await connection.execute("DROP VIEW IF EXISTS crm_pipeline_analytics CASCADE")
			
			# Drop triggers and functions
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_pipeline_stages_updated_at ON crm_pipeline_stages")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_pipeline_stages_updated_at CASCADE")
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_sales_pipelines_updated_at ON crm_sales_pipelines")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_sales_pipelines_updated_at CASCADE")
			
			# Drop tables (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_opportunity_stage_history CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_pipeline_stages CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_sales_pipelines CASCADE")
			
			# Drop enums
			await connection.execute("DROP TYPE IF EXISTS crm_automation_trigger CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_stage_category CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_stage_type CASCADE")
			
			logger.info("✅ Sales pipeline migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback sales pipeline migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if main tables exist
			tables_exist = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.tables 
				WHERE table_name IN ('crm_sales_pipelines', 'crm_pipeline_stages', 'crm_opportunity_stage_history')
			""")
			
			if tables_exist != 3:
				return False
			
			# Check if enums exist
			enum_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_type 
				WHERE typname IN ('crm_stage_type', 'crm_stage_category', 'crm_automation_trigger')
			""")
			
			if enum_count != 3:
				return False
			
			# Check if views exist
			view_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.views 
				WHERE table_name IN ('crm_pipeline_analytics', 'crm_stage_performance')
			""")
			
			if view_count != 2:
				return False
			
			# Check if functions exist
			function_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.routines 
				WHERE routine_name IN (
					'calculate_stage_conversion_rates',
					'calculate_pipeline_forecast',
					'identify_pipeline_bottlenecks',
					'update_crm_sales_pipelines_updated_at',
					'update_crm_pipeline_stages_updated_at'
				)
			""")
			
			if function_count < 5:
				return False
			
			# Check key indexes
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename IN ('crm_sales_pipelines', 'crm_pipeline_stages', 'crm_opportunity_stage_history')
				AND indexname IN (
					'idx_crm_sales_pipelines_tenant',
					'idx_crm_pipeline_stages_tenant',
					'idx_crm_opportunity_stage_history_tenant',
					'idx_crm_pipeline_stages_unique_order'
				)
			""")
			
			if index_count < 4:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False