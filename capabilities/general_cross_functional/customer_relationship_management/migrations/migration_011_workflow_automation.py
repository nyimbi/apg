"""
APG Customer Relationship Management - Workflow Automation Migration

Database migration to create workflow automation tables and supporting structures
for advanced workflow management, execution tracking, and automation analytics.

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


class WorkflowAutomationMigration(BaseMigration):
	"""Migration for workflow automation functionality"""
	
	def _get_migration_id(self) -> str:
		return "011_workflow_automation"
	
	def _get_version(self) -> str:
		return "011"
	
	def _get_description(self) -> str:
		return "Create workflow automation tables and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating workflow automation structures...")
			
			# Create workflow trigger type enum
			await connection.execute("""
				CREATE TYPE crm_workflow_trigger_type AS ENUM (
					'manual', 'schedule', 'record_created', 'record_updated', 'record_deleted',
					'field_changed', 'status_changed', 'email_received', 'email_sent',
					'activity_created', 'opportunity_stage_changed', 'lead_score_changed',
					'date_based', 'time_based', 'webhook', 'api_call', 'form_submitted'
				)
			""")
			
			# Create workflow status enum
			await connection.execute("""
				CREATE TYPE crm_workflow_status AS ENUM (
					'draft', 'active', 'paused', 'inactive', 'archived'
				)
			""")
			
			# Create workflow execution status enum
			await connection.execute("""
				CREATE TYPE crm_workflow_execution_status AS ENUM (
					'pending', 'running', 'completed', 'failed', 'cancelled', 'timeout'
				)
			""")
			
			# Create action type enum
			await connection.execute("""
				CREATE TYPE crm_workflow_action_type AS ENUM (
					'send_email', 'create_task', 'update_record', 'create_record',
					'assign_user', 'change_status', 'add_tag', 'remove_tag',
					'create_activity', 'schedule_meeting', 'send_sms', 'webhook_call',
					'api_request', 'create_opportunity', 'update_lead_score',
					'move_pipeline_stage', 'create_note', 'send_notification',
					'wait_delay', 'conditional_branch', 'custom_function'
				)
			""")
			
			# Create workflows table
			await connection.execute("""
				CREATE TABLE crm_workflows (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Basic information
					name TEXT NOT NULL,
					description TEXT,
					category TEXT DEFAULT 'general',
					tags TEXT[] DEFAULT '{}',
					
					-- Status and settings
					status crm_workflow_status DEFAULT 'draft',
					is_active BOOLEAN DEFAULT false,
					priority INTEGER DEFAULT 5,
					
					-- Trigger configuration
					trigger_type crm_workflow_trigger_type NOT NULL,
					trigger_conditions JSONB DEFAULT '{}',
					trigger_schedule JSONB DEFAULT '{}',
					
					-- Execution settings
					execution_limit INTEGER,
					execution_timeout INTEGER DEFAULT 300,
					max_retries INTEGER DEFAULT 3,
					retry_delay INTEGER DEFAULT 60,
					
					-- Actions configuration
					actions JSONB NOT NULL DEFAULT '[]',
					action_sequence TEXT[] DEFAULT '{}',
					
					-- Advanced settings
					enable_logging BOOLEAN DEFAULT true,
					log_level TEXT DEFAULT 'info',
					enable_analytics BOOLEAN DEFAULT true,
					enable_notifications BOOLEAN DEFAULT false,
					notification_emails TEXT[] DEFAULT '{}',
					
					-- Conditions and filters
					entry_conditions JSONB DEFAULT '{}',
					exit_conditions JSONB DEFAULT '{}',
					filter_conditions JSONB DEFAULT '{}',
					
					-- Performance tracking
					total_executions INTEGER DEFAULT 0,
					successful_executions INTEGER DEFAULT 0,
					failed_executions INTEGER DEFAULT 0,
					average_execution_time DECIMAL(8,2) DEFAULT 0,
					last_execution_at TIMESTAMP WITH TIME ZONE,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Constraints
					CONSTRAINT check_workflow_name_length CHECK (char_length(name) >= 1 AND char_length(name) <= 200),
					CONSTRAINT check_workflow_description_length CHECK (char_length(description) <= 1000),
					CONSTRAINT check_workflow_category_length CHECK (char_length(category) <= 50),
					CONSTRAINT check_priority_range CHECK (priority >= 1 AND priority <= 10),
					CONSTRAINT check_execution_limit_positive CHECK (execution_limit IS NULL OR execution_limit > 0),
					CONSTRAINT check_execution_timeout_positive CHECK (execution_timeout > 0),
					CONSTRAINT check_max_retries_positive CHECK (max_retries >= 0),
					CONSTRAINT check_retry_delay_positive CHECK (retry_delay >= 0),
					CONSTRAINT check_execution_counters_positive CHECK (
						total_executions >= 0 AND 
						successful_executions >= 0 AND 
						failed_executions >= 0 AND
						successful_executions + failed_executions <= total_executions
					),
					CONSTRAINT check_average_execution_time_positive CHECK (average_execution_time >= 0)
				)
			""")
			
			# Create workflow executions table
			await connection.execute("""
				CREATE TABLE crm_workflow_executions (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					workflow_id TEXT NOT NULL,
					
					-- Execution context
					triggered_by TEXT NOT NULL,
					trigger_data JSONB DEFAULT '{}',
					execution_context JSONB DEFAULT '{}',
					
					-- Associated records
					record_id TEXT,
					record_type TEXT,
					related_records JSONB DEFAULT '{}',
					
					-- Execution details
					status crm_workflow_execution_status DEFAULT 'pending',
					started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					completed_at TIMESTAMP WITH TIME ZONE,
					execution_time_ms INTEGER,
					
					-- Progress tracking
					current_step INTEGER DEFAULT 0,
					total_steps INTEGER DEFAULT 0,
					completed_steps INTEGER DEFAULT 0,
					failed_steps INTEGER DEFAULT 0,
					
					-- Results and logs
					execution_results JSONB DEFAULT '{}',
					step_results JSONB DEFAULT '[]',
					error_details JSONB DEFAULT '{}',
					execution_logs JSONB DEFAULT '[]',
					
					-- Retry information
					retry_count INTEGER DEFAULT 0,
					last_retry_at TIMESTAMP WITH TIME ZONE,
					next_retry_at TIMESTAMP WITH TIME ZONE,
					
					-- Performance metrics
					cpu_time_ms INTEGER DEFAULT 0,
					memory_usage_mb INTEGER DEFAULT 0,
					api_calls_made INTEGER DEFAULT 0,
					data_processed_bytes INTEGER DEFAULT 0,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Foreign key constraints
					FOREIGN KEY (workflow_id) REFERENCES crm_workflows(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_execution_time_positive CHECK (execution_time_ms IS NULL OR execution_time_ms >= 0),
					CONSTRAINT check_step_counters_positive CHECK (
						current_step >= 0 AND 
						total_steps >= 0 AND 
						completed_steps >= 0 AND 
						failed_steps >= 0 AND
						completed_steps + failed_steps <= total_steps AND
						current_step <= total_steps
					),
					CONSTRAINT check_retry_count_positive CHECK (retry_count >= 0),
					CONSTRAINT check_performance_metrics_positive CHECK (
						cpu_time_ms >= 0 AND 
						memory_usage_mb >= 0 AND 
						api_calls_made >= 0 AND 
						data_processed_bytes >= 0
					)
				)
			""")
			
			# Create workflow action executions table
			await connection.execute("""
				CREATE TABLE crm_workflow_action_executions (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					execution_id TEXT NOT NULL,
					workflow_id TEXT NOT NULL,
					
					-- Action details
					action_id TEXT NOT NULL,
					action_type crm_workflow_action_type NOT NULL,
					action_config JSONB DEFAULT '{}',
					step_order INTEGER NOT NULL,
					
					-- Execution details
					status crm_workflow_execution_status DEFAULT 'pending',
					started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					completed_at TIMESTAMP WITH TIME ZONE,
					execution_time_ms INTEGER,
					
					-- Results
					action_result JSONB DEFAULT '{}',
					output_data JSONB DEFAULT '{}',
					error_details JSONB DEFAULT '{}',
					logs JSONB DEFAULT '[]',
					
					-- Retry information
					retry_count INTEGER DEFAULT 0,
					max_retries INTEGER DEFAULT 3,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Foreign key constraints
					FOREIGN KEY (execution_id) REFERENCES crm_workflow_executions(id) ON DELETE CASCADE,
					FOREIGN KEY (workflow_id) REFERENCES crm_workflows(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_step_order_positive CHECK (step_order > 0),
					CONSTRAINT check_action_execution_time_positive CHECK (execution_time_ms IS NULL OR execution_time_ms >= 0),
					CONSTRAINT check_action_retry_count_positive CHECK (retry_count >= 0),
					CONSTRAINT check_action_max_retries_positive CHECK (max_retries >= 0)
				)
			""")
			
			# Create workflow schedules table
			await connection.execute("""
				CREATE TABLE crm_workflow_schedules (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					workflow_id TEXT NOT NULL,
					
					-- Schedule configuration
					schedule_type TEXT NOT NULL, -- 'cron', 'interval', 'once'
					cron_expression TEXT,
					interval_seconds INTEGER,
					schedule_at TIMESTAMP WITH TIME ZONE,
					
					-- Status and settings
					is_active BOOLEAN DEFAULT true,
					timezone TEXT DEFAULT 'UTC',
					
					-- Execution tracking
					last_execution_at TIMESTAMP WITH TIME ZONE,
					next_execution_at TIMESTAMP WITH TIME ZONE,
					total_executions INTEGER DEFAULT 0,
					failed_executions INTEGER DEFAULT 0,
					
					-- Limits
					max_executions INTEGER,
					expires_at TIMESTAMP WITH TIME ZONE,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					
					-- Foreign key constraints
					FOREIGN KEY (workflow_id) REFERENCES crm_workflows(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_schedule_type CHECK (schedule_type IN ('cron', 'interval', 'once')),
					CONSTRAINT check_interval_seconds_positive CHECK (interval_seconds IS NULL OR interval_seconds > 0),
					CONSTRAINT check_execution_counters_positive CHECK (
						total_executions >= 0 AND 
						failed_executions >= 0 AND 
						failed_executions <= total_executions
					),
					CONSTRAINT check_max_executions_positive CHECK (max_executions IS NULL OR max_executions > 0),
					CONSTRAINT check_schedule_required CHECK (
						(schedule_type = 'cron' AND cron_expression IS NOT NULL) OR
						(schedule_type = 'interval' AND interval_seconds IS NOT NULL) OR
						(schedule_type = 'once' AND schedule_at IS NOT NULL)
					)
				)
			""")
			
			# Create workflow analytics table
			await connection.execute("""
				CREATE TABLE crm_workflow_analytics (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					workflow_id TEXT,
					
					-- Time period
					period_start TIMESTAMP WITH TIME ZONE NOT NULL,
					period_end TIMESTAMP WITH TIME ZONE NOT NULL,
					period_type TEXT NOT NULL, -- 'hour', 'day', 'week', 'month'
					
					-- Execution metrics
					total_executions INTEGER DEFAULT 0,
					successful_executions INTEGER DEFAULT 0,
					failed_executions INTEGER DEFAULT 0,
					cancelled_executions INTEGER DEFAULT 0,
					
					-- Performance metrics
					average_execution_time_ms INTEGER DEFAULT 0,
					min_execution_time_ms INTEGER DEFAULT 0,
					max_execution_time_ms INTEGER DEFAULT 0,
					total_execution_time_ms BIGINT DEFAULT 0,
					
					-- Resource usage
					total_cpu_time_ms BIGINT DEFAULT 0,
					total_memory_usage_mb BIGINT DEFAULT 0,
					total_api_calls INTEGER DEFAULT 0,
					total_data_processed_bytes BIGINT DEFAULT 0,
					
					-- Error analysis
					error_categories JSONB DEFAULT '{}',
					retry_statistics JSONB DEFAULT '{}',
					failure_patterns JSONB DEFAULT '{}',
					
					-- Trigger analysis
					trigger_frequencies JSONB DEFAULT '{}',
					trigger_success_rates JSONB DEFAULT '{}',
					
					-- Action analysis
					action_performance JSONB DEFAULT '{}',
					bottleneck_actions JSONB DEFAULT '{}',
					
					-- Trends
					execution_trend DECIMAL(5,2) DEFAULT 0,
					success_rate_trend DECIMAL(5,2) DEFAULT 0,
					performance_trend DECIMAL(5,2) DEFAULT 0,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					
					-- Foreign key constraints
					FOREIGN KEY (workflow_id) REFERENCES crm_workflows(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_period_valid CHECK (period_end > period_start),
					CONSTRAINT check_period_type CHECK (period_type IN ('hour', 'day', 'week', 'month')),
					CONSTRAINT check_execution_counters_positive CHECK (
						total_executions >= 0 AND 
						successful_executions >= 0 AND 
						failed_executions >= 0 AND 
						cancelled_executions >= 0 AND
						successful_executions + failed_executions + cancelled_executions <= total_executions
					),
					CONSTRAINT check_performance_metrics_positive CHECK (
						average_execution_time_ms >= 0 AND 
						min_execution_time_ms >= 0 AND 
						max_execution_time_ms >= 0 AND 
						total_execution_time_ms >= 0
					),
					CONSTRAINT check_resource_usage_positive CHECK (
						total_cpu_time_ms >= 0 AND 
						total_memory_usage_mb >= 0 AND 
						total_api_calls >= 0 AND 
						total_data_processed_bytes >= 0
					)
				)
			""")
			
			# Create indexes for workflows table
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_tenant 
				ON crm_workflows(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_status 
				ON crm_workflows(status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_active 
				ON crm_workflows(is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_trigger_type 
				ON crm_workflows(trigger_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_category 
				ON crm_workflows(category)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_priority 
				ON crm_workflows(priority)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_created_at 
				ON crm_workflows(created_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_last_execution_at 
				ON crm_workflows(last_execution_at)
			""")
			
			# Create GIN indexes for JSONB fields
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_trigger_conditions 
				ON crm_workflows USING GIN (trigger_conditions)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_actions 
				ON crm_workflows USING GIN (actions)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_tags 
				ON crm_workflows USING GIN (tags)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_metadata 
				ON crm_workflows USING GIN (metadata)
			""")
			
			# Create composite indexes for workflows
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_tenant_active 
				ON crm_workflows(tenant_id, is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_tenant_status 
				ON crm_workflows(tenant_id, status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflows_tenant_trigger_type 
				ON crm_workflows(tenant_id, trigger_type)
			""")
			
			# Create indexes for workflow executions table
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_tenant 
				ON crm_workflow_executions(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_workflow 
				ON crm_workflow_executions(workflow_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_status 
				ON crm_workflow_executions(status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_triggered_by 
				ON crm_workflow_executions(triggered_by)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_record 
				ON crm_workflow_executions(record_id, record_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_started_at 
				ON crm_workflow_executions(started_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_completed_at 
				ON crm_workflow_executions(completed_at)
			""")
			
			# Create GIN indexes for executions JSONB fields
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_trigger_data 
				ON crm_workflow_executions USING GIN (trigger_data)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_execution_results 
				ON crm_workflow_executions USING GIN (execution_results)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_metadata 
				ON crm_workflow_executions USING GIN (metadata)
			""")
			
			# Create composite indexes for executions
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_tenant_workflow 
				ON crm_workflow_executions(tenant_id, workflow_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_workflow_status 
				ON crm_workflow_executions(workflow_id, status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_executions_workflow_started_at 
				ON crm_workflow_executions(workflow_id, started_at)
			""")
			
			# Create indexes for action executions table
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_action_executions_tenant 
				ON crm_workflow_action_executions(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_action_executions_execution 
				ON crm_workflow_action_executions(execution_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_action_executions_workflow 
				ON crm_workflow_action_executions(workflow_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_action_executions_action_type 
				ON crm_workflow_action_executions(action_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_action_executions_status 
				ON crm_workflow_action_executions(status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_action_executions_step_order 
				ON crm_workflow_action_executions(step_order)
			""")
			
			# Create composite indexes for action executions
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_action_executions_execution_step 
				ON crm_workflow_action_executions(execution_id, step_order)
			""")
			
			# Create indexes for schedules table
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_schedules_tenant 
				ON crm_workflow_schedules(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_schedules_workflow 
				ON crm_workflow_schedules(workflow_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_schedules_active 
				ON crm_workflow_schedules(is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_schedules_next_execution_at 
				ON crm_workflow_schedules(next_execution_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_schedules_schedule_type 
				ON crm_workflow_schedules(schedule_type)
			""")
			
			# Create indexes for analytics table
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_analytics_tenant 
				ON crm_workflow_analytics(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_analytics_workflow 
				ON crm_workflow_analytics(workflow_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_analytics_period 
				ON crm_workflow_analytics(period_start, period_end)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_analytics_period_type 
				ON crm_workflow_analytics(period_type)
			""")
			
			# Create composite indexes for analytics
			await connection.execute("""
				CREATE INDEX idx_crm_workflow_analytics_tenant_workflow_period 
				ON crm_workflow_analytics(tenant_id, workflow_id, period_start)
			""")
			
			# Create triggers for updating updated_at timestamps
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_workflows_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_workflows_updated_at
					BEFORE UPDATE ON crm_workflows
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_workflows_updated_at()
			""")
			
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_workflow_schedules_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_workflow_schedules_updated_at
					BEFORE UPDATE ON crm_workflow_schedules
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_workflow_schedules_updated_at()
			""")
			
			# Create view for workflow performance summary
			await connection.execute("""
				CREATE VIEW crm_workflow_performance AS
				SELECT 
					w.id as workflow_id,
					w.tenant_id,
					w.name as workflow_name,
					w.status,
					w.is_active,
					w.trigger_type,
					w.category,
					w.priority,
					
					-- Execution metrics
					COUNT(we.id) as total_executions_last_30_days,
					COUNT(we.id) FILTER (WHERE we.status = 'completed') as successful_executions,
					COUNT(we.id) FILTER (WHERE we.status = 'failed') as failed_executions,
					COUNT(we.id) FILTER (WHERE we.status = 'cancelled') as cancelled_executions,
					
					-- Performance metrics
					COALESCE(AVG(we.execution_time_ms), 0) as average_execution_time,
					COALESCE(MIN(we.execution_time_ms), 0) as min_execution_time,
					COALESCE(MAX(we.execution_time_ms), 0) as max_execution_time,
					
					-- Success rate
					CASE 
						WHEN COUNT(we.id) > 0 THEN
							(COUNT(we.id) FILTER (WHERE we.status = 'completed')::DECIMAL / COUNT(we.id)) * 100
						ELSE 0
					END as success_rate,
					
					-- Recent activity
					MAX(we.started_at) as last_execution_at,
					COUNT(we.id) FILTER (WHERE we.started_at >= NOW() - INTERVAL '24 hours') as executions_last_24h,
					COUNT(we.id) FILTER (WHERE we.started_at >= NOW() - INTERVAL '7 days') as executions_last_7d,
					
					-- Error patterns
					STRING_AGG(DISTINCT we.error_details->>'category', ', ') as error_categories
					
				FROM crm_workflows w
				LEFT JOIN crm_workflow_executions we ON w.id = we.workflow_id 
					AND we.started_at >= NOW() - INTERVAL '30 days'
				GROUP BY w.id, w.tenant_id, w.name, w.status, w.is_active, 
						 w.trigger_type, w.category, w.priority
			""")
			
			# Create view for execution analytics
			await connection.execute("""
				CREATE VIEW crm_workflow_execution_analytics AS
				SELECT 
					we.workflow_id,
					we.tenant_id,
					DATE_TRUNC('day', we.started_at) as execution_date,
					
					-- Daily metrics
					COUNT(*) as total_executions,
					COUNT(*) FILTER (WHERE we.status = 'completed') as successful_executions,
					COUNT(*) FILTER (WHERE we.status = 'failed') as failed_executions,
					AVG(we.execution_time_ms) as avg_execution_time,
					
					-- Resource usage
					AVG(we.cpu_time_ms) as avg_cpu_time,
					AVG(we.memory_usage_mb) as avg_memory_usage,
					SUM(we.api_calls_made) as total_api_calls,
					SUM(we.data_processed_bytes) as total_data_processed,
					
					-- Retry analysis
					AVG(we.retry_count) as avg_retry_count,
					COUNT(*) FILTER (WHERE we.retry_count > 0) as executions_with_retries
					
				FROM crm_workflow_executions we
				WHERE we.started_at >= NOW() - INTERVAL '90 days'
				GROUP BY we.workflow_id, we.tenant_id, DATE_TRUNC('day', we.started_at)
				ORDER BY execution_date DESC
			""")
			
			# Create function for calculating workflow success rate
			await connection.execute("""
				CREATE OR REPLACE FUNCTION calculate_workflow_success_rate(
					workflow_filter TEXT DEFAULT NULL,
					tenant_filter TEXT DEFAULT NULL,
					days_back INTEGER DEFAULT 30
				)
				RETURNS TABLE(
					workflow_id TEXT,
					workflow_name TEXT,
					total_executions BIGINT,
					successful_executions BIGINT,
					success_rate DECIMAL
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						w.id as workflow_id,
						w.name as workflow_name,
						COUNT(we.id) as total_executions,
						COUNT(we.id) FILTER (WHERE we.status = 'completed') as successful_executions,
						CASE 
							WHEN COUNT(we.id) > 0 THEN
								(COUNT(we.id) FILTER (WHERE we.status = 'completed')::DECIMAL / COUNT(we.id)) * 100
							ELSE 0.0
						END as success_rate
					FROM crm_workflows w
					LEFT JOIN crm_workflow_executions we ON w.id = we.workflow_id
						AND we.started_at >= NOW() - (days_back || ' days')::INTERVAL
					WHERE (workflow_filter IS NULL OR w.id = workflow_filter)
					AND (tenant_filter IS NULL OR w.tenant_id = tenant_filter)
					AND w.is_active = true
					GROUP BY w.id, w.name
					ORDER BY success_rate DESC, total_executions DESC;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for identifying slow workflows
			await connection.execute("""
				CREATE OR REPLACE FUNCTION identify_slow_workflows(
					tenant_filter TEXT,
					threshold_ms INTEGER DEFAULT 10000,
					days_back INTEGER DEFAULT 30
				)
				RETURNS TABLE(
					workflow_id TEXT,
					workflow_name TEXT,
					avg_execution_time INTEGER,
					max_execution_time INTEGER,
					slow_executions BIGINT,
					total_executions BIGINT,
					slowness_score DECIMAL
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						w.id as workflow_id,
						w.name as workflow_name,
						AVG(we.execution_time_ms)::INTEGER as avg_execution_time,
						MAX(we.execution_time_ms) as max_execution_time,
						COUNT(we.id) FILTER (WHERE we.execution_time_ms > threshold_ms) as slow_executions,
						COUNT(we.id) as total_executions,
						(AVG(we.execution_time_ms) / threshold_ms * 100)::DECIMAL as slowness_score
					FROM crm_workflows w
					JOIN crm_workflow_executions we ON w.id = we.workflow_id
					WHERE w.tenant_id = tenant_filter
					AND we.started_at >= NOW() - (days_back || ' days')::INTERVAL
					AND we.status = 'completed'
					GROUP BY w.id, w.name
					HAVING AVG(we.execution_time_ms) > threshold_ms
					ORDER BY slowness_score DESC;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Workflow automation structures created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create workflow automation structures: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back workflow automation migration...")
			
			# Drop functions
			await connection.execute("DROP FUNCTION IF EXISTS identify_slow_workflows CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS calculate_workflow_success_rate CASCADE")
			
			# Drop views
			await connection.execute("DROP VIEW IF EXISTS crm_workflow_execution_analytics CASCADE")
			await connection.execute("DROP VIEW IF EXISTS crm_workflow_performance CASCADE")
			
			# Drop triggers and functions
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_workflow_schedules_updated_at ON crm_workflow_schedules")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_workflow_schedules_updated_at CASCADE")
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_workflows_updated_at ON crm_workflows")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_workflows_updated_at CASCADE")
			
			# Drop tables (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_workflow_analytics CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_workflow_schedules CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_workflow_action_executions CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_workflow_executions CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_workflows CASCADE")
			
			# Drop enums
			await connection.execute("DROP TYPE IF EXISTS crm_workflow_action_type CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_workflow_execution_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_workflow_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_workflow_trigger_type CASCADE")
			
			logger.info("✅ Workflow automation migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback workflow automation migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if main tables exist
			tables_exist = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.tables 
				WHERE table_name IN (
					'crm_workflows', 'crm_workflow_executions', 
					'crm_workflow_action_executions', 'crm_workflow_schedules', 
					'crm_workflow_analytics'
				)
			""")
			
			if tables_exist != 5:
				return False
			
			# Check if enums exist
			enum_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_type 
				WHERE typname IN (
					'crm_workflow_trigger_type', 'crm_workflow_status',
					'crm_workflow_execution_status', 'crm_workflow_action_type'
				)
			""")
			
			if enum_count != 4:
				return False
			
			# Check if views exist
			view_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.views 
				WHERE table_name IN ('crm_workflow_performance', 'crm_workflow_execution_analytics')
			""")
			
			if view_count != 2:
				return False
			
			# Check if functions exist
			function_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.routines 
				WHERE routine_name IN (
					'calculate_workflow_success_rate',
					'identify_slow_workflows',
					'update_crm_workflows_updated_at',
					'update_crm_workflow_schedules_updated_at'
				)
			""")
			
			if function_count < 4:
				return False
			
			# Check key indexes
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename IN ('crm_workflows', 'crm_workflow_executions', 'crm_workflow_schedules')
				AND indexname IN (
					'idx_crm_workflows_tenant',
					'idx_crm_workflow_executions_tenant',
					'idx_crm_workflow_schedules_tenant'
				)
			""")
			
			if index_count < 3:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False