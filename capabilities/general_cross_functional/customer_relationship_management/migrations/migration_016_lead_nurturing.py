"""
APG Customer Relationship Management - Lead Nurturing Migration

Database migration to create lead nurturing workflow tables and supporting 
structures for automated lead nurturing campaigns with AI-powered personalization.

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


class LeadNurturingMigration(BaseMigration):
	"""Migration for lead nurturing functionality"""
	
	def _get_migration_id(self) -> str:
		return "016_lead_nurturing"
	
	def _get_version(self) -> str:
		return "016"
	
	def _get_description(self) -> str:
		return "Lead nurturing workflows with AI-powered personalization and multi-channel campaigns"
	
	def _get_dependencies(self) -> list[str]:
		return ["015_lead_assignment"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating lead nurturing tables...")
			
			# Create nurturing status enum
			await conn.execute("""
				CREATE TYPE nurturing_status AS ENUM (
					'draft',
					'active',
					'paused',
					'completed',
					'archived'
				);
			""")
			
			# Create trigger type enum
			await conn.execute("""
				CREATE TYPE trigger_type AS ENUM (
					'lead_created',
					'score_threshold',
					'behavior_based',
					'time_based',
					'form_submission',
					'email_interaction',
					'website_activity',
					'manual_trigger',
					'stage_change',
					'inactivity'
				);
			""")
			
			# Create action type enum
			await conn.execute("""
				CREATE TYPE action_type AS ENUM (
					'send_email',
					'send_sms',
					'assign_to_rep',
					'update_score',
					'add_tag',
					'remove_tag',
					'create_task',
					'schedule_call',
					'send_notification',
					'update_field',
					'wait_delay',
					'conditional_split'
				);
			""")
			
			# Create channel type enum
			await conn.execute("""
				CREATE TYPE channel_type AS ENUM (
					'email',
					'sms',
					'phone',
					'social_media',
					'direct_mail',
					'webinar',
					'in_app',
					'push_notification'
				);
			""")
			
			# Create personalization level enum
			await conn.execute("""
				CREATE TYPE personalization_level AS ENUM (
					'basic',
					'standard',
					'advanced',
					'ai_powered'
				);
			""")
			
			# Create nurturing workflows table
			await conn.execute("""
				CREATE TABLE crm_nurturing_workflows (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					status nurturing_status DEFAULT 'draft',
					trigger_type trigger_type NOT NULL,
					trigger_conditions JSONB NOT NULL DEFAULT '[]',
					entry_criteria JSONB NOT NULL DEFAULT '[]',
					exit_criteria JSONB NOT NULL DEFAULT '[]',
					actions JSONB NOT NULL DEFAULT '[]',
					start_action_id VARCHAR(36) NOT NULL,
					max_leads_per_day INTEGER,
					priority INTEGER DEFAULT 5,
					time_zone VARCHAR(50) DEFAULT 'UTC',
					goal_type VARCHAR(100),
					goal_value DECIMAL(15,2),
					conversion_event VARCHAR(100),
					total_enrolled INTEGER DEFAULT 0,
					currently_active INTEGER DEFAULT 0,
					completed_successfully INTEGER DEFAULT 0,
					conversion_rate DECIMAL(5,2) DEFAULT 0.00,
					avg_time_to_conversion_days DECIMAL(10,2) DEFAULT 0.00,
					ai_optimization_enabled BOOLEAN DEFAULT false,
					auto_pause_low_performance BOOLEAN DEFAULT false,
					performance_threshold DECIMAL(5,2) DEFAULT 5.00,
					metadata JSONB NOT NULL DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create nurturing enrollments table
			await conn.execute("""
				CREATE TABLE crm_nurturing_enrollments (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					workflow_id VARCHAR(36) NOT NULL REFERENCES crm_nurturing_workflows(id) ON DELETE CASCADE,
					lead_id VARCHAR(36) NOT NULL,
					enrollment_source VARCHAR(50) NOT NULL,
					enrollment_trigger trigger_type NOT NULL,
					enrolled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					enrolled_by VARCHAR(36),
					current_action_id VARCHAR(36),
					current_step INTEGER DEFAULT 0,
					total_steps INTEGER DEFAULT 0,
					is_active BOOLEAN DEFAULT true,
					is_paused BOOLEAN DEFAULT false,
					paused_at TIMESTAMP WITH TIME ZONE,
					paused_reason TEXT,
					completed_at TIMESTAMP WITH TIME ZONE,
					completion_reason TEXT,
					success BOOLEAN,
					emails_sent INTEGER DEFAULT 0,
					emails_opened INTEGER DEFAULT 0,
					emails_clicked INTEGER DEFAULT 0,
					forms_submitted INTEGER DEFAULT 0,
					meetings_scheduled INTEGER DEFAULT 0,
					engagement_score DECIMAL(8,2) DEFAULT 0.00,
					lead_score_change DECIMAL(8,2) DEFAULT 0.00,
					time_to_conversion_hours DECIMAL(10,2),
					lead_score_at_enrollment DECIMAL(5,2),
					lead_stage_at_enrollment VARCHAR(50),
					lead_source VARCHAR(100),
					lead_industry VARCHAR(100),
					metadata JSONB NOT NULL DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create nurturing executions table
			await conn.execute("""
				CREATE TABLE crm_nurturing_executions (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					enrollment_id VARCHAR(36) NOT NULL REFERENCES crm_nurturing_enrollments(id) ON DELETE CASCADE,
					workflow_id VARCHAR(36) NOT NULL REFERENCES crm_nurturing_workflows(id) ON DELETE CASCADE,
					action_id VARCHAR(36) NOT NULL,
					lead_id VARCHAR(36) NOT NULL,
					scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
					executed_at TIMESTAMP WITH TIME ZONE,
					status VARCHAR(20) DEFAULT 'pending',
					success BOOLEAN,
					result_data JSONB DEFAULT '{}',
					error_message TEXT,
					delivery_status VARCHAR(50),
					opened_at TIMESTAMP WITH TIME ZONE,
					clicked_at TIMESTAMP WITH TIME ZONE,
					responded_at TIMESTAMP WITH TIME ZONE,
					attempt_number INTEGER DEFAULT 1,
					max_attempts INTEGER DEFAULT 3,
					next_retry_at TIMESTAMP WITH TIME ZONE,
					metadata JSONB NOT NULL DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create nurturing analytics table
			await conn.execute("""
				CREATE TABLE crm_nurturing_analytics (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					workflow_id VARCHAR(36) REFERENCES crm_nurturing_workflows(id),
					period_start TIMESTAMP WITH TIME ZONE NOT NULL,
					period_end TIMESTAMP WITH TIME ZONE NOT NULL,
					total_enrollments INTEGER DEFAULT 0,
					active_enrollments INTEGER DEFAULT 0,
					completed_enrollments INTEGER DEFAULT 0,
					conversion_rate DECIMAL(5,2) DEFAULT 0.00,
					emails_sent INTEGER DEFAULT 0,
					email_open_rate DECIMAL(5,2) DEFAULT 0.00,
					email_click_rate DECIMAL(5,2) DEFAULT 0.00,
					form_submission_rate DECIMAL(5,2) DEFAULT 0.00,
					meeting_booking_rate DECIMAL(5,2) DEFAULT 0.00,
					avg_time_to_conversion_days DECIMAL(10,2) DEFAULT 0.00,
					avg_engagement_score DECIMAL(8,2) DEFAULT 0.00,
					lead_score_improvement DECIMAL(8,2) DEFAULT 0.00,
					top_performing_workflows JSONB DEFAULT '[]',
					workflow_performance JSONB DEFAULT '{}',
					action_performance JSONB DEFAULT '{}',
					channel_performance JSONB DEFAULT '{}',
					optimal_send_times JSONB DEFAULT '{}',
					optimization_suggestions JSONB DEFAULT '[]',
					performance_alerts JSONB DEFAULT '[]',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create nurturing templates table
			await conn.execute("""
				CREATE TABLE crm_nurturing_templates (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					template_type VARCHAR(50) NOT NULL,
					channel channel_type NOT NULL,
					subject VARCHAR(500),
					content TEXT NOT NULL,
					personalization_level personalization_level DEFAULT 'standard',
					personalization_fields JSONB DEFAULT '[]',
					dynamic_content JSONB DEFAULT '{}',
					version INTEGER DEFAULT 1,
					is_active BOOLEAN DEFAULT true,
					usage_count INTEGER DEFAULT 0,
					performance_score DECIMAL(5,2) DEFAULT 0.00,
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create nurturing triggers table
			await conn.execute("""
				CREATE TABLE crm_nurturing_triggers (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					workflow_id VARCHAR(36) NOT NULL REFERENCES crm_nurturing_workflows(id) ON DELETE CASCADE,
					trigger_type trigger_type NOT NULL,
					trigger_name VARCHAR(255) NOT NULL,
					conditions JSONB NOT NULL DEFAULT '[]',
					is_active BOOLEAN DEFAULT true,
					fire_count INTEGER DEFAULT 0,
					last_fired_at TIMESTAMP WITH TIME ZONE,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_nurturing_workflows_tenant ON crm_nurturing_workflows(tenant_id);")
			await conn.execute("CREATE INDEX idx_nurturing_workflows_status ON crm_nurturing_workflows(status);")
			await conn.execute("CREATE INDEX idx_nurturing_workflows_trigger ON crm_nurturing_workflows(trigger_type);")
			await conn.execute("CREATE INDEX idx_nurturing_workflows_priority ON crm_nurturing_workflows(priority);")
			
			await conn.execute("CREATE INDEX idx_nurturing_enrollments_tenant ON crm_nurturing_enrollments(tenant_id);")
			await conn.execute("CREATE INDEX idx_nurturing_enrollments_workflow ON crm_nurturing_enrollments(workflow_id);")
			await conn.execute("CREATE INDEX idx_nurturing_enrollments_lead ON crm_nurturing_enrollments(lead_id);")
			await conn.execute("CREATE INDEX idx_nurturing_enrollments_active ON crm_nurturing_enrollments(is_active);")
			await conn.execute("CREATE INDEX idx_nurturing_enrollments_date ON crm_nurturing_enrollments(enrolled_at);")
			
			await conn.execute("CREATE INDEX idx_nurturing_executions_tenant ON crm_nurturing_executions(tenant_id);")
			await conn.execute("CREATE INDEX idx_nurturing_executions_enrollment ON crm_nurturing_executions(enrollment_id);")
			await conn.execute("CREATE INDEX idx_nurturing_executions_workflow ON crm_nurturing_executions(workflow_id);")
			await conn.execute("CREATE INDEX idx_nurturing_executions_lead ON crm_nurturing_executions(lead_id);")
			await conn.execute("CREATE INDEX idx_nurturing_executions_scheduled ON crm_nurturing_executions(scheduled_at);")
			await conn.execute("CREATE INDEX idx_nurturing_executions_status ON crm_nurturing_executions(status);")
			
			await conn.execute("CREATE INDEX idx_nurturing_analytics_tenant ON crm_nurturing_analytics(tenant_id);")
			await conn.execute("CREATE INDEX idx_nurturing_analytics_workflow ON crm_nurturing_analytics(workflow_id);")
			await conn.execute("CREATE INDEX idx_nurturing_analytics_period ON crm_nurturing_analytics(period_start, period_end);")
			
			await conn.execute("CREATE INDEX idx_nurturing_templates_tenant ON crm_nurturing_templates(tenant_id);")
			await conn.execute("CREATE INDEX idx_nurturing_templates_type ON crm_nurturing_templates(template_type);")
			await conn.execute("CREATE INDEX idx_nurturing_templates_channel ON crm_nurturing_templates(channel);")
			await conn.execute("CREATE INDEX idx_nurturing_templates_active ON crm_nurturing_templates(is_active);")
			
			await conn.execute("CREATE INDEX idx_nurturing_triggers_tenant ON crm_nurturing_triggers(tenant_id);")
			await conn.execute("CREATE INDEX idx_nurturing_triggers_workflow ON crm_nurturing_triggers(workflow_id);")
			await conn.execute("CREATE INDEX idx_nurturing_triggers_type ON crm_nurturing_triggers(trigger_type);")
			await conn.execute("CREATE INDEX idx_nurturing_triggers_active ON crm_nurturing_triggers(is_active);")
			
			# Create unique constraints
			await conn.execute("CREATE UNIQUE INDEX idx_nurturing_enrollments_unique ON crm_nurturing_enrollments(workflow_id, lead_id) WHERE is_active = true;")
			
			logger.info("✅ Lead nurturing tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create lead nurturing tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping lead nurturing tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_nurturing_triggers CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_nurturing_templates CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_nurturing_analytics CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_nurturing_executions CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_nurturing_enrollments CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_nurturing_workflows CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS personalization_level CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS channel_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS action_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS trigger_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS nurturing_status CASCADE;")
			
			logger.info("✅ Lead nurturing tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop lead nurturing tables: {str(e)}")
			raise