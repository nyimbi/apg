"""
APG Customer Relationship Management - Approval Workflows Migration

Database migration to create approval workflow management tables and supporting 
structures for multi-step approval processes, escalation management, and audit trails.

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


class ApprovalWorkflowsMigration(BaseMigration):
	"""Migration for approval workflows functionality"""
	
	def _get_migration_id(self) -> str:
		return "014_approval_workflows"
	
	def _get_version(self) -> str:
		return "014"
	
	def _get_description(self) -> str:
		return "Approval workflows with multi-step processes and escalation management"
	
	def _get_dependencies(self) -> list[str]:
		return ["013_calendar_activity_management"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating approval workflows tables...")
			
			# Create approval types enum
			await conn.execute("""
				CREATE TYPE approval_type AS ENUM (
					'opportunity_discount',
					'opportunity_closure', 
					'contract_approval',
					'pricing_approval',
					'refund_approval',
					'campaign_approval',
					'budget_approval',
					'expense_approval',
					'user_access',
					'data_export',
					'custom'
				);
			""")
			
			# Create approval status enum
			await conn.execute("""
				CREATE TYPE approval_status AS ENUM (
					'pending',
					'in_review',
					'approved', 
					'rejected',
					'cancelled',
					'expired'
				);
			""")
			
			# Create approval action enum
			await conn.execute("""
				CREATE TYPE approval_action AS ENUM (
					'approve',
					'reject',
					'delegate',
					'request_changes',
					'escalate',
					'cancel'
				);
			""")
			
			# Create escalation trigger enum
			await conn.execute("""
				CREATE TYPE escalation_trigger AS ENUM (
					'timeout',
					'manual',
					'automatic',
					'policy_violation'
				);
			""")
			
			# Create approval workflow templates table
			await conn.execute("""
				CREATE TABLE crm_approval_workflow_templates (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					approval_type approval_type NOT NULL,
					is_active BOOLEAN DEFAULT true,
					requires_reason BOOLEAN DEFAULT false,
					auto_approve_threshold DECIMAL(15,2),
					max_timeout_hours INTEGER DEFAULT 168,
					parallel_approval BOOLEAN DEFAULT false,
					approval_steps JSONB NOT NULL DEFAULT '[]',
					escalation_rules JSONB NOT NULL DEFAULT '{}',
					notification_settings JSONB NOT NULL DEFAULT '{}',
					conditions JSONB NOT NULL DEFAULT '{}',
					metadata JSONB NOT NULL DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36),
					updated_by VARCHAR(36)
				);
			""")
			
			# Create approval requests table
			await conn.execute("""
				CREATE TABLE crm_approval_requests (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					template_id VARCHAR(36) NOT NULL REFERENCES crm_approval_workflow_templates(id),
					approval_type approval_type NOT NULL,
					status approval_status DEFAULT 'pending',
					title VARCHAR(500) NOT NULL,
					description TEXT,
					requested_amount DECIMAL(15,2),
					request_reason TEXT,
					requested_by VARCHAR(36) NOT NULL,
					requested_by_name VARCHAR(255) NOT NULL,
					requested_by_email VARCHAR(255) NOT NULL,
					requester_department VARCHAR(100),
					priority INTEGER DEFAULT 3,
					current_step INTEGER DEFAULT 0,
					total_steps INTEGER DEFAULT 0,
					completed_steps INTEGER DEFAULT 0,
					approval_percentage DECIMAL(5,2) DEFAULT 0.00,
					auto_approved BOOLEAN DEFAULT false,
					expires_at TIMESTAMP WITH TIME ZONE,
					submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					started_at TIMESTAMP WITH TIME ZONE,
					completed_at TIMESTAMP WITH TIME ZONE,
					entity_type VARCHAR(100),
					entity_id VARCHAR(36),
					reference_number VARCHAR(100),
					attachments JSONB DEFAULT '[]',
					custom_fields JSONB DEFAULT '{}',
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create approval steps table
			await conn.execute("""
				CREATE TABLE crm_approval_steps (
					id VARCHAR(36) PRIMARY KEY,
					approval_request_id VARCHAR(36) NOT NULL REFERENCES crm_approval_requests(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					step_number INTEGER NOT NULL,
					step_name VARCHAR(255) NOT NULL,
					step_description TEXT,
					approver_id VARCHAR(36),
					approver_name VARCHAR(255),
					approver_email VARCHAR(255),
					approver_role VARCHAR(100),
					is_required BOOLEAN DEFAULT true,
					can_delegate BOOLEAN DEFAULT false,
					timeout_hours INTEGER DEFAULT 72,
					status approval_status DEFAULT 'pending',
					assigned_at TIMESTAMP WITH TIME ZONE,
					responded_at TIMESTAMP WITH TIME ZONE,
					completed_at TIMESTAMP WITH TIME ZONE,
					action_taken approval_action,
					response_notes TEXT,
					rejection_reason TEXT,
					delegated_to VARCHAR(36),
					delegated_at TIMESTAMP WITH TIME ZONE,
					delegation_reason TEXT,
					escalated BOOLEAN DEFAULT false,
					escalated_at TIMESTAMP WITH TIME ZONE,
					escalation_trigger escalation_trigger,
					reminders_sent INTEGER DEFAULT 0,
					last_reminder_at TIMESTAMP WITH TIME ZONE,
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create approval history table
			await conn.execute("""
				CREATE TABLE crm_approval_history (
					id VARCHAR(36) PRIMARY KEY,
					approval_request_id VARCHAR(36) NOT NULL REFERENCES crm_approval_requests(id) ON DELETE CASCADE,
					step_id VARCHAR(36) REFERENCES crm_approval_steps(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					event_type VARCHAR(50) NOT NULL,
					event_description TEXT NOT NULL,
					actor_id VARCHAR(36),
					actor_name VARCHAR(255),
					actor_email VARCHAR(255),
					old_status approval_status,
					new_status approval_status,
					action_taken approval_action,
					notes TEXT,
					ip_address INET,
					user_agent TEXT,
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create approval analytics table
			await conn.execute("""
				CREATE TABLE crm_approval_analytics (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					approval_type approval_type NOT NULL,
					template_id VARCHAR(36) REFERENCES crm_approval_workflow_templates(id),
					date_period DATE NOT NULL,
					period_type VARCHAR(20) NOT NULL DEFAULT 'daily',
					total_requests INTEGER DEFAULT 0,
					approved_requests INTEGER DEFAULT 0,
					rejected_requests INTEGER DEFAULT 0,
					cancelled_requests INTEGER DEFAULT 0,
					expired_requests INTEGER DEFAULT 0,
					avg_approval_time_hours DECIMAL(10,2) DEFAULT 0.00,
					avg_steps_completed DECIMAL(5,2) DEFAULT 0.00,
					escalation_rate DECIMAL(5,2) DEFAULT 0.00,
					auto_approval_rate DECIMAL(5,2) DEFAULT 0.00,
					department_breakdown JSONB DEFAULT '{}',
					approver_performance JSONB DEFAULT '{}',
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_approval_templates_tenant ON crm_approval_workflow_templates(tenant_id);")
			await conn.execute("CREATE INDEX idx_approval_templates_type ON crm_approval_workflow_templates(approval_type);")
			await conn.execute("CREATE INDEX idx_approval_templates_active ON crm_approval_workflow_templates(is_active);")
			
			await conn.execute("CREATE INDEX idx_approval_requests_tenant ON crm_approval_requests(tenant_id);")
			await conn.execute("CREATE INDEX idx_approval_requests_status ON crm_approval_requests(status);")
			await conn.execute("CREATE INDEX idx_approval_requests_type ON crm_approval_requests(approval_type);")
			await conn.execute("CREATE INDEX idx_approval_requests_requester ON crm_approval_requests(requested_by);")
			await conn.execute("CREATE INDEX idx_approval_requests_date ON crm_approval_requests(submitted_at);")
			
			await conn.execute("CREATE INDEX idx_approval_steps_request ON crm_approval_steps(approval_request_id);")
			await conn.execute("CREATE INDEX idx_approval_steps_approver ON crm_approval_steps(approver_id);")
			await conn.execute("CREATE INDEX idx_approval_steps_status ON crm_approval_steps(status);")
			await conn.execute("CREATE INDEX idx_approval_steps_assigned ON crm_approval_steps(assigned_at);")
			
			await conn.execute("CREATE INDEX idx_approval_history_request ON crm_approval_history(approval_request_id);")
			await conn.execute("CREATE INDEX idx_approval_history_date ON crm_approval_history(created_at);")
			await conn.execute("CREATE INDEX idx_approval_history_actor ON crm_approval_history(actor_id);")
			
			await conn.execute("CREATE INDEX idx_approval_analytics_tenant ON crm_approval_analytics(tenant_id);")
			await conn.execute("CREATE INDEX idx_approval_analytics_period ON crm_approval_analytics(date_period);")
			await conn.execute("CREATE INDEX idx_approval_analytics_type ON crm_approval_analytics(approval_type);")
			
			logger.info("✅ Approval workflows tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create approval workflows tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping approval workflows tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_approval_analytics CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_approval_history CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_approval_steps CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_approval_requests CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_approval_workflow_templates CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS escalation_trigger CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS approval_action CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS approval_status CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS approval_type CASCADE;")
			
			logger.info("✅ Approval workflows tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop approval workflows tables: {str(e)}")
			raise