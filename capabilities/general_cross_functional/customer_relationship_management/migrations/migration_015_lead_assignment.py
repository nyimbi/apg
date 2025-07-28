"""
APG Customer Relationship Management - Lead Assignment Migration

Database migration to create lead assignment rule tables and supporting 
structures for intelligent lead routing, workload balancing, and assignment analytics.

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


class LeadAssignmentMigration(BaseMigration):
	"""Migration for lead assignment functionality"""
	
	def _get_migration_id(self) -> str:
		return "015_lead_assignment"
	
	def _get_version(self) -> str:
		return "015"
	
	def _get_description(self) -> str:
		return "Lead assignment rules with intelligent routing and workload balancing"
	
	def _get_dependencies(self) -> list[str]:
		return ["014_approval_workflows"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating lead assignment tables...")
			
			# Create assignment type enum
			await conn.execute("""
				CREATE TYPE assignment_type AS ENUM (
					'round_robin',
					'territory_based',
					'skill_based',
					'workload_balanced',
					'performance_based',
					'company_size',
					'industry_based',
					'lead_score_based',
					'custom_rule'
				);
			""")
			
			# Create assignment status enum
			await conn.execute("""
				CREATE TYPE assignment_status AS ENUM (
					'active',
					'inactive',
					'paused',
					'archived'
				);
			""")
			
			# Create assignment priority enum
			await conn.execute("""
				CREATE TYPE assignment_priority AS ENUM (
					'critical',
					'high',
					'medium',
					'low'
				);
			""")
			
			# Create workload metric enum
			await conn.execute("""
				CREATE TYPE workload_metric AS ENUM (
					'active_leads',
					'open_opportunities',
					'monthly_quota',
					'response_time',
					'conversion_rate',
					'weighted_pipeline'
				);
			""")
			
			# Create lead assignment rules table
			await conn.execute("""
				CREATE TABLE crm_lead_assignment_rules (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					assignment_type assignment_type NOT NULL,
					status assignment_status DEFAULT 'active',
					priority assignment_priority DEFAULT 'medium',
					conditions JSONB NOT NULL DEFAULT '[]',
					targets JSONB NOT NULL DEFAULT '[]',
					round_robin_position INTEGER DEFAULT 0,
					workload_metrics JSONB NOT NULL DEFAULT '[]',
					rebalance_frequency INTEGER DEFAULT 24,
					max_assignments_per_hour INTEGER,
					business_hours_only BOOLEAN DEFAULT false,
					time_zone VARCHAR(50) DEFAULT 'UTC',
					exclude_weekends BOOLEAN DEFAULT false,
					escalation_timeout_hours INTEGER DEFAULT 24,
					total_assignments INTEGER DEFAULT 0,
					successful_assignments INTEGER DEFAULT 0,
					failed_assignments INTEGER DEFAULT 0,
					avg_assignment_time_ms DECIMAL(10,2) DEFAULT 0.00,
					last_assignment_at TIMESTAMP WITH TIME ZONE,
					metadata JSONB NOT NULL DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create lead assignments table
			await conn.execute("""
				CREATE TABLE crm_lead_assignments (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					lead_id VARCHAR(36) NOT NULL,
					rule_id VARCHAR(36) NOT NULL REFERENCES crm_lead_assignment_rules(id),
					assigned_to VARCHAR(36) NOT NULL,
					assigned_to_name VARCHAR(255) NOT NULL,
					assigned_to_email VARCHAR(255) NOT NULL,
					assigned_team VARCHAR(100),
					assigned_territory VARCHAR(100),
					assignment_reason TEXT NOT NULL,
					assignment_score DECIMAL(5,2) DEFAULT 0.00,
					assignment_method assignment_type NOT NULL,
					assignment_duration_ms INTEGER DEFAULT 0,
					lead_score DECIMAL(5,2),
					lead_source VARCHAR(100),
					lead_industry VARCHAR(100),
					lead_company_size VARCHAR(50),
					lead_territory VARCHAR(100),
					is_accepted BOOLEAN DEFAULT false,
					accepted_at TIMESTAMP WITH TIME ZONE,
					is_reassigned BOOLEAN DEFAULT false,
					reassigned_at TIMESTAMP WITH TIME ZONE,
					reassignment_reason TEXT,
					first_contact_at TIMESTAMP WITH TIME ZONE,
					qualified_at TIMESTAMP WITH TIME ZONE,
					converted_at TIMESTAMP WITH TIME ZONE,
					response_time_hours DECIMAL(8,2),
					metadata JSONB NOT NULL DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create assignment analytics table
			await conn.execute("""
				CREATE TABLE crm_assignment_analytics (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					period_start TIMESTAMP WITH TIME ZONE NOT NULL,
					period_end TIMESTAMP WITH TIME ZONE NOT NULL,
					total_assignments INTEGER DEFAULT 0,
					successful_assignments INTEGER DEFAULT 0,
					failed_assignments INTEGER DEFAULT 0,
					avg_assignment_time_ms DECIMAL(10,2) DEFAULT 0.00,
					assignment_acceptance_rate DECIMAL(5,2) DEFAULT 0.00,
					avg_response_time_hours DECIMAL(8,2) DEFAULT 0.00,
					conversion_rate DECIMAL(5,2) DEFAULT 0.00,
					reassignment_rate DECIMAL(5,2) DEFAULT 0.00,
					assignments_by_rule JSONB DEFAULT '{}',
					assignments_by_user JSONB DEFAULT '{}',
					assignments_by_team JSONB DEFAULT '{}',
					assignments_by_territory JSONB DEFAULT '{}',
					user_performance JSONB DEFAULT '{}',
					team_performance JSONB DEFAULT '{}',
					rule_performance JSONB DEFAULT '{}',
					optimization_suggestions JSONB DEFAULT '[]',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create assignment workload tracking table
			await conn.execute("""
				CREATE TABLE crm_assignment_workloads (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					user_id VARCHAR(36) NOT NULL,
					user_name VARCHAR(255) NOT NULL,
					team VARCHAR(100),
					territory VARCHAR(100),
					date_tracked DATE NOT NULL,
					active_leads_count INTEGER DEFAULT 0,
					open_opportunities_count INTEGER DEFAULT 0,
					monthly_assignments INTEGER DEFAULT 0,
					avg_response_time_hours DECIMAL(8,2) DEFAULT 0.00,
					conversion_rate DECIMAL(5,2) DEFAULT 0.00,
					workload_score DECIMAL(8,2) DEFAULT 0.00,
					capacity_utilization DECIMAL(5,2) DEFAULT 0.00,
					performance_score DECIMAL(5,2) DEFAULT 0.00,
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(tenant_id, user_id, date_tracked)
				);
			""")
			
			# Create assignment notification queue table
			await conn.execute("""
				CREATE TABLE crm_assignment_notifications (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					assignment_id VARCHAR(36) NOT NULL REFERENCES crm_lead_assignments(id),
					notification_type VARCHAR(50) NOT NULL,
					recipient_id VARCHAR(36) NOT NULL,
					recipient_email VARCHAR(255) NOT NULL,
					subject VARCHAR(500),
					message TEXT,
					status VARCHAR(20) DEFAULT 'pending',
					sent_at TIMESTAMP WITH TIME ZONE,
					delivery_attempts INTEGER DEFAULT 0,
					last_attempt_at TIMESTAMP WITH TIME ZONE,
					error_message TEXT,
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_assignment_rules_tenant ON crm_lead_assignment_rules(tenant_id);")
			await conn.execute("CREATE INDEX idx_assignment_rules_status ON crm_lead_assignment_rules(status);")
			await conn.execute("CREATE INDEX idx_assignment_rules_priority ON crm_lead_assignment_rules(priority);")
			await conn.execute("CREATE INDEX idx_assignment_rules_type ON crm_lead_assignment_rules(assignment_type);")
			
			await conn.execute("CREATE INDEX idx_lead_assignments_tenant ON crm_lead_assignments(tenant_id);")
			await conn.execute("CREATE INDEX idx_lead_assignments_lead ON crm_lead_assignments(lead_id);")
			await conn.execute("CREATE INDEX idx_lead_assignments_assignee ON crm_lead_assignments(assigned_to);")
			await conn.execute("CREATE INDEX idx_lead_assignments_rule ON crm_lead_assignments(rule_id);")
			await conn.execute("CREATE INDEX idx_lead_assignments_date ON crm_lead_assignments(created_at);")
			await conn.execute("CREATE INDEX idx_lead_assignments_team ON crm_lead_assignments(assigned_team);")
			await conn.execute("CREATE INDEX idx_lead_assignments_territory ON crm_lead_assignments(assigned_territory);")
			
			await conn.execute("CREATE INDEX idx_assignment_analytics_tenant ON crm_assignment_analytics(tenant_id);")
			await conn.execute("CREATE INDEX idx_assignment_analytics_period ON crm_assignment_analytics(period_start, period_end);")
			
			await conn.execute("CREATE INDEX idx_assignment_workloads_tenant ON crm_assignment_workloads(tenant_id);")
			await conn.execute("CREATE INDEX idx_assignment_workloads_user ON crm_assignment_workloads(user_id);")
			await conn.execute("CREATE INDEX idx_assignment_workloads_date ON crm_assignment_workloads(date_tracked);")
			
			await conn.execute("CREATE INDEX idx_assignment_notifications_tenant ON crm_assignment_notifications(tenant_id);")
			await conn.execute("CREATE INDEX idx_assignment_notifications_assignment ON crm_assignment_notifications(assignment_id);")
			await conn.execute("CREATE INDEX idx_assignment_notifications_status ON crm_assignment_notifications(status);")
			
			logger.info("✅ Lead assignment tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create lead assignment tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping lead assignment tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_assignment_notifications CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_assignment_workloads CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_assignment_analytics CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_lead_assignments CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_lead_assignment_rules CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS workload_metric CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS assignment_priority CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS assignment_status CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS assignment_type CASCADE;")
			
			logger.info("✅ Lead assignment tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop lead assignment tables: {str(e)}")
			raise