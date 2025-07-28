"""
APG Customer Relationship Management - Webhook Management Migration

Database migration to create webhook management tables and supporting 
structures for event-driven integrations, delivery tracking, and monitoring.

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


class WebhookManagementMigration(BaseMigration):
	"""Migration for webhook management functionality"""
	
	def _get_migration_id(self) -> str:
		return "022_webhook_management"
	
	def _get_version(self) -> str:
		return "022"
	
	def _get_description(self) -> str:
		return "Webhook management system with event processing and delivery tracking"
	
	def _get_dependencies(self) -> list[str]:
		return ["021_api_gateway"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating webhook management tables...")
			
			# Create event action enum
			await conn.execute("""
				CREATE TYPE webhook_event_action AS ENUM (
					'created',
					'updated',
					'deleted',
					'status_changed',
					'assigned',
					'completed',
					'cancelled',
					'approved',
					'rejected',
					'archived',
					'restored'
				);
			""")
			
			# Create event category enum
			await conn.execute("""
				CREATE TYPE webhook_event_category AS ENUM (
					'contact',
					'lead',
					'opportunity',
					'account',
					'activity',
					'campaign',
					'user',
					'system',
					'integration',
					'workflow',
					'approval',
					'report'
				);
			""")
			
			# Create delivery status enum
			await conn.execute("""
				CREATE TYPE webhook_delivery_status AS ENUM (
					'pending',
					'delivering',
					'delivered',
					'failed',
					'retrying',
					'cancelled',
					'expired'
				);
			""")
			
			# Create webhook events table
			await conn.execute("""
				CREATE TABLE crm_webhook_events (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					event_type VARCHAR(100) NOT NULL,
					event_category webhook_event_category NOT NULL,
					event_action webhook_event_action NOT NULL,
					entity_id VARCHAR(36) NOT NULL,
					entity_type VARCHAR(50) NOT NULL,
					entity_data JSONB NOT NULL DEFAULT '{}',
					previous_data JSONB,
					change_summary JSONB DEFAULT '{}',
					metadata JSONB DEFAULT '{}',
					correlation_id VARCHAR(36),
					parent_event_id VARCHAR(36) REFERENCES crm_webhook_events(id),
					user_id VARCHAR(36),
					user_agent TEXT,
					ip_address INET,
					source_system VARCHAR(100),
					batch_id VARCHAR(36),
					priority INTEGER DEFAULT 100,
					expires_at TIMESTAMP WITH TIME ZONE,
					timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					processed BOOLEAN DEFAULT false,
					processed_at TIMESTAMP WITH TIME ZONE,
					processing_errors JSONB DEFAULT '[]',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create webhook subscriptions table
			await conn.execute("""
				CREATE TABLE crm_webhook_subscriptions (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					subscription_name VARCHAR(255) NOT NULL,
					webhook_endpoint_id VARCHAR(36) NOT NULL REFERENCES crm_api_webhooks(id) ON DELETE CASCADE,
					event_filters JSONB DEFAULT '{}',
					field_filters JSONB DEFAULT '[]',
					entity_filters JSONB DEFAULT '{}',
					user_filters JSONB DEFAULT '[]',
					batch_config JSONB DEFAULT '{}',
					rate_limit_config JSONB DEFAULT '{}',
					priority INTEGER DEFAULT 100,
					delivery_mode VARCHAR(20) DEFAULT 'immediate',
					buffer_time_seconds INTEGER DEFAULT 0,
					max_buffer_size INTEGER DEFAULT 1,
					deduplication_enabled BOOLEAN DEFAULT false,
					deduplication_window_seconds INTEGER DEFAULT 300,
					transformation_rules JSONB DEFAULT '{}',
					validation_rules JSONB DEFAULT '{}',
					error_handling JSONB DEFAULT '{"on_error": "retry", "max_errors": 10}',
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL
				);
			""")
			
			# Enhance webhook deliveries table with additional fields
			await conn.execute("""
				ALTER TABLE crm_webhook_deliveries ADD COLUMN IF NOT EXISTS 
					delivery_status webhook_delivery_status DEFAULT 'pending';
			""")
			
			await conn.execute("""
				ALTER TABLE crm_webhook_deliveries ADD COLUMN IF NOT EXISTS 
					subscription_id VARCHAR(36) REFERENCES crm_webhook_subscriptions(id);
			""")
			
			await conn.execute("""
				ALTER TABLE crm_webhook_deliveries ADD COLUMN IF NOT EXISTS 
					batch_id VARCHAR(36);
			""")
			
			await conn.execute("""
				ALTER TABLE crm_webhook_deliveries ADD COLUMN IF NOT EXISTS 
					priority INTEGER DEFAULT 100;
			""")
			
			await conn.execute("""
				ALTER TABLE crm_webhook_deliveries ADD COLUMN IF NOT EXISTS 
					max_retries INTEGER DEFAULT 3;
			""")
			
			await conn.execute("""
				ALTER TABLE crm_webhook_deliveries ADD COLUMN IF NOT EXISTS 
					backoff_multiplier DECIMAL(3,2) DEFAULT 2.0;
			""")
			
			await conn.execute("""
				ALTER TABLE crm_webhook_deliveries ADD COLUMN IF NOT EXISTS 
					signature_validation BOOLEAN DEFAULT true;
			""")
			
			await conn.execute("""
				ALTER TABLE crm_webhook_deliveries ADD COLUMN IF NOT EXISTS 
					circuit_breaker_triggered BOOLEAN DEFAULT false;
			""")
			
			# Create webhook event subscriptions junction table
			await conn.execute("""
				CREATE TABLE crm_webhook_event_subscriptions (
					id VARCHAR(36) PRIMARY KEY,
					event_id VARCHAR(36) NOT NULL REFERENCES crm_webhook_events(id) ON DELETE CASCADE,
					subscription_id VARCHAR(36) NOT NULL REFERENCES crm_webhook_subscriptions(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					matched_filters JSONB DEFAULT '{}',
					transformation_applied BOOLEAN DEFAULT false,
					delivery_scheduled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					delivery_priority INTEGER DEFAULT 100,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(event_id, subscription_id)
				);
			""")
			
			# Create webhook templates table
			await conn.execute("""
				CREATE TABLE crm_webhook_templates (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					template_name VARCHAR(255) NOT NULL,
					description TEXT,
					event_categories JSONB NOT NULL DEFAULT '[]',
					template_type VARCHAR(50) DEFAULT 'jinja2',
					template_content TEXT NOT NULL,
					sample_data JSONB DEFAULT '{}',
					validation_schema JSONB DEFAULT '{}',
					variables JSONB DEFAULT '[]',
					helper_functions JSONB DEFAULT '{}',
					is_public BOOLEAN DEFAULT false,
					usage_count INTEGER DEFAULT 0,
					rating DECIMAL(3,2) DEFAULT 0.00,
					tags JSONB DEFAULT '[]',
					version VARCHAR(20) DEFAULT '1.0',
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL
				);
			""")
			
			# Create webhook delivery batches table
			await conn.execute("""
				CREATE TABLE crm_webhook_delivery_batches (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					subscription_id VARCHAR(36) NOT NULL REFERENCES crm_webhook_subscriptions(id) ON DELETE CASCADE,
					batch_name VARCHAR(255),
					batch_size INTEGER NOT NULL,
					total_events INTEGER DEFAULT 0,
					successful_deliveries INTEGER DEFAULT 0,
					failed_deliveries INTEGER DEFAULT 0,
					status VARCHAR(20) DEFAULT 'pending',
					batch_payload JSONB NOT NULL DEFAULT '{}',
					aggregation_rules JSONB DEFAULT '{}',
					started_at TIMESTAMP WITH TIME ZONE,
					completed_at TIMESTAMP WITH TIME ZONE,
					total_processing_time_ms DECIMAL(10,3),
					error_summary JSONB DEFAULT '{}',
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create webhook metrics aggregation table
			await conn.execute("""
				CREATE TABLE crm_webhook_metrics (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					webhook_id VARCHAR(36) REFERENCES crm_api_webhooks(id) ON DELETE CASCADE,
					metric_date DATE NOT NULL,
					metric_hour INTEGER DEFAULT 0,
					total_events INTEGER DEFAULT 0,
					successful_deliveries INTEGER DEFAULT 0,
					failed_deliveries INTEGER DEFAULT 0,
					retried_deliveries INTEGER DEFAULT 0,
					avg_delivery_time_ms DECIMAL(10,3) DEFAULT 0.000,
					p95_delivery_time_ms DECIMAL(10,3) DEFAULT 0.000,
					total_payload_bytes BIGINT DEFAULT 0,
					unique_event_types INTEGER DEFAULT 0,
					circuit_breaker_triggers INTEGER DEFAULT 0,
					rate_limit_hits INTEGER DEFAULT 0,
					event_type_breakdown JSONB DEFAULT '{}',
					status_code_breakdown JSONB DEFAULT '{}',
					error_breakdown JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(tenant_id, webhook_id, metric_date, metric_hour)
				);
			""")
			
			# Create webhook security audit table
			await conn.execute("""
				CREATE TABLE crm_webhook_security_audit (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					webhook_id VARCHAR(36) REFERENCES crm_api_webhooks(id) ON DELETE CASCADE,
					delivery_id VARCHAR(36) REFERENCES crm_webhook_deliveries(id) ON DELETE CASCADE,
					security_event VARCHAR(50) NOT NULL,
					severity_level VARCHAR(20) NOT NULL,
					description TEXT NOT NULL,
					client_ip INET,
					user_agent TEXT,
					request_headers JSONB DEFAULT '{}',
					response_headers JSONB DEFAULT '{}',
					signature_verified BOOLEAN,
					rate_limit_exceeded BOOLEAN DEFAULT false,
					suspicious_activity BOOLEAN DEFAULT false,
					blocked BOOLEAN DEFAULT false,
					remediation_action VARCHAR(100),
					investigation_notes TEXT,
					resolved BOOLEAN DEFAULT false,
					resolved_by VARCHAR(36),
					resolved_at TIMESTAMP WITH TIME ZONE,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_webhook_events_tenant ON crm_webhook_events(tenant_id);")
			await conn.execute("CREATE INDEX idx_webhook_events_type ON crm_webhook_events(event_type);")
			await conn.execute("CREATE INDEX idx_webhook_events_category ON crm_webhook_events(event_category);")
			await conn.execute("CREATE INDEX idx_webhook_events_entity ON crm_webhook_events(entity_type, entity_id);")
			await conn.execute("CREATE INDEX idx_webhook_events_timestamp ON crm_webhook_events(timestamp);")
			await conn.execute("CREATE INDEX idx_webhook_events_processed ON crm_webhook_events(processed);")
			await conn.execute("CREATE INDEX idx_webhook_events_correlation ON crm_webhook_events(correlation_id);")
			await conn.execute("CREATE INDEX idx_webhook_events_user ON crm_webhook_events(user_id);")
			await conn.execute("CREATE INDEX idx_webhook_events_batch ON crm_webhook_events(batch_id);")
			await conn.execute("CREATE INDEX idx_webhook_events_expires ON crm_webhook_events(expires_at);")
			
			await conn.execute("CREATE INDEX idx_webhook_subscriptions_tenant ON crm_webhook_subscriptions(tenant_id);")
			await conn.execute("CREATE INDEX idx_webhook_subscriptions_webhook ON crm_webhook_subscriptions(webhook_endpoint_id);")
			await conn.execute("CREATE INDEX idx_webhook_subscriptions_active ON crm_webhook_subscriptions(is_active);")
			await conn.execute("CREATE INDEX idx_webhook_subscriptions_mode ON crm_webhook_subscriptions(delivery_mode);")
			await conn.execute("CREATE INDEX idx_webhook_subscriptions_priority ON crm_webhook_subscriptions(priority DESC);")
			
			await conn.execute("CREATE INDEX idx_webhook_deliveries_status ON crm_webhook_deliveries(delivery_status);")
			await conn.execute("CREATE INDEX idx_webhook_deliveries_subscription ON crm_webhook_deliveries(subscription_id);")
			await conn.execute("CREATE INDEX idx_webhook_deliveries_batch ON crm_webhook_deliveries(batch_id);")
			await conn.execute("CREATE INDEX idx_webhook_deliveries_priority ON crm_webhook_deliveries(priority DESC);")
			await conn.execute("CREATE INDEX idx_webhook_deliveries_circuit ON crm_webhook_deliveries(circuit_breaker_triggered);")
			
			await conn.execute("CREATE INDEX idx_webhook_event_subscriptions_event ON crm_webhook_event_subscriptions(event_id);")
			await conn.execute("CREATE INDEX idx_webhook_event_subscriptions_subscription ON crm_webhook_event_subscriptions(subscription_id);")
			await conn.execute("CREATE INDEX idx_webhook_event_subscriptions_tenant ON crm_webhook_event_subscriptions(tenant_id);")
			await conn.execute("CREATE INDEX idx_webhook_event_subscriptions_scheduled ON crm_webhook_event_subscriptions(delivery_scheduled_at);")
			
			await conn.execute("CREATE INDEX idx_webhook_templates_tenant ON crm_webhook_templates(tenant_id);")
			await conn.execute("CREATE INDEX idx_webhook_templates_categories ON crm_webhook_templates USING GIN(event_categories);")
			await conn.execute("CREATE INDEX idx_webhook_templates_public ON crm_webhook_templates(is_public);")
			await conn.execute("CREATE INDEX idx_webhook_templates_active ON crm_webhook_templates(is_active);")
			await conn.execute("CREATE INDEX idx_webhook_templates_usage ON crm_webhook_templates(usage_count DESC);")
			
			await conn.execute("CREATE INDEX idx_webhook_delivery_batches_subscription ON crm_webhook_delivery_batches(subscription_id);")
			await conn.execute("CREATE INDEX idx_webhook_delivery_batches_tenant ON crm_webhook_delivery_batches(tenant_id);")
			await conn.execute("CREATE INDEX idx_webhook_delivery_batches_status ON crm_webhook_delivery_batches(status);")
			await conn.execute("CREATE INDEX idx_webhook_delivery_batches_created ON crm_webhook_delivery_batches(created_at);")
			
			await conn.execute("CREATE INDEX idx_webhook_metrics_tenant ON crm_webhook_metrics(tenant_id);")
			await conn.execute("CREATE INDEX idx_webhook_metrics_webhook ON crm_webhook_metrics(webhook_id);")
			await conn.execute("CREATE INDEX idx_webhook_metrics_date ON crm_webhook_metrics(metric_date);")
			await conn.execute("CREATE INDEX idx_webhook_metrics_hour ON crm_webhook_metrics(metric_hour);")
			
			await conn.execute("CREATE INDEX idx_webhook_security_audit_tenant ON crm_webhook_security_audit(tenant_id);")
			await conn.execute("CREATE INDEX idx_webhook_security_audit_webhook ON crm_webhook_security_audit(webhook_id);")
			await conn.execute("CREATE INDEX idx_webhook_security_audit_delivery ON crm_webhook_security_audit(delivery_id);")
			await conn.execute("CREATE INDEX idx_webhook_security_audit_event ON crm_webhook_security_audit(security_event);")
			await conn.execute("CREATE INDEX idx_webhook_security_audit_severity ON crm_webhook_security_audit(severity_level);")
			await conn.execute("CREATE INDEX idx_webhook_security_audit_blocked ON crm_webhook_security_audit(blocked);")
			await conn.execute("CREATE INDEX idx_webhook_security_audit_resolved ON crm_webhook_security_audit(resolved);")
			await conn.execute("CREATE INDEX idx_webhook_security_audit_created ON crm_webhook_security_audit(created_at);")
			
			# Insert default webhook templates
			await conn.execute("""
				INSERT INTO crm_webhook_templates (
					id, tenant_id, template_name, description, event_categories,
					template_content, sample_data, is_public, created_by
				) VALUES 
				(
					'tpl_webhook_basic',
					'system',
					'Basic Event Template',
					'Simple webhook payload template for all event types',
					'["contact", "lead", "opportunity", "account"]',
					'{"event": {"id": "{{ event.id }}", "type": "{{ event.event_type }}", "timestamp": "{{ event.timestamp }}"}, "data": {{ event.entity_data | tojson }}, "tenant": "{{ event.tenant_id }}"}',
					'{"event": {"id": "evt_123", "type": "contact.created", "timestamp": "2025-01-28T10:00:00Z"}, "data": {"name": "John Doe", "email": "john@example.com"}, "tenant": "tenant_123"}',
					true,
					'system'
				),
				(
					'tpl_webhook_slack',
					'system',
					'Slack Notification Template',
					'Slack-compatible webhook template for team notifications',
					'["lead", "opportunity"]',
					'{"text": "{{ event.event_action | title }} {{ event.entity_type }}: {{ event.entity_data.name or event.entity_data.title }}", "attachments": [{"color": "{% if event.event_action == \"created\" %}good{% elif event.event_action == \"deleted\" %}danger{% else %}warning{% endif %}", "fields": [{"title": "ID", "value": "{{ event.entity_id }}", "short": true}, {"title": "User", "value": "{{ event.user_id or \"System\" }}", "short": true}]}]}',
					'{"text": "Created Lead: Acme Corp", "attachments": [{"color": "good", "fields": [{"title": "ID", "value": "lead_123", "short": true}, {"title": "User", "value": "user_456", "short": true}]}]}',
					true,
					'system'
				),
				(
					'tpl_webhook_discord',
					'system',
					'Discord Notification Template',
					'Discord-compatible webhook template with rich embeds',
					'["opportunity", "account"]',
					'{"embeds": [{"title": "{{ event.event_action | title }} {{ event.entity_type | title }}", "description": "{{ event.entity_data.name or event.entity_data.title }}", "color": {% if event.event_action == "created" %}3066993{% elif event.event_action == "deleted" %}15158332{% else %}15105570{% endif %}, "fields": [{"name": "ID", "value": "{{ event.entity_id }}", "inline": true}, {"name": "Timestamp", "value": "{{ event.timestamp }}", "inline": true}], "footer": {"text": "APG CRM"}}]}',
					'{"embeds": [{"title": "Created Opportunity", "description": "Big Deal Opportunity", "color": 3066993, "fields": [{"name": "ID", "value": "opp_789", "inline": true}, {"name": "Timestamp", "value": "2025-01-28T10:00:00Z", "inline": true}], "footer": {"text": "APG CRM"}}]}',
					true,
					'system'
				),
				(
					'tpl_webhook_teams',
					'system',
					'Microsoft Teams Template',
					'Microsoft Teams-compatible webhook template with action cards',
					'["contact", "lead", "opportunity", "account", "activity"]',
					'{"@type": "MessageCard", "@context": "http://schema.org/extensions", "summary": "{{ event.event_action | title }} {{ event.entity_type | title }}", "themeColor": "{% if event.event_action == \"created\" %}00FF00{% elif event.event_action == \"deleted\" %}FF0000{% else %}FFA500{% endif %}", "sections": [{"activityTitle": "{{ event.event_action | title }} {{ event.entity_type | title }}", "activitySubtitle": "{{ event.entity_data.name or event.entity_data.title or event.entity_id }}", "facts": [{"name": "ID:", "value": "{{ event.entity_id }}"}, {"name": "Timestamp:", "value": "{{ event.timestamp }}"}, {"name": "User:", "value": "{{ event.user_id or \"System\" }}"}]}]}',
					'{"@type": "MessageCard", "@context": "http://schema.org/extensions", "summary": "Created Contact", "themeColor": "00FF00", "sections": [{"activityTitle": "Created Contact", "activitySubtitle": "Jane Smith", "facts": [{"name": "ID:", "value": "contact_456"}, {"name": "Timestamp:", "value": "2025-01-28T10:00:00Z"}, {"name": "User:", "value": "user_123"}]}]}',
					true,
					'system'
				)
			""")
			
			# Insert sample webhook subscriptions for system webhooks
			await conn.execute("""
				INSERT INTO crm_webhook_subscriptions (
					id, tenant_id, subscription_name, webhook_endpoint_id,
					event_filters, delivery_mode, priority, created_by
				) VALUES 
				(
					'sub_system_contact_events',
					'system',
					'System Contact Events',
					'endpoint_contacts_list',
					'{"event_categories": ["contact"], "event_actions": ["created", "updated", "deleted"]}',
					'immediate',
					200,
					'system'
				),
				(
					'sub_system_opportunity_events',
					'system', 
					'System Opportunity Events',
					'endpoint_opportunities_list',
					'{"event_categories": ["opportunity"], "event_actions": ["created", "updated", "status_changed"]}',
					'immediate',
					150,
					'system'
				)
			""")
			
			logger.info("✅ Webhook management tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create webhook management tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping webhook management tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_webhook_security_audit CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_webhook_metrics CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_webhook_delivery_batches CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_webhook_templates CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_webhook_event_subscriptions CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_webhook_subscriptions CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_webhook_events CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS webhook_delivery_status CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS webhook_event_category CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS webhook_event_action CASCADE;")
			
			logger.info("✅ Webhook management tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop webhook management tables: {str(e)}")
			raise