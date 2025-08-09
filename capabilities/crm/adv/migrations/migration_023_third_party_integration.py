"""
APG Customer Relationship Management - Third-Party Integration Migration

Database migration to create third-party integration tables and supporting 
structures for connector management, field mapping, synchronization, and monitoring.

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


class ThirdPartyIntegrationMigration(BaseMigration):
	"""Migration for third-party integration functionality"""
	
	def _get_migration_id(self) -> str:
		return "023_third_party_integration"
	
	def _get_version(self) -> str:
		return "023"
	
	def _get_description(self) -> str:
		return "Third-party integration framework with connectors and sync management"
	
	def _get_dependencies(self) -> list[str]:
		return ["022_webhook_management"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating third-party integration tables...")
			
			# Create integration type enum
			await conn.execute("""
				CREATE TYPE integration_type AS ENUM (
					'salesforce',
					'hubspot',
					'pipedrive',
					'zoho',
					'zapier',
					'webhook',
					'rest_api',
					'graphql',
					'soap',
					'ftp',
					'email',
					'database',
					'custom'
				);
			""")
			
			# Create authentication type enum
			await conn.execute("""
				CREATE TYPE authentication_type AS ENUM (
					'oauth2',
					'api_key',
					'basic_auth',
					'bearer_token',
					'jwt',
					'hmac',
					'custom',
					'none'
				);
			""")
			
			# Create sync direction enum
			await conn.execute("""
				CREATE TYPE sync_direction AS ENUM (
					'bidirectional',
					'inbound',
					'outbound'
				);
			""")
			
			# Create sync status enum
			await conn.execute("""
				CREATE TYPE sync_status AS ENUM (
					'active',
					'paused',
					'error',
					'stopped'
				);
			""")
			
			# Create data operation enum
			await conn.execute("""
				CREATE TYPE data_operation AS ENUM (
					'create',
					'update',
					'delete',
					'upsert',
					'read'
				);
			""")
			
			# Create integration connectors table
			await conn.execute("""
				CREATE TABLE crm_integration_connectors (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					connector_name VARCHAR(255) NOT NULL,
					description TEXT,
					integration_type integration_type NOT NULL,
					platform_name VARCHAR(100) NOT NULL,
					platform_version VARCHAR(50) DEFAULT 'v1',
					base_url VARCHAR(1000) NOT NULL,
					authentication_type authentication_type NOT NULL,
					authentication_config JSONB NOT NULL DEFAULT '{}',
					connection_timeout INTEGER DEFAULT 30,
					request_timeout INTEGER DEFAULT 60,
					max_retries INTEGER DEFAULT 3,
					retry_delay_seconds INTEGER DEFAULT 5,
					rate_limit_config JSONB DEFAULT '{}',
					supported_operations JSONB DEFAULT '[]',
					supported_entities JSONB DEFAULT '[]',
					custom_headers JSONB DEFAULT '{}',
					webhook_config JSONB DEFAULT '{}',
					batch_config JSONB DEFAULT '{}',
					transformation_rules JSONB DEFAULT '{}',
					is_active BOOLEAN DEFAULT true,
					last_sync_at TIMESTAMP WITH TIME ZONE,
					last_success_at TIMESTAMP WITH TIME ZONE,
					last_failure_at TIMESTAMP WITH TIME ZONE,
					last_failure_reason TEXT,
					connection_status VARCHAR(50) DEFAULT 'unknown',
					tags JSONB DEFAULT '[]',
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create field mappings table
			await conn.execute("""
				CREATE TABLE crm_field_mappings (
					id VARCHAR(36) PRIMARY KEY,
					connector_id VARCHAR(36) NOT NULL REFERENCES crm_integration_connectors(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					mapping_name VARCHAR(255) NOT NULL,
					source_entity VARCHAR(100) NOT NULL,
					target_entity VARCHAR(100) NOT NULL,
					field_mappings JSONB NOT NULL DEFAULT '[]',
					transformation_functions JSONB DEFAULT '{}',
					validation_rules JSONB DEFAULT '{}',
					default_values JSONB DEFAULT '{}',
					sync_direction sync_direction NOT NULL,
					conflict_resolution VARCHAR(50) DEFAULT 'target_wins',
					batch_size INTEGER DEFAULT 100,
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create sync configurations table
			await conn.execute("""
				CREATE TABLE crm_sync_configurations (
					id VARCHAR(36) PRIMARY KEY,
					connector_id VARCHAR(36) NOT NULL REFERENCES crm_integration_connectors(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					sync_name VARCHAR(255) NOT NULL,
					description TEXT,
					sync_frequency VARCHAR(50) DEFAULT 'manual',
					schedule_config JSONB DEFAULT '{}',
					timezone VARCHAR(50) DEFAULT 'UTC',
					entity_filters JSONB DEFAULT '{}',
					field_filters JSONB DEFAULT '[]',
					date_range_config JSONB DEFAULT '{}',
					batch_size INTEGER DEFAULT 100,
					max_concurrent_batches INTEGER DEFAULT 5,
					enable_deduplication BOOLEAN DEFAULT true,
					deduplication_fields JSONB DEFAULT '[]',
					error_handling JSONB DEFAULT '{}',
					retry_config JSONB DEFAULT '{}',
					sync_status sync_status DEFAULT 'paused',
					last_sync_at TIMESTAMP WITH TIME ZONE,
					next_sync_at TIMESTAMP WITH TIME ZONE,
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create sync executions table
			await conn.execute("""
				CREATE TABLE crm_sync_executions (
					id VARCHAR(36) PRIMARY KEY,
					sync_config_id VARCHAR(36) NOT NULL REFERENCES crm_sync_configurations(id) ON DELETE CASCADE,
					connector_id VARCHAR(36) NOT NULL REFERENCES crm_integration_connectors(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					execution_type VARCHAR(50) DEFAULT 'scheduled',
					trigger_source VARCHAR(100),
					status VARCHAR(50) DEFAULT 'running',
					total_records INTEGER DEFAULT 0,
					processed_records INTEGER DEFAULT 0,
					successful_records INTEGER DEFAULT 0,
					failed_records INTEGER DEFAULT 0,
					skipped_records INTEGER DEFAULT 0,
					started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					completed_at TIMESTAMP WITH TIME ZONE,
					duration_seconds DECIMAL(10,3),
					throughput_records_per_second DECIMAL(10,3),
					summary JSONB DEFAULT '{}',
					error_details JSONB DEFAULT '[]',
					warnings JSONB DEFAULT '[]',
					affected_entities JSONB DEFAULT '[]',
					sync_statistics JSONB DEFAULT '{}'
				);
			""")
			
			# Create integration logs table
			await conn.execute("""
				CREATE TABLE crm_integration_logs (
					id VARCHAR(36) PRIMARY KEY,
					connector_id VARCHAR(36) REFERENCES crm_integration_connectors(id) ON DELETE CASCADE,
					sync_execution_id VARCHAR(36) REFERENCES crm_sync_executions(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					log_level VARCHAR(20) NOT NULL,
					message TEXT NOT NULL,
					operation VARCHAR(50),
					entity_type VARCHAR(100),
					entity_id VARCHAR(36),
					request_data JSONB DEFAULT '{}',
					response_data JSONB DEFAULT '{}',
					response_status INTEGER,
					response_time_ms DECIMAL(10,3),
					error_code VARCHAR(50),
					error_message TEXT,
					stack_trace TEXT,
					metadata JSONB DEFAULT '{}',
					timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create data transformation rules table
			await conn.execute("""
				CREATE TABLE crm_data_transformation_rules (
					id VARCHAR(36) PRIMARY KEY,
					connector_id VARCHAR(36) NOT NULL REFERENCES crm_integration_connectors(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					rule_name VARCHAR(255) NOT NULL,
					description TEXT,
					source_entity VARCHAR(100) NOT NULL,
					target_entity VARCHAR(100) NOT NULL,
					transformation_type VARCHAR(50) NOT NULL,
					transformation_config JSONB NOT NULL DEFAULT '{}',
					conditions JSONB DEFAULT '[]',
					priority INTEGER DEFAULT 100,
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create integration metrics table
			await conn.execute("""
				CREATE TABLE crm_integration_metrics (
					id VARCHAR(36) PRIMARY KEY,
					connector_id VARCHAR(36) REFERENCES crm_integration_connectors(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					metric_date DATE NOT NULL,
					metric_hour INTEGER DEFAULT 0,
					total_sync_executions INTEGER DEFAULT 0,
					successful_sync_executions INTEGER DEFAULT 0,
					failed_sync_executions INTEGER DEFAULT 0,
					total_records_processed INTEGER DEFAULT 0,
					successful_records INTEGER DEFAULT 0,
					failed_records INTEGER DEFAULT 0,
					avg_sync_duration_seconds DECIMAL(10,3) DEFAULT 0.000,
					avg_throughput_records_per_second DECIMAL(10,3) DEFAULT 0.000,
					data_volume_bytes BIGINT DEFAULT 0,
					api_calls_made INTEGER DEFAULT 0,
					rate_limit_hits INTEGER DEFAULT 0,
					authentication_failures INTEGER DEFAULT 0,
					connection_failures INTEGER DEFAULT 0,
					transformation_errors INTEGER DEFAULT 0,
					entity_breakdown JSONB DEFAULT '{}',
					operation_breakdown JSONB DEFAULT '{}',
					error_breakdown JSONB DEFAULT '{}',
					performance_metrics JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(tenant_id, connector_id, metric_date, metric_hour)
				);
			""")
			
			# Create sync conflicts table
			await conn.execute("""
				CREATE TABLE crm_sync_conflicts (
					id VARCHAR(36) PRIMARY KEY,
					sync_execution_id VARCHAR(36) NOT NULL REFERENCES crm_sync_executions(id) ON DELETE CASCADE,
					connector_id VARCHAR(36) NOT NULL REFERENCES crm_integration_connectors(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					entity_type VARCHAR(100) NOT NULL,
					entity_id VARCHAR(36) NOT NULL,
					source_system VARCHAR(100) NOT NULL,
					target_system VARCHAR(100) NOT NULL,
					conflict_type VARCHAR(50) NOT NULL,
					conflict_field VARCHAR(100),
					source_value JSONB,
					target_value JSONB,
					suggested_resolution VARCHAR(100),
					resolution_strategy VARCHAR(50),
					resolved BOOLEAN DEFAULT false,
					resolved_by VARCHAR(36),
					resolved_at TIMESTAMP WITH TIME ZONE,
					resolution_notes TEXT,
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create OAuth tokens table for secure token storage
			await conn.execute("""
				CREATE TABLE crm_oauth_tokens (
					id VARCHAR(36) PRIMARY KEY,
					connector_id VARCHAR(36) NOT NULL REFERENCES crm_integration_connectors(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					token_type VARCHAR(50) NOT NULL,
					access_token_hash VARCHAR(128) NOT NULL,
					refresh_token_hash VARCHAR(128),
					token_scope TEXT,
					expires_at TIMESTAMP WITH TIME ZONE,
					refresh_expires_at TIMESTAMP WITH TIME ZONE,
					last_refreshed_at TIMESTAMP WITH TIME ZONE,
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(connector_id, token_type)
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_integration_connectors_tenant ON crm_integration_connectors(tenant_id);")
			await conn.execute("CREATE INDEX idx_integration_connectors_type ON crm_integration_connectors(integration_type);")
			await conn.execute("CREATE INDEX idx_integration_connectors_active ON crm_integration_connectors(is_active);")
			await conn.execute("CREATE INDEX idx_integration_connectors_status ON crm_integration_connectors(connection_status);")
			await conn.execute("CREATE INDEX idx_integration_connectors_platform ON crm_integration_connectors(platform_name);")
			
			await conn.execute("CREATE INDEX idx_field_mappings_connector ON crm_field_mappings(connector_id);")
			await conn.execute("CREATE INDEX idx_field_mappings_tenant ON crm_field_mappings(tenant_id);")
			await conn.execute("CREATE INDEX idx_field_mappings_entities ON crm_field_mappings(source_entity, target_entity);")
			await conn.execute("CREATE INDEX idx_field_mappings_direction ON crm_field_mappings(sync_direction);")
			await conn.execute("CREATE INDEX idx_field_mappings_active ON crm_field_mappings(is_active);")
			
			await conn.execute("CREATE INDEX idx_sync_configurations_connector ON crm_sync_configurations(connector_id);")
			await conn.execute("CREATE INDEX idx_sync_configurations_tenant ON crm_sync_configurations(tenant_id);")
			await conn.execute("CREATE INDEX idx_sync_configurations_status ON crm_sync_configurations(sync_status);")
			await conn.execute("CREATE INDEX idx_sync_configurations_frequency ON crm_sync_configurations(sync_frequency);")
			await conn.execute("CREATE INDEX idx_sync_configurations_next_sync ON crm_sync_configurations(next_sync_at);")
			await conn.execute("CREATE INDEX idx_sync_configurations_active ON crm_sync_configurations(is_active);")
			
			await conn.execute("CREATE INDEX idx_sync_executions_config ON crm_sync_executions(sync_config_id);")
			await conn.execute("CREATE INDEX idx_sync_executions_connector ON crm_sync_executions(connector_id);")
			await conn.execute("CREATE INDEX idx_sync_executions_tenant ON crm_sync_executions(tenant_id);")
			await conn.execute("CREATE INDEX idx_sync_executions_status ON crm_sync_executions(status);")
			await conn.execute("CREATE INDEX idx_sync_executions_started ON crm_sync_executions(started_at);")
			await conn.execute("CREATE INDEX idx_sync_executions_type ON crm_sync_executions(execution_type);")
			
			await conn.execute("CREATE INDEX idx_integration_logs_connector ON crm_integration_logs(connector_id);")
			await conn.execute("CREATE INDEX idx_integration_logs_execution ON crm_integration_logs(sync_execution_id);")
			await conn.execute("CREATE INDEX idx_integration_logs_tenant ON crm_integration_logs(tenant_id);")
			await conn.execute("CREATE INDEX idx_integration_logs_level ON crm_integration_logs(log_level);")
			await conn.execute("CREATE INDEX idx_integration_logs_timestamp ON crm_integration_logs(timestamp);")
			await conn.execute("CREATE INDEX idx_integration_logs_operation ON crm_integration_logs(operation);")
			await conn.execute("CREATE INDEX idx_integration_logs_entity ON crm_integration_logs(entity_type, entity_id);")
			
			await conn.execute("CREATE INDEX idx_transformation_rules_connector ON crm_data_transformation_rules(connector_id);")
			await conn.execute("CREATE INDEX idx_transformation_rules_tenant ON crm_data_transformation_rules(tenant_id);")
			await conn.execute("CREATE INDEX idx_transformation_rules_entities ON crm_data_transformation_rules(source_entity, target_entity);")
			await conn.execute("CREATE INDEX idx_transformation_rules_type ON crm_data_transformation_rules(transformation_type);")
			await conn.execute("CREATE INDEX idx_transformation_rules_priority ON crm_data_transformation_rules(priority DESC);")
			await conn.execute("CREATE INDEX idx_transformation_rules_active ON crm_data_transformation_rules(is_active);")
			
			await conn.execute("CREATE INDEX idx_integration_metrics_connector ON crm_integration_metrics(connector_id);")
			await conn.execute("CREATE INDEX idx_integration_metrics_tenant ON crm_integration_metrics(tenant_id);")
			await conn.execute("CREATE INDEX idx_integration_metrics_date ON crm_integration_metrics(metric_date);")
			await conn.execute("CREATE INDEX idx_integration_metrics_hour ON crm_integration_metrics(metric_hour);")
			
			await conn.execute("CREATE INDEX idx_sync_conflicts_execution ON crm_sync_conflicts(sync_execution_id);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_connector ON crm_sync_conflicts(connector_id);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_tenant ON crm_sync_conflicts(tenant_id);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_entity ON crm_sync_conflicts(entity_type, entity_id);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_resolved ON crm_sync_conflicts(resolved);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_type ON crm_sync_conflicts(conflict_type);")
			
			await conn.execute("CREATE INDEX idx_oauth_tokens_connector ON crm_oauth_tokens(connector_id);")
			await conn.execute("CREATE INDEX idx_oauth_tokens_tenant ON crm_oauth_tokens(tenant_id);")
			await conn.execute("CREATE INDEX idx_oauth_tokens_type ON crm_oauth_tokens(token_type);")
			await conn.execute("CREATE INDEX idx_oauth_tokens_expires ON crm_oauth_tokens(expires_at);")
			await conn.execute("CREATE INDEX idx_oauth_tokens_active ON crm_oauth_tokens(is_active);")
			
			# Insert sample integration connectors
			await conn.execute("""
				INSERT INTO crm_integration_connectors (
					id, tenant_id, connector_name, description, integration_type,
					platform_name, platform_version, base_url, authentication_type,
					authentication_config, supported_operations, supported_entities,
					custom_headers, connection_status, created_by
				) VALUES 
				(
					'connector_salesforce_demo',
					'system',
					'Salesforce Production',
					'Main Salesforce integration for production CRM data',
					'salesforce',
					'Salesforce',
					'v58.0',
					'https://mycompany.my.salesforce.com',
					'oauth2',
					'{"client_id": "demo_client_id", "client_secret": "demo_client_secret", "scope": "api refresh_token"}',
					'["create", "read", "update", "delete"]',
					'["Contact", "Lead", "Opportunity", "Account"]',
					'{"Sforce-Call-Options": "client=APG-CRM"}',
					'disconnected',
					'system'
				),
				(
					'connector_hubspot_demo',
					'system',
					'HubSpot Marketing',
					'HubSpot integration for marketing automation and lead management',
					'hubspot',
					'HubSpot',
					'v3',
					'https://api.hubapi.com',
					'api_key',
					'{"api_key": "demo_api_key", "key_header": "Authorization"}',
					'["create", "read", "update"]',
					'["contact", "company", "deal", "ticket"]',
					'{"Content-Type": "application/json"}',
					'disconnected',
					'system'
				),
				(
					'connector_zapier_webhook',
					'system',
					'Zapier Webhook Integration',
					'Generic webhook integration for Zapier automation',
					'webhook',
					'Zapier',
					'v1',
					'https://hooks.zapier.com/hooks/catch',
					'none',
					'{}',
					'["create"]',
					'["contact", "lead", "opportunity"]',
					'{"Content-Type": "application/json"}',
					'unknown',
					'system'
				)
			""")
			
			# Insert sample field mappings
			await conn.execute("""
				INSERT INTO crm_field_mappings (
					id, connector_id, tenant_id, mapping_name, source_entity,
					target_entity, field_mappings, sync_direction, created_by
				) VALUES 
				(
					'mapping_sf_contact',
					'connector_salesforce_demo',
					'system',
					'Salesforce Contact Mapping',
					'Contact',
					'crm_contacts',
					'[
						{"source_field": "FirstName", "target_field": "first_name", "transformation": "trim", "required": true},
						{"source_field": "LastName", "target_field": "last_name", "transformation": "trim", "required": true},
						{"source_field": "Email", "target_field": "email", "transformation": "email_normalize", "required": false},
						{"source_field": "Phone", "target_field": "phone", "transformation": "phone_normalize", "required": false},
						{"source_field": "AccountId", "target_field": "account_id", "transformation": null, "required": false}
					]',
					'bidirectional',
					'system'
				),
				(
					'mapping_hubspot_deal',
					'connector_hubspot_demo',
					'system',
					'HubSpot Deal Mapping',
					'deal',
					'crm_opportunities',
					'[
						{"source_field": "dealname", "target_field": "name", "transformation": "trim", "required": true},
						{"source_field": "amount", "target_field": "value", "transformation": "currency_normalize", "required": false},
						{"source_field": "dealstage", "target_field": "stage", "transformation": null, "required": true},
						{"source_field": "closedate", "target_field": "close_date", "transformation": "date_iso", "required": false},
						{"source_field": "pipeline", "target_field": "pipeline_id", "transformation": null, "required": false}
					]',
					'inbound',
					'system'
				)
			""")
			
			# Insert sample sync configurations
			await conn.execute("""
				INSERT INTO crm_sync_configurations (
					id, connector_id, tenant_id, sync_name, description,
					sync_frequency, schedule_config, batch_size, sync_status, created_by
				) VALUES 
				(
					'sync_salesforce_contacts',
					'connector_salesforce_demo',
					'system',
					'Salesforce Contact Sync',
					'Bidirectional sync of contacts between Salesforce and APG CRM',
					'hourly',
					'{"hour_interval": 1, "start_time": "00:00", "end_time": "23:59"}',
					50,
					'paused',
					'system'
				),
				(
					'sync_hubspot_deals',
					'connector_hubspot_demo',
					'system',
					'HubSpot Deal Import',
					'Import deals from HubSpot to APG CRM opportunities',
					'daily',
					'{"time": "02:00", "weekdays": [1,2,3,4,5]}',
					100,
					'paused',
					'system'
				)
			""")
			
			logger.info("✅ Third-party integration tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create third-party integration tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping third-party integration tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_oauth_tokens CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_sync_conflicts CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_integration_metrics CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_data_transformation_rules CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_integration_logs CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_sync_executions CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_sync_configurations CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_field_mappings CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_integration_connectors CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS data_operation CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS sync_status CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS sync_direction CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS authentication_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS integration_type CASCADE;")
			
			logger.info("✅ Third-party integration tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop third-party integration tables: {str(e)}")
			raise