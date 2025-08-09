"""
APG Customer Relationship Management - API Versioning Migration

Database migration to create API versioning and deprecation management tables
for comprehensive version lifecycle management, client tracking, and migration support.

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


class APIVersioningMigration(BaseMigration):
	"""Migration for API versioning and deprecation management"""
	
	def _get_migration_id(self) -> str:
		return "025_api_versioning"
	
	def _get_version(self) -> str:
		return "025"
	
	def _get_description(self) -> str:
		return "API versioning and deprecation management with client tracking"
	
	def _get_dependencies(self) -> list[str]:
		return ["024_realtime_sync"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating API versioning and deprecation tables...")
			
			# Create API version status enum
			await conn.execute("""
				CREATE TYPE api_version_status AS ENUM (
					'development',
					'beta',
					'stable',
					'deprecated',
					'sunset',
					'retired'
				);
			""")
			
			# Create deprecation severity enum
			await conn.execute("""
				CREATE TYPE deprecation_severity AS ENUM (
					'low',
					'medium',
					'high',
					'critical'
				);
			""")
			
			# Create versioning strategy enum
			await conn.execute("""
				CREATE TYPE versioning_strategy AS ENUM (
					'url_path',
					'header',
					'query_parameter',
					'content_type',
					'custom'
				);
			""")
			
			# Create migration complexity enum
			await conn.execute("""
				CREATE TYPE migration_complexity AS ENUM (
					'trivial',
					'simple',
					'moderate',
					'complex',
					'breaking'
				);
			""")
			
			# Create API versions table
			await conn.execute("""
				CREATE TABLE crm_api_versions (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					version_number VARCHAR(20) NOT NULL,
					version_name VARCHAR(100),
					status api_version_status DEFAULT 'development',
					is_default BOOLEAN DEFAULT false,
					release_date DATE,
					deprecation_date DATE,
					sunset_date DATE,
					retirement_date DATE,
					supported_endpoints JSONB DEFAULT '[]',
					deprecated_endpoints JSONB DEFAULT '[]',
					breaking_changes JSONB DEFAULT '[]',
					feature_flags JSONB DEFAULT '{}',
					documentation_url VARCHAR(1000),
					changelog JSONB DEFAULT '[]',
					migration_guide_url VARCHAR(1000),
					compatibility_matrix JSONB DEFAULT '{}',
					performance_benchmarks JSONB DEFAULT '{}',
					security_updates JSONB DEFAULT '[]',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36),
					metadata JSONB DEFAULT '{}',
					UNIQUE(tenant_id, version_number)
				);
			""")
			
			# Create deprecation notices table
			await conn.execute("""
				CREATE TABLE crm_deprecation_notices (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					version_id VARCHAR(36) NOT NULL REFERENCES crm_api_versions(id) ON DELETE CASCADE,
					endpoint_path VARCHAR(500) NOT NULL,
					http_method VARCHAR(10) DEFAULT 'GET',
					severity deprecation_severity DEFAULT 'medium',
					deprecation_reason TEXT NOT NULL,
					replacement_endpoint VARCHAR(500),
					replacement_version VARCHAR(20),
					migration_instructions TEXT,
					automated_migration_available BOOLEAN DEFAULT false,
					grace_period_days INTEGER DEFAULT 90,
					client_impact_assessment JSONB DEFAULT '{}',
					business_justification TEXT,
					technical_debt_score INTEGER DEFAULT 0,
					usage_analytics JSONB DEFAULT '{}',
					communication_plan JSONB DEFAULT '{}',
					scheduled_notifications JSONB DEFAULT '[]',
					exemption_criteria JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create version migrations table
			await conn.execute("""
				CREATE TABLE crm_version_migrations (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					from_version_id VARCHAR(36) NOT NULL REFERENCES crm_api_versions(id) ON DELETE CASCADE,
					to_version_id VARCHAR(36) NOT NULL REFERENCES crm_api_versions(id) ON DELETE CASCADE,
					migration_name VARCHAR(255) NOT NULL,
					complexity migration_complexity DEFAULT 'moderate',
					is_breaking_change BOOLEAN DEFAULT false,
					is_automated BOOLEAN DEFAULT false,
					field_mappings JSONB DEFAULT '{}',
					transformation_rules JSONB DEFAULT '{}',
					validation_rules JSONB DEFAULT '{}',
					data_migration_required BOOLEAN DEFAULT false,
					rollback_strategy JSONB DEFAULT '{}',
					testing_requirements JSONB DEFAULT '{}',
					prerequisites JSONB DEFAULT '[]',
					success_criteria JSONB DEFAULT '{}',
					risk_assessment JSONB DEFAULT '{}',
					estimated_effort_hours INTEGER DEFAULT 0,
					migration_script TEXT,
					documentation TEXT,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create client version usage table
			await conn.execute("""
				CREATE TABLE crm_client_version_usage (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					version_id VARCHAR(36) NOT NULL REFERENCES crm_api_versions(id) ON DELETE CASCADE,
					client_id VARCHAR(200) NOT NULL,
					client_name VARCHAR(255),
					client_type VARCHAR(50) DEFAULT 'external',
					user_agent VARCHAR(1000),
					ip_address INET,
					first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					request_count BIGINT DEFAULT 1,
					error_count BIGINT DEFAULT 0,
					avg_response_time_ms DECIMAL(10,3) DEFAULT 0.000,
					endpoints_used JSONB DEFAULT '[]',
					feature_usage JSONB DEFAULT '{}',
					compliance_status VARCHAR(50) DEFAULT 'compliant',
					migration_status VARCHAR(50) DEFAULT 'not_started',
					migration_timeline JSONB DEFAULT '{}',
					support_contact JSONB DEFAULT '{}',
					business_criticality VARCHAR(20) DEFAULT 'medium',
					migration_blockers JSONB DEFAULT '[]',
					usage_patterns JSONB DEFAULT '{}',
					performance_metrics JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					metadata JSONB DEFAULT '{}',
					UNIQUE(tenant_id, version_id, client_id)
				);
			""")
			
			# Create version configuration table
			await conn.execute("""
				CREATE TABLE crm_version_configurations (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					version_id VARCHAR(36) NOT NULL REFERENCES crm_api_versions(id) ON DELETE CASCADE,
					strategy versioning_strategy DEFAULT 'url_path',
					strategy_config JSONB DEFAULT '{}',
					header_name VARCHAR(100),
					query_parameter_name VARCHAR(100),
					content_type_pattern VARCHAR(200),
					custom_extraction_logic TEXT,
					fallback_version VARCHAR(20),
					version_priority INTEGER DEFAULT 100,
					load_balancing_config JSONB DEFAULT '{}',
					caching_config JSONB DEFAULT '{}',
					rate_limiting_config JSONB DEFAULT '{}',
					monitoring_config JSONB DEFAULT '{}',
					feature_toggles JSONB DEFAULT '{}',
					environment_specific JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create version analytics table
			await conn.execute("""
				CREATE TABLE crm_version_analytics (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					version_id VARCHAR(36) NOT NULL REFERENCES crm_api_versions(id) ON DELETE CASCADE,
					metric_date DATE NOT NULL,
					metric_hour INTEGER DEFAULT 0,
					total_requests BIGINT DEFAULT 0,
					successful_requests BIGINT DEFAULT 0,
					failed_requests BIGINT DEFAULT 0,
					error_4xx_count BIGINT DEFAULT 0,
					error_5xx_count BIGINT DEFAULT 0,
					unique_clients INTEGER DEFAULT 0,
					new_clients INTEGER DEFAULT 0,
					returning_clients INTEGER DEFAULT 0,
					avg_response_time_ms DECIMAL(10,3) DEFAULT 0.000,
					p95_response_time_ms DECIMAL(10,3) DEFAULT 0.000,
					p99_response_time_ms DECIMAL(10,3) DEFAULT 0.000,
					throughput_requests_per_second DECIMAL(10,3) DEFAULT 0.000,
					data_transfer_bytes BIGINT DEFAULT 0,
					endpoint_usage_breakdown JSONB DEFAULT '{}',
					client_type_breakdown JSONB DEFAULT '{}',
					geographic_distribution JSONB DEFAULT '{}',
					feature_usage_stats JSONB DEFAULT '{}',
					performance_percentiles JSONB DEFAULT '{}',
					error_analysis JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(tenant_id, version_id, metric_date, metric_hour)
				);
			""")
			
			# Create migration execution logs table
			await conn.execute("""
				CREATE TABLE crm_migration_execution_logs (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					migration_id VARCHAR(36) NOT NULL REFERENCES crm_version_migrations(id) ON DELETE CASCADE,
					client_id VARCHAR(200) NOT NULL,
					execution_status VARCHAR(50) DEFAULT 'pending',
					started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					completed_at TIMESTAMP WITH TIME ZONE,
					duration_seconds INTEGER,
					steps_completed INTEGER DEFAULT 0,
					total_steps INTEGER DEFAULT 0,
					success_rate DECIMAL(5,2) DEFAULT 0.00,
					errors_encountered JSONB DEFAULT '[]',
					warnings_generated JSONB DEFAULT '[]',
					rollback_required BOOLEAN DEFAULT false,
					rollback_executed BOOLEAN DEFAULT false,
					validation_results JSONB DEFAULT '{}',
					performance_impact JSONB DEFAULT '{}',
					data_integrity_check JSONB DEFAULT '{}',
					executed_by VARCHAR(36),
					automation_metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create version access control table
			await conn.execute("""
				CREATE TABLE crm_version_access_control (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					version_id VARCHAR(36) NOT NULL REFERENCES crm_api_versions(id) ON DELETE CASCADE,
					client_id VARCHAR(200),
					client_group VARCHAR(100),
					access_level VARCHAR(50) DEFAULT 'read_only',
					allowed_endpoints JSONB DEFAULT '[]',
					blocked_endpoints JSONB DEFAULT '[]',
					rate_limit_override JSONB DEFAULT '{}',
					ip_whitelist JSONB DEFAULT '[]',
					ip_blacklist JSONB DEFAULT '[]',
					time_restrictions JSONB DEFAULT '{}',
					feature_permissions JSONB DEFAULT '{}',
					data_access_scope JSONB DEFAULT '{}',
					audit_requirements JSONB DEFAULT '{}',
					compliance_tags JSONB DEFAULT '[]',
					is_active BOOLEAN DEFAULT true,
					expires_at TIMESTAMP WITH TIME ZONE,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create version health monitoring table
			await conn.execute("""
				CREATE TABLE crm_version_health_monitoring (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					version_id VARCHAR(36) NOT NULL REFERENCES crm_api_versions(id) ON DELETE CASCADE,
					check_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					health_score DECIMAL(5,2) DEFAULT 100.00,
					availability_percentage DECIMAL(5,2) DEFAULT 100.00,
					response_time_health DECIMAL(5,2) DEFAULT 100.00,
					error_rate_health DECIMAL(5,2) DEFAULT 100.00,
					throughput_health DECIMAL(5,2) DEFAULT 100.00,
					resource_utilization JSONB DEFAULT '{}',
					active_alerts JSONB DEFAULT '[]',
					performance_trends JSONB DEFAULT '{}',
					capacity_metrics JSONB DEFAULT '{}',
					dependency_health JSONB DEFAULT '{}',
					sla_compliance JSONB DEFAULT '{}',
					incident_history JSONB DEFAULT '[]',
					maintenance_windows JSONB DEFAULT '[]',
					automated_remediation JSONB DEFAULT '{}',
					escalation_triggers JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create comprehensive indexes for performance
			await conn.execute("CREATE INDEX idx_api_versions_tenant ON crm_api_versions(tenant_id);")
			await conn.execute("CREATE INDEX idx_api_versions_status ON crm_api_versions(status);")
			await conn.execute("CREATE INDEX idx_api_versions_default ON crm_api_versions(is_default);")
			await conn.execute("CREATE INDEX idx_api_versions_release ON crm_api_versions(release_date);")
			await conn.execute("CREATE INDEX idx_api_versions_deprecation ON crm_api_versions(deprecation_date);")
			await conn.execute("CREATE INDEX idx_api_versions_sunset ON crm_api_versions(sunset_date);")
			await conn.execute("CREATE INDEX idx_api_versions_endpoints ON crm_api_versions USING GIN(supported_endpoints);")
			
			await conn.execute("CREATE INDEX idx_deprecation_notices_tenant ON crm_deprecation_notices(tenant_id);")
			await conn.execute("CREATE INDEX idx_deprecation_notices_version ON crm_deprecation_notices(version_id);")
			await conn.execute("CREATE INDEX idx_deprecation_notices_endpoint ON crm_deprecation_notices(endpoint_path);")
			await conn.execute("CREATE INDEX idx_deprecation_notices_severity ON crm_deprecation_notices(severity);")
			await conn.execute("CREATE INDEX idx_deprecation_notices_method ON crm_deprecation_notices(http_method);")
			await conn.execute("CREATE INDEX idx_deprecation_notices_automated ON crm_deprecation_notices(automated_migration_available);")
			
			await conn.execute("CREATE INDEX idx_version_migrations_tenant ON crm_version_migrations(tenant_id);")
			await conn.execute("CREATE INDEX idx_version_migrations_from ON crm_version_migrations(from_version_id);")
			await conn.execute("CREATE INDEX idx_version_migrations_to ON crm_version_migrations(to_version_id);")
			await conn.execute("CREATE INDEX idx_version_migrations_complexity ON crm_version_migrations(complexity);")
			await conn.execute("CREATE INDEX idx_version_migrations_breaking ON crm_version_migrations(is_breaking_change);")
			await conn.execute("CREATE INDEX idx_version_migrations_automated ON crm_version_migrations(is_automated);")
			
			await conn.execute("CREATE INDEX idx_client_usage_tenant ON crm_client_version_usage(tenant_id);")
			await conn.execute("CREATE INDEX idx_client_usage_version ON crm_client_version_usage(version_id);")
			await conn.execute("CREATE INDEX idx_client_usage_client ON crm_client_version_usage(client_id);")
			await conn.execute("CREATE INDEX idx_client_usage_type ON crm_client_version_usage(client_type);")
			await conn.execute("CREATE INDEX idx_client_usage_last_seen ON crm_client_version_usage(last_seen);")
			await conn.execute("CREATE INDEX idx_client_usage_migration_status ON crm_client_version_usage(migration_status);")
			await conn.execute("CREATE INDEX idx_client_usage_criticality ON crm_client_version_usage(business_criticality);")
			await conn.execute("CREATE INDEX idx_client_usage_compliance ON crm_client_version_usage(compliance_status);")
			
			await conn.execute("CREATE INDEX idx_version_configs_tenant ON crm_version_configurations(tenant_id);")
			await conn.execute("CREATE INDEX idx_version_configs_version ON crm_version_configurations(version_id);")
			await conn.execute("CREATE INDEX idx_version_configs_strategy ON crm_version_configurations(strategy);")
			await conn.execute("CREATE INDEX idx_version_configs_priority ON crm_version_configurations(version_priority DESC);")
			
			await conn.execute("CREATE INDEX idx_version_analytics_tenant ON crm_version_analytics(tenant_id);")
			await conn.execute("CREATE INDEX idx_version_analytics_version ON crm_version_analytics(version_id);")
			await conn.execute("CREATE INDEX idx_version_analytics_date ON crm_version_analytics(metric_date);")
			await conn.execute("CREATE INDEX idx_version_analytics_hour ON crm_version_analytics(metric_hour);")
			await conn.execute("CREATE INDEX idx_version_analytics_requests ON crm_version_analytics(total_requests DESC);")
			
			await conn.execute("CREATE INDEX idx_migration_logs_tenant ON crm_migration_execution_logs(tenant_id);")
			await conn.execute("CREATE INDEX idx_migration_logs_migration ON crm_migration_execution_logs(migration_id);")
			await conn.execute("CREATE INDEX idx_migration_logs_client ON crm_migration_execution_logs(client_id);")
			await conn.execute("CREATE INDEX idx_migration_logs_status ON crm_migration_execution_logs(execution_status);")
			await conn.execute("CREATE INDEX idx_migration_logs_started ON crm_migration_execution_logs(started_at);")
			
			await conn.execute("CREATE INDEX idx_access_control_tenant ON crm_version_access_control(tenant_id);")
			await conn.execute("CREATE INDEX idx_access_control_version ON crm_version_access_control(version_id);")
			await conn.execute("CREATE INDEX idx_access_control_client ON crm_version_access_control(client_id);")
			await conn.execute("CREATE INDEX idx_access_control_group ON crm_version_access_control(client_group);")
			await conn.execute("CREATE INDEX idx_access_control_active ON crm_version_access_control(is_active);")
			await conn.execute("CREATE INDEX idx_access_control_expires ON crm_version_access_control(expires_at);")
			
			await conn.execute("CREATE INDEX idx_health_monitoring_tenant ON crm_version_health_monitoring(tenant_id);")
			await conn.execute("CREATE INDEX idx_health_monitoring_version ON crm_version_health_monitoring(version_id);")
			await conn.execute("CREATE INDEX idx_health_monitoring_timestamp ON crm_version_health_monitoring(check_timestamp);")
			await conn.execute("CREATE INDEX idx_health_monitoring_score ON crm_version_health_monitoring(health_score);")
			await conn.execute("CREATE INDEX idx_health_monitoring_availability ON crm_version_health_monitoring(availability_percentage);")
			
			# Insert sample API versions
			await conn.execute("""
				INSERT INTO crm_api_versions (
					id, tenant_id, version_number, version_name, status,
					is_default, release_date, supported_endpoints, created_by
				) VALUES 
				(
					'api_version_v1',
					'system',
					'1.0',
					'Initial API Version',
					'stable',
					false,
					'2024-01-01',
					'["/api/v1/contacts", "/api/v1/accounts", "/api/v1/opportunities", "/api/v1/activities"]',
					'system'
				),
				(
					'api_version_v2',
					'system',
					'2.0',
					'Enhanced CRM API',
					'stable',
					true,
					'2024-06-01',
					'["/api/v2/contacts", "/api/v2/accounts", "/api/v2/opportunities", "/api/v2/activities", "/api/v2/analytics", "/api/v2/integrations"]',
					'system'
				),
				(
					'api_version_v3',
					'system',
					'3.0',
					'Next Generation API',
					'beta',
					false,
					'2024-12-01',
					'["/api/v3/contacts", "/api/v3/accounts", "/api/v3/opportunities", "/api/v3/activities", "/api/v3/analytics", "/api/v3/integrations", "/api/v3/ai-insights"]',
					'system'
				)
			""")
			
			# Insert sample deprecation notices
			await conn.execute("""
				INSERT INTO crm_deprecation_notices (
					id, tenant_id, version_id, endpoint_path, http_method,
					severity, deprecation_reason, replacement_endpoint, replacement_version,
					grace_period_days, created_by
				) VALUES 
				(
					'deprecation_v1_contacts',
					'system',
					'api_version_v1',
					'/api/v1/contacts',
					'GET',
					'medium',
					'Legacy contact endpoint lacks advanced filtering and pagination capabilities',
					'/api/v2/contacts',
					'2.0',
					180,
					'system'
				),
				(
					'deprecation_v1_bulk_import',
					'system',
					'api_version_v1',
					'/api/v1/bulk-import',
					'POST',
					'high',
					'Bulk import endpoint has security vulnerabilities and performance issues',
					'/api/v2/batch-operations',
					'2.0',
					90,
					'system'
				)
			""")
			
			# Insert sample version configurations
			await conn.execute("""
				INSERT INTO crm_version_configurations (
					id, tenant_id, version_id, strategy, header_name,
					version_priority, created_by
				) VALUES 
				(
					'config_v1_header',
					'system',
					'api_version_v1',
					'header',
					'API-Version',
					50,
					'system'
				),
				(
					'config_v2_path',
					'system',
					'api_version_v2',
					'url_path',
					100,
					'system'
				),
				(
					'config_v3_path',
					'system',
					'api_version_v3',
					'url_path',
					150,
					'system'
				)
			""")
			
			logger.info("✅ API versioning and deprecation tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create API versioning tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping API versioning and deprecation tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_version_health_monitoring CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_version_access_control CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_migration_execution_logs CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_version_analytics CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_version_configurations CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_client_version_usage CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_version_migrations CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_deprecation_notices CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_api_versions CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS migration_complexity CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS versioning_strategy CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS deprecation_severity CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS api_version_status CASCADE;")
			
			logger.info("✅ API versioning and deprecation tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop API versioning tables: {str(e)}")
			raise