"""
APG Customer Relationship Management - API Gateway Migration

Database migration to create API gateway tables and supporting 
structures for rate limiting, endpoint management, request tracking, and monitoring.

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


class APIGatewayMigration(BaseMigration):
	"""Migration for API gateway functionality"""
	
	def _get_migration_id(self) -> str:
		return "021_api_gateway"
	
	def _get_version(self) -> str:
		return "021"
	
	def _get_description(self) -> str:
		return "Enterprise API gateway with rate limiting and monitoring"
	
	def _get_dependencies(self) -> list[str]:
		return ["020_performance_benchmarking"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating API gateway tables...")
			
			# Create rate limit type enum
			await conn.execute("""
				CREATE TYPE rate_limit_type AS ENUM (
					'requests_per_minute',
					'requests_per_hour',
					'requests_per_day',
					'bandwidth_per_minute',
					'bandwidth_per_hour',
					'concurrent_requests'
				);
			""")
			
			# Create enforcement action enum
			await conn.execute("""
				CREATE TYPE enforcement_action AS ENUM (
					'reject',
					'delay',
					'throttle',
					'warn',
					'log_only'
				);
			""")
			
			# Create scope type enum
			await conn.execute("""
				CREATE TYPE rate_limit_scope AS ENUM (
					'global',
					'tenant',
					'user',
					'ip',
					'api_key',
					'endpoint'
				);
			""")
			
			# Create HTTP method enum
			await conn.execute("""
				CREATE TYPE http_method AS ENUM (
					'GET',
					'POST',
					'PUT',
					'DELETE',
					'PATCH',
					'HEAD',
					'OPTIONS'
				);
			""")
			
			# Create rate limit rules table
			await conn.execute("""
				CREATE TABLE crm_rate_limit_rules (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					rule_name VARCHAR(255) NOT NULL,
					description TEXT,
					resource_pattern VARCHAR(500) NOT NULL,
					rate_limit_type rate_limit_type NOT NULL,
					limit_value INTEGER NOT NULL,
					window_size_seconds INTEGER DEFAULT 60,
					burst_limit INTEGER,
					scope rate_limit_scope DEFAULT 'tenant',
					enforcement_action enforcement_action DEFAULT 'reject',
					override_headers JSONB DEFAULT '{}',
					exception_conditions JSONB DEFAULT '[]',
					whitelist_ips JSONB DEFAULT '[]',
					blacklist_ips JSONB DEFAULT '[]',
					is_active BOOLEAN DEFAULT true,
					priority INTEGER DEFAULT 100,
					effective_from DATE DEFAULT CURRENT_DATE,
					effective_until DATE,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create API endpoints table
			await conn.execute("""
				CREATE TABLE crm_api_endpoints (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					endpoint_path VARCHAR(500) NOT NULL,
					http_methods JSONB NOT NULL DEFAULT '[]',
					description TEXT,
					version VARCHAR(20) DEFAULT 'v1',
					is_public BOOLEAN DEFAULT false,
					authentication_required BOOLEAN DEFAULT true,
					authorization_rules JSONB DEFAULT '[]',
					rate_limit_rules JSONB DEFAULT '[]',
					transformation_rules JSONB DEFAULT '{}',
					caching_config JSONB DEFAULT '{}',
					monitoring_config JSONB DEFAULT '{}',
					deprecated BOOLEAN DEFAULT false,
					deprecation_date DATE,
					replacement_endpoint VARCHAR(500),
					swagger_spec JSONB DEFAULT '{}',
					tags JSONB DEFAULT '[]',
					security_headers JSONB DEFAULT '{}',
					cors_config JSONB DEFAULT '{}',
					timeout_seconds INTEGER DEFAULT 30,
					max_request_size_mb INTEGER DEFAULT 10,
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create API requests table
			await conn.execute("""
				CREATE TABLE crm_api_requests (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					endpoint_id VARCHAR(36) REFERENCES crm_api_endpoints(id),
					request_path VARCHAR(1000) NOT NULL,
					http_method VARCHAR(10) NOT NULL,
					client_ip INET,
					user_agent TEXT,
					user_id VARCHAR(36),
					api_key_id VARCHAR(36),
					session_id VARCHAR(100),
					request_headers JSONB DEFAULT '{}',
					query_parameters JSONB DEFAULT '{}',
					request_body_size INTEGER DEFAULT 0,
					response_status INTEGER,
					response_body_size INTEGER DEFAULT 0,
					response_time_ms DECIMAL(10,3) DEFAULT 0.000,
					rate_limit_applied BOOLEAN DEFAULT false,
					rate_limit_rule_id VARCHAR(36),
					cache_hit BOOLEAN DEFAULT false,
					cache_key VARCHAR(100),
					transformation_applied BOOLEAN DEFAULT false,
					errors JSONB DEFAULT '[]',
					warnings JSONB DEFAULT '[]',
					trace_id VARCHAR(100),
					span_id VARCHAR(100),
					metadata JSONB DEFAULT '{}',
					timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create API gateway metrics table
			await conn.execute("""
				CREATE TABLE crm_api_gateway_metrics (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					metric_date DATE NOT NULL,
					metric_hour INTEGER DEFAULT 0,
					total_requests INTEGER DEFAULT 0,
					successful_requests INTEGER DEFAULT 0,
					failed_requests INTEGER DEFAULT 0,
					rate_limited_requests INTEGER DEFAULT 0,
					cached_requests INTEGER DEFAULT 0,
					transformed_requests INTEGER DEFAULT 0,
					avg_response_time_ms DECIMAL(10,3) DEFAULT 0.000,
					p50_response_time_ms DECIMAL(10,3) DEFAULT 0.000,
					p95_response_time_ms DECIMAL(10,3) DEFAULT 0.000,
					p99_response_time_ms DECIMAL(10,3) DEFAULT 0.000,
					total_bandwidth_bytes BIGINT DEFAULT 0,
					inbound_bandwidth_bytes BIGINT DEFAULT 0,
					outbound_bandwidth_bytes BIGINT DEFAULT 0,
					unique_clients INTEGER DEFAULT 0,
					unique_endpoints INTEGER DEFAULT 0,
					top_endpoints JSONB DEFAULT '[]',
					top_clients JSONB DEFAULT '[]',
					error_breakdown JSONB DEFAULT '{}',
					status_code_breakdown JSONB DEFAULT '{}',
					method_breakdown JSONB DEFAULT '{}',
					geography_breakdown JSONB DEFAULT '{}',
					device_breakdown JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(tenant_id, metric_date, metric_hour)
				);
			""")
			
			# Create API keys table
			await conn.execute("""
				CREATE TABLE crm_api_keys (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					key_name VARCHAR(255) NOT NULL,
					description TEXT,
					api_key_hash VARCHAR(128) NOT NULL UNIQUE,
					api_key_prefix VARCHAR(20) NOT NULL,
					user_id VARCHAR(36),
					permissions JSONB DEFAULT '[]',
					allowed_ips JSONB DEFAULT '[]',
					allowed_endpoints JSONB DEFAULT '[]',
					rate_limit_overrides JSONB DEFAULT '{}',
					quota_limits JSONB DEFAULT '{}',
					usage_stats JSONB DEFAULT '{}',
					is_active BOOLEAN DEFAULT true,
					expires_at TIMESTAMP WITH TIME ZONE,
					last_used_at TIMESTAMP WITH TIME ZONE,
					last_used_ip INET,
					revocation_reason TEXT,
					revoked_at TIMESTAMP WITH TIME ZONE,
					revoked_by VARCHAR(36),
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL
				);
			""")
			
			# Create API webhooks table
			await conn.execute("""
				CREATE TABLE crm_api_webhooks (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					webhook_name VARCHAR(255) NOT NULL,
					description TEXT,
					endpoint_url VARCHAR(1000) NOT NULL,
					http_method VARCHAR(10) DEFAULT 'POST',
					headers JSONB DEFAULT '{}',
					authentication JSONB DEFAULT '{}',
					event_types JSONB NOT NULL DEFAULT '[]',
					filters JSONB DEFAULT '{}',
					transformation_template TEXT,
					retry_config JSONB DEFAULT '{"max_retries": 3, "retry_delay_seconds": 60}',
					timeout_seconds INTEGER DEFAULT 30,
					is_active BOOLEAN DEFAULT true,
					failure_count INTEGER DEFAULT 0,
					last_success_at TIMESTAMP WITH TIME ZONE,
					last_failure_at TIMESTAMP WITH TIME ZONE,
					last_failure_reason TEXT,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL
				);
			""")
			
			# Create webhook delivery log table
			await conn.execute("""
				CREATE TABLE crm_webhook_deliveries (
					id VARCHAR(36) PRIMARY KEY,
					webhook_id VARCHAR(36) NOT NULL REFERENCES crm_api_webhooks(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					event_type VARCHAR(100) NOT NULL,
					event_id VARCHAR(36),
					payload JSONB NOT NULL,
					delivery_url VARCHAR(1000) NOT NULL,
					http_method VARCHAR(10) NOT NULL,
					headers JSONB DEFAULT '{}',
					response_status INTEGER,
					response_headers JSONB DEFAULT '{}',
					response_body TEXT,
					delivery_time_ms DECIMAL(10,3),
					attempt_number INTEGER DEFAULT 1,
					success BOOLEAN DEFAULT false,
					error_message TEXT,
					scheduled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					delivered_at TIMESTAMP WITH TIME ZONE,
					next_retry_at TIMESTAMP WITH TIME ZONE,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create API circuit breakers table
			await conn.execute("""
				CREATE TABLE crm_api_circuit_breakers (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					breaker_name VARCHAR(255) NOT NULL,
					resource_pattern VARCHAR(500) NOT NULL,
					failure_threshold INTEGER DEFAULT 50,
					success_threshold INTEGER DEFAULT 10,
					timeout_seconds INTEGER DEFAULT 60,
					window_size_seconds INTEGER DEFAULT 300,
					half_open_max_calls INTEGER DEFAULT 3,
					current_state VARCHAR(20) DEFAULT 'closed',
					failure_count INTEGER DEFAULT 0,
					success_count INTEGER DEFAULT 0,
					last_failure_time TIMESTAMP WITH TIME ZONE,
					next_attempt_time TIMESTAMP WITH TIME ZONE,
					state_changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					UNIQUE(tenant_id, resource_pattern)
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_rate_limit_rules_tenant ON crm_rate_limit_rules(tenant_id);")
			await conn.execute("CREATE INDEX idx_rate_limit_rules_pattern ON crm_rate_limit_rules(resource_pattern);")
			await conn.execute("CREATE INDEX idx_rate_limit_rules_active ON crm_rate_limit_rules(is_active);")
			await conn.execute("CREATE INDEX idx_rate_limit_rules_priority ON crm_rate_limit_rules(priority DESC);")
			await conn.execute("CREATE INDEX idx_rate_limit_rules_scope ON crm_rate_limit_rules(scope);")
			
			await conn.execute("CREATE INDEX idx_api_endpoints_tenant ON crm_api_endpoints(tenant_id);")
			await conn.execute("CREATE INDEX idx_api_endpoints_path ON crm_api_endpoints(endpoint_path);")
			await conn.execute("CREATE INDEX idx_api_endpoints_version ON crm_api_endpoints(version);")
			await conn.execute("CREATE INDEX idx_api_endpoints_active ON crm_api_endpoints(is_active);")
			await conn.execute("CREATE INDEX idx_api_endpoints_deprecated ON crm_api_endpoints(deprecated);")
			
			await conn.execute("CREATE INDEX idx_api_requests_tenant ON crm_api_requests(tenant_id);")
			await conn.execute("CREATE INDEX idx_api_requests_endpoint ON crm_api_requests(endpoint_id);")
			await conn.execute("CREATE INDEX idx_api_requests_timestamp ON crm_api_requests(timestamp);")
			await conn.execute("CREATE INDEX idx_api_requests_client_ip ON crm_api_requests(client_ip);")
			await conn.execute("CREATE INDEX idx_api_requests_user ON crm_api_requests(user_id);")
			await conn.execute("CREATE INDEX idx_api_requests_status ON crm_api_requests(response_status);")
			await conn.execute("CREATE INDEX idx_api_requests_path ON crm_api_requests(request_path);")
			await conn.execute("CREATE INDEX idx_api_requests_method ON crm_api_requests(http_method);")
			await conn.execute("CREATE INDEX idx_api_requests_rate_limited ON crm_api_requests(rate_limit_applied);")
			
			await conn.execute("CREATE INDEX idx_api_gateway_metrics_tenant ON crm_api_gateway_metrics(tenant_id);")
			await conn.execute("CREATE INDEX idx_api_gateway_metrics_date ON crm_api_gateway_metrics(metric_date);")
			await conn.execute("CREATE INDEX idx_api_gateway_metrics_hour ON crm_api_gateway_metrics(metric_hour);")
			
			await conn.execute("CREATE INDEX idx_api_keys_tenant ON crm_api_keys(tenant_id);")
			await conn.execute("CREATE INDEX idx_api_keys_hash ON crm_api_keys(api_key_hash);")
			await conn.execute("CREATE INDEX idx_api_keys_prefix ON crm_api_keys(api_key_prefix);")
			await conn.execute("CREATE INDEX idx_api_keys_user ON crm_api_keys(user_id);")
			await conn.execute("CREATE INDEX idx_api_keys_active ON crm_api_keys(is_active);")
			await conn.execute("CREATE INDEX idx_api_keys_expires ON crm_api_keys(expires_at);")
			
			await conn.execute("CREATE INDEX idx_api_webhooks_tenant ON crm_api_webhooks(tenant_id);")
			await conn.execute("CREATE INDEX idx_api_webhooks_active ON crm_api_webhooks(is_active);")
			await conn.execute("CREATE INDEX idx_api_webhooks_events ON crm_api_webhooks USING GIN(event_types);")
			
			await conn.execute("CREATE INDEX idx_webhook_deliveries_webhook ON crm_webhook_deliveries(webhook_id);")
			await conn.execute("CREATE INDEX idx_webhook_deliveries_tenant ON crm_webhook_deliveries(tenant_id);")
			await conn.execute("CREATE INDEX idx_webhook_deliveries_event ON crm_webhook_deliveries(event_type);")
			await conn.execute("CREATE INDEX idx_webhook_deliveries_scheduled ON crm_webhook_deliveries(scheduled_at);")
			await conn.execute("CREATE INDEX idx_webhook_deliveries_success ON crm_webhook_deliveries(success);")
			await conn.execute("CREATE INDEX idx_webhook_deliveries_retry ON crm_webhook_deliveries(next_retry_at);")
			
			await conn.execute("CREATE INDEX idx_circuit_breakers_tenant ON crm_api_circuit_breakers(tenant_id);")
			await conn.execute("CREATE INDEX idx_circuit_breakers_pattern ON crm_api_circuit_breakers(resource_pattern);")
			await conn.execute("CREATE INDEX idx_circuit_breakers_state ON crm_api_circuit_breakers(current_state);")
			await conn.execute("CREATE INDEX idx_circuit_breakers_active ON crm_api_circuit_breakers(is_active);")
			
			# Insert default rate limit rules
			await conn.execute("""
				INSERT INTO crm_rate_limit_rules (
					id, tenant_id, rule_name, description, resource_pattern,
					rate_limit_type, limit_value, window_size_seconds, scope,
					enforcement_action, priority, created_by
				) VALUES 
				(
					'rate_default_global',
					'system',
					'Global Rate Limit',
					'Default global rate limiting for all API endpoints',
					'/api/*',
					'requests_per_minute',
					1000,
					60,
					'global',
					'reject',
					50,
					'system'
				),
				(
					'rate_tenant_standard',
					'system',
					'Tenant Standard Rate Limit',
					'Standard rate limiting per tenant',
					'/api/*',
					'requests_per_minute',
					500,
					60,
					'tenant',
					'reject',
					75,
					'system'
				),
				(
					'rate_user_standard',
					'system',
					'User Standard Rate Limit',
					'Standard rate limiting per user',
					'/api/*',
					'requests_per_minute',
					100,
					60,
					'user',
					'reject',
					100,
					'system'
				),
				(
					'rate_auth_endpoints',
					'system',
					'Authentication Endpoints Rate Limit',
					'Strict rate limiting for authentication endpoints',
					'/api/auth/*',
					'requests_per_minute',
					10,
					60,
					'ip',
					'reject',
					200,
					'system'
				)
			""")
			
			# Insert sample API endpoints
			await conn.execute("""
				INSERT INTO crm_api_endpoints (
					id, tenant_id, endpoint_path, http_methods, description, version,
					is_public, authentication_required, caching_config, created_by
				) VALUES 
				(
					'endpoint_contacts_list',
					'system',
					'/api/v1/contacts',
					'["GET"]',
					'List contacts with pagination and filtering',
					'v1',
					false,
					true,
					'{"enabled": true, "ttl_seconds": 300, "vary_headers": ["X-Tenant-ID"]}',
					'system'
				),
				(
					'endpoint_opportunities_list',
					'system',
					'/api/v1/opportunities',
					'["GET"]',
					'List opportunities with advanced filtering',
					'v1',
					false,
					true,
					'{"enabled": true, "ttl_seconds": 180, "vary_headers": ["X-Tenant-ID", "X-User-ID"]}',
					'system'
				),
				(
					'endpoint_analytics_dashboard',
					'system',
					'/api/v1/analytics/dashboard',
					'["GET"]',
					'Get analytics dashboard data',
					'v1',
					false,
					true,
					'{"enabled": true, "ttl_seconds": 600, "cache_key_include": ["dashboard_type"]}',
					'system'
				)
			""")
			
			logger.info("✅ API gateway tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create API gateway tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping API gateway tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_api_circuit_breakers CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_webhook_deliveries CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_api_webhooks CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_api_keys CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_api_gateway_metrics CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_api_requests CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_api_endpoints CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_rate_limit_rules CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS http_method CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS rate_limit_scope CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS enforcement_action CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS rate_limit_type CASCADE;")
			
			logger.info("✅ API gateway tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop API gateway tables: {str(e)}")
			raise