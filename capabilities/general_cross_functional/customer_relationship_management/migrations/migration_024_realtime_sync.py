"""
APG Customer Relationship Management - Real-Time Synchronization Migration

Database migration to create real-time sync tables and supporting 
structures for event streaming, change detection, conflict resolution, and monitoring.

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


class RealtimeSyncMigration(BaseMigration):
	"""Migration for real-time synchronization functionality"""
	
	def _get_migration_id(self) -> str:
		return "024_realtime_sync"
	
	def _get_version(self) -> str:
		return "024"
	
	def _get_description(self) -> str:
		return "Real-time data synchronization with event streaming and conflict resolution"
	
	def _get_dependencies(self) -> list[str]:
		return ["023_third_party_integration"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating real-time synchronization tables...")
			
			# Create sync event type enum
			await conn.execute("""
				CREATE TYPE sync_event_type AS ENUM (
					'entity.created',
					'entity.updated',
					'entity.deleted',
					'entity.restored',
					'relationship.created',
					'relationship.updated',
					'relationship.deleted',
					'batch.operation',
					'schema.changed',
					'system.event'
				);
			""")
			
			# Create conflict resolution strategy enum
			await conn.execute("""
				CREATE TYPE conflict_resolution_strategy AS ENUM (
					'timestamp_wins',
					'source_wins',
					'target_wins',
					'manual_resolution',
					'merge_strategy',
					'custom_logic'
				);
			""")
			
			# Create sync status enum
			await conn.execute("""
				CREATE TYPE sync_status AS ENUM (
					'active',
					'paused',
					'error',
					'degraded',
					'maintenance'
				);
			""")
			
			# Create change detection mode enum
			await conn.execute("""
				CREATE TYPE change_detection_mode AS ENUM (
					'timestamp_based',
					'hash_based',
					'field_level',
					'event_driven',
					'hybrid'
				);
			""")
			
			# Create real-time sync configurations table
			await conn.execute("""
				CREATE TABLE crm_realtime_sync_configs (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					config_name VARCHAR(255) NOT NULL,
					description TEXT,
					entity_types JSONB NOT NULL DEFAULT '[]',
					field_filters JSONB DEFAULT '{}',
					exclude_fields JSONB DEFAULT '{}',
					change_detection_mode change_detection_mode DEFAULT 'timestamp_based',
					detection_config JSONB DEFAULT '{}',
					sync_direction VARCHAR(50) DEFAULT 'bidirectional',
					conflict_resolution conflict_resolution_strategy DEFAULT 'timestamp_wins',
					batch_size INTEGER DEFAULT 100,
					max_retry_attempts INTEGER DEFAULT 3,
					retry_delay_seconds INTEGER DEFAULT 5,
					throttle_rate_per_second INTEGER DEFAULT 1000,
					max_concurrent_syncs INTEGER DEFAULT 10,
					sync_timeout_seconds INTEGER DEFAULT 30,
					target_systems JSONB DEFAULT '[]',
					system_priorities JSONB DEFAULT '{}',
					enable_deduplication BOOLEAN DEFAULT true,
					enable_conflict_detection BOOLEAN DEFAULT true,
					enable_audit_trail BOOLEAN DEFAULT true,
					maintain_sync_history BOOLEAN DEFAULT true,
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create sync events table
			await conn.execute("""
				CREATE TABLE crm_sync_events (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					event_type sync_event_type NOT NULL,
					entity_type VARCHAR(100) NOT NULL,
					entity_id VARCHAR(36) NOT NULL,
					current_data JSONB NOT NULL DEFAULT '{}',
					previous_data JSONB DEFAULT '{}',
					changed_fields JSONB DEFAULT '[]',
					source_system VARCHAR(100) DEFAULT 'crm',
					target_systems JSONB DEFAULT '[]',
					sync_priority INTEGER DEFAULT 100,
					batch_id VARCHAR(36),
					correlation_id VARCHAR(36),
					data_hash VARCHAR(64),
					version INTEGER DEFAULT 1,
					conflict_detected BOOLEAN DEFAULT false,
					user_id VARCHAR(36),
					session_id VARCHAR(100),
					metadata JSONB DEFAULT '{}',
					timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					processed BOOLEAN DEFAULT false,
					processed_at TIMESTAMP WITH TIME ZONE,
					processing_errors JSONB DEFAULT '[]',
					completed_at TIMESTAMP WITH TIME ZONE
				);
			""")
			
			# Create sync conflicts table (enhanced from third-party integration)
			await conn.execute("""
				CREATE TABLE crm_sync_conflicts (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					entity_type VARCHAR(100) NOT NULL,
					entity_id VARCHAR(36) NOT NULL,
					source_system VARCHAR(100) NOT NULL,
					target_system VARCHAR(100) NOT NULL,
					conflict_field VARCHAR(100) NOT NULL,
					source_value JSONB,
					target_value JSONB,
					source_timestamp TIMESTAMP WITH TIME ZONE,
					target_timestamp TIMESTAMP WITH TIME ZONE,
					resolution_strategy conflict_resolution_strategy DEFAULT 'timestamp_wins',
					resolved_value JSONB,
					resolved BOOLEAN DEFAULT false,
					resolved_by VARCHAR(36),
					resolved_at TIMESTAMP WITH TIME ZONE,
					auto_resolved BOOLEAN DEFAULT false,
					resolution_notes TEXT,
					conflict_severity VARCHAR(20) DEFAULT 'medium',
					business_impact JSONB DEFAULT '{}',
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create sync operations table
			await conn.execute("""
				CREATE TABLE crm_sync_operations (
					id VARCHAR(36) PRIMARY KEY,
					sync_event_id VARCHAR(36) NOT NULL REFERENCES crm_sync_events(id) ON DELETE CASCADE,
					config_id VARCHAR(36) NOT NULL REFERENCES crm_realtime_sync_configs(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					operation_type VARCHAR(50) NOT NULL,
					target_system VARCHAR(100) NOT NULL,
					status VARCHAR(50) DEFAULT 'pending',
					attempt_number INTEGER DEFAULT 1,
					max_attempts INTEGER DEFAULT 3,
					started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					completed_at TIMESTAMP WITH TIME ZONE,
					duration_ms DECIMAL(10,3),
					request_payload JSONB DEFAULT '{}',
					response_payload JSONB DEFAULT '{}',
					response_status INTEGER,
					error_message TEXT,
					error_code VARCHAR(50),
					retry_after TIMESTAMP WITH TIME ZONE,
					lock_acquired BOOLEAN DEFAULT false,
					lock_key VARCHAR(200),
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create change detection snapshots table
			await conn.execute("""
				CREATE TABLE crm_change_snapshots (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					entity_type VARCHAR(100) NOT NULL,
					entity_id VARCHAR(36) NOT NULL,
					snapshot_hash VARCHAR(64) NOT NULL,
					snapshot_data JSONB NOT NULL,
					field_hashes JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE,
					UNIQUE(tenant_id, entity_type, entity_id)
				);
			""")
			
			# Create sync metrics aggregation table
			await conn.execute("""
				CREATE TABLE crm_sync_metrics (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					config_id VARCHAR(36) REFERENCES crm_realtime_sync_configs(id) ON DELETE CASCADE,
					metric_date DATE NOT NULL,
					metric_hour INTEGER DEFAULT 0,
					total_events INTEGER DEFAULT 0,
					successful_events INTEGER DEFAULT 0,
					failed_events INTEGER DEFAULT 0,
					conflicts_detected INTEGER DEFAULT 0,
					conflicts_resolved INTEGER DEFAULT 0,
					auto_resolved_conflicts INTEGER DEFAULT 0,
					manual_resolved_conflicts INTEGER DEFAULT 0,
					avg_processing_time_ms DECIMAL(10,3) DEFAULT 0.000,
					p95_processing_time_ms DECIMAL(10,3) DEFAULT 0.000,
					throughput_events_per_second DECIMAL(10,3) DEFAULT 0.000,
					data_volume_bytes BIGINT DEFAULT 0,
					unique_entities INTEGER DEFAULT 0,
					target_system_breakdown JSONB DEFAULT '{}',
					event_type_breakdown JSONB DEFAULT '{}',
					entity_type_breakdown JSONB DEFAULT '{}',
					error_breakdown JSONB DEFAULT '{}',
					performance_metrics JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(tenant_id, config_id, metric_date, metric_hour)
				);
			""")
			
			# Create sync locks table for distributed coordination
			await conn.execute("""
				CREATE TABLE crm_sync_locks (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					resource_type VARCHAR(100) NOT NULL,
					resource_id VARCHAR(100) NOT NULL,
					lock_key VARCHAR(200) NOT NULL UNIQUE,
					node_id VARCHAR(100) NOT NULL,
					acquired_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
					heartbeat_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					metadata JSONB DEFAULT '{}',
					INDEX(tenant_id, resource_type, resource_id)
				);
			""")
			
			# Create sync node registry table
			await conn.execute("""
				CREATE TABLE crm_sync_nodes (
					id VARCHAR(36) PRIMARY KEY,
					node_id VARCHAR(100) NOT NULL UNIQUE,
					node_type VARCHAR(50) DEFAULT 'worker',
					status VARCHAR(50) DEFAULT 'active',
					capabilities JSONB DEFAULT '[]',
					current_load INTEGER DEFAULT 0,
					max_capacity INTEGER DEFAULT 100,
					version VARCHAR(20),
					started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create sync batches table
			await conn.execute("""
				CREATE TABLE crm_sync_batches (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					config_id VARCHAR(36) NOT NULL REFERENCES crm_realtime_sync_configs(id) ON DELETE CASCADE,
					batch_name VARCHAR(255),
					batch_type VARCHAR(50) DEFAULT 'automatic',
					total_events INTEGER DEFAULT 0,
					processed_events INTEGER DEFAULT 0,
					successful_events INTEGER DEFAULT 0,
					failed_events INTEGER DEFAULT 0,
					batch_status VARCHAR(50) DEFAULT 'pending',
					priority INTEGER DEFAULT 100,
					started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					completed_at TIMESTAMP WITH TIME ZONE,
					processing_node VARCHAR(100),
					batch_config JSONB DEFAULT '{}',
					results_summary JSONB DEFAULT '{}',
					error_summary JSONB DEFAULT '{}',
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create sync event subscriptions table
			await conn.execute("""
				CREATE TABLE crm_sync_subscriptions (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					subscriber_name VARCHAR(255) NOT NULL,
					subscriber_type VARCHAR(50) NOT NULL,
					event_filters JSONB DEFAULT '{}',
					entity_filters JSONB DEFAULT '{}',
					field_filters JSONB DEFAULT '{}',
					callback_url VARCHAR(1000),
					callback_config JSONB DEFAULT '{}',
					delivery_mode VARCHAR(50) DEFAULT 'push',
					buffer_config JSONB DEFAULT '{}',
					retry_config JSONB DEFAULT '{}',
					rate_limit_config JSONB DEFAULT '{}',
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					last_delivery_at TIMESTAMP WITH TIME ZONE,
					delivery_stats JSONB DEFAULT '{}'
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_realtime_sync_configs_tenant ON crm_realtime_sync_configs(tenant_id);")
			await conn.execute("CREATE INDEX idx_realtime_sync_configs_active ON crm_realtime_sync_configs(is_active);")
			await conn.execute("CREATE INDEX idx_realtime_sync_configs_entities ON crm_realtime_sync_configs USING GIN(entity_types);")
			await conn.execute("CREATE INDEX idx_realtime_sync_configs_targets ON crm_realtime_sync_configs USING GIN(target_systems);")
			
			await conn.execute("CREATE INDEX idx_sync_events_tenant ON crm_sync_events(tenant_id);")
			await conn.execute("CREATE INDEX idx_sync_events_type ON crm_sync_events(event_type);")
			await conn.execute("CREATE INDEX idx_sync_events_entity ON crm_sync_events(entity_type, entity_id);")
			await conn.execute("CREATE INDEX idx_sync_events_timestamp ON crm_sync_events(timestamp);")
			await conn.execute("CREATE INDEX idx_sync_events_processed ON crm_sync_events(processed);")
			await conn.execute("CREATE INDEX idx_sync_events_correlation ON crm_sync_events(correlation_id);")
			await conn.execute("CREATE INDEX idx_sync_events_batch ON crm_sync_events(batch_id);")
			await conn.execute("CREATE INDEX idx_sync_events_user ON crm_sync_events(user_id);")
			await conn.execute("CREATE INDEX idx_sync_events_conflicts ON crm_sync_events(conflict_detected);")
			await conn.execute("CREATE INDEX idx_sync_events_targets ON crm_sync_events USING GIN(target_systems);")
			
			await conn.execute("CREATE INDEX idx_sync_conflicts_tenant ON crm_sync_conflicts(tenant_id);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_entity ON crm_sync_conflicts(entity_type, entity_id);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_systems ON crm_sync_conflicts(source_system, target_system);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_resolved ON crm_sync_conflicts(resolved);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_field ON crm_sync_conflicts(conflict_field);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_severity ON crm_sync_conflicts(conflict_severity);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_timestamp ON crm_sync_conflicts(source_timestamp, target_timestamp);")
			await conn.execute("CREATE INDEX idx_sync_conflicts_created ON crm_sync_conflicts(created_at);")
			
			await conn.execute("CREATE INDEX idx_sync_operations_event ON crm_sync_operations(sync_event_id);")
			await conn.execute("CREATE INDEX idx_sync_operations_config ON crm_sync_operations(config_id);")
			await conn.execute("CREATE INDEX idx_sync_operations_tenant ON crm_sync_operations(tenant_id);")
			await conn.execute("CREATE INDEX idx_sync_operations_status ON crm_sync_operations(status);")
			await conn.execute("CREATE INDEX idx_sync_operations_target ON crm_sync_operations(target_system);")
			await conn.execute("CREATE INDEX idx_sync_operations_retry ON crm_sync_operations(retry_after);")
			await conn.execute("CREATE INDEX idx_sync_operations_lock ON crm_sync_operations(lock_acquired, lock_key);")
			
			await conn.execute("CREATE INDEX idx_change_snapshots_tenant ON crm_change_snapshots(tenant_id);")
			await conn.execute("CREATE INDEX idx_change_snapshots_entity ON crm_change_snapshots(entity_type, entity_id);")
			await conn.execute("CREATE INDEX idx_change_snapshots_hash ON crm_change_snapshots(snapshot_hash);")
			await conn.execute("CREATE INDEX idx_change_snapshots_created ON crm_change_snapshots(created_at);")
			await conn.execute("CREATE INDEX idx_change_snapshots_expires ON crm_change_snapshots(expires_at);")
			
			await conn.execute("CREATE INDEX idx_sync_metrics_tenant ON crm_sync_metrics(tenant_id);")
			await conn.execute("CREATE INDEX idx_sync_metrics_config ON crm_sync_metrics(config_id);")
			await conn.execute("CREATE INDEX idx_sync_metrics_date ON crm_sync_metrics(metric_date);")
			await conn.execute("CREATE INDEX idx_sync_metrics_hour ON crm_sync_metrics(metric_hour);")
			
			await conn.execute("CREATE INDEX idx_sync_locks_tenant ON crm_sync_locks(tenant_id);")
			await conn.execute("CREATE INDEX idx_sync_locks_resource ON crm_sync_locks(resource_type, resource_id);")
			await conn.execute("CREATE INDEX idx_sync_locks_expires ON crm_sync_locks(expires_at);")
			await conn.execute("CREATE INDEX idx_sync_locks_node ON crm_sync_locks(node_id);")
			await conn.execute("CREATE INDEX idx_sync_locks_heartbeat ON crm_sync_locks(heartbeat_at);")
			
			await conn.execute("CREATE INDEX idx_sync_nodes_node_id ON crm_sync_nodes(node_id);")
			await conn.execute("CREATE INDEX idx_sync_nodes_status ON crm_sync_nodes(status);")
			await conn.execute("CREATE INDEX idx_sync_nodes_type ON crm_sync_nodes(node_type);")
			await conn.execute("CREATE INDEX idx_sync_nodes_heartbeat ON crm_sync_nodes(last_heartbeat);")
			await conn.execute("CREATE INDEX idx_sync_nodes_load ON crm_sync_nodes(current_load);")
			
			await conn.execute("CREATE INDEX idx_sync_batches_tenant ON crm_sync_batches(tenant_id);")
			await conn.execute("CREATE INDEX idx_sync_batches_config ON crm_sync_batches(config_id);")
			await conn.execute("CREATE INDEX idx_sync_batches_status ON crm_sync_batches(batch_status);")
			await conn.execute("CREATE INDEX idx_sync_batches_priority ON crm_sync_batches(priority DESC);")
			await conn.execute("CREATE INDEX idx_sync_batches_started ON crm_sync_batches(started_at);")
			await conn.execute("CREATE INDEX idx_sync_batches_node ON crm_sync_batches(processing_node);")
			
			await conn.execute("CREATE INDEX idx_sync_subscriptions_tenant ON crm_sync_subscriptions(tenant_id);")
			await conn.execute("CREATE INDEX idx_sync_subscriptions_type ON crm_sync_subscriptions(subscriber_type);")
			await conn.execute("CREATE INDEX idx_sync_subscriptions_active ON crm_sync_subscriptions(is_active);")
			await conn.execute("CREATE INDEX idx_sync_subscriptions_mode ON crm_sync_subscriptions(delivery_mode);")
			await conn.execute("CREATE INDEX idx_sync_subscriptions_delivery ON crm_sync_subscriptions(last_delivery_at);")
			
			# Insert sample real-time sync configurations
			await conn.execute("""
				INSERT INTO crm_realtime_sync_configs (
					id, tenant_id, config_name, description, entity_types,
					change_detection_mode, conflict_resolution, sync_direction,
					target_systems, enable_conflict_detection, created_by
				) VALUES 
				(
					'sync_config_contacts_realtime',
					'system',
					'Real-time Contact Sync',
					'Real-time bidirectional synchronization of contact data across all systems',
					'["contact", "person"]',
					'event_driven',
					'timestamp_wins',
					'bidirectional',
					'["salesforce", "hubspot", "mailchimp"]',
					true,
					'system'
				),
				(
					'sync_config_opportunities_realtime',
					'system',
					'Real-time Opportunity Sync',
					'Real-time synchronization of sales opportunities with conflict detection',
					'["opportunity", "deal"]',
					'hybrid',
					'manual_resolution',
					'bidirectional',  
					'["salesforce", "pipedrive"]',
					true,
					'system'
				),
				(
					'sync_config_accounts_realtime',
					'system',
					'Real-time Account Sync',
					'Real-time account data synchronization with field-level change detection',
					'["account", "company"]',
					'field_level',
					'merge_strategy',
					'outbound',
					'["salesforce", "hubspot", "quickbooks"]',
					true,
					'system'
				)
			""")
			
			# Insert sample sync subscriptions
			await conn.execute("""
				INSERT INTO crm_sync_subscriptions (
					id, tenant_id, subscriber_name, subscriber_type,
					event_filters, delivery_mode, created_by
				) VALUES 
				(
					'sub_analytics_events',
					'system',
					'Analytics Dashboard',
					'internal_dashboard',
					'{"event_types": ["entity.created", "entity.updated"], "entity_types": ["contact", "opportunity"]}',
					'push',
					'system'
				),
				(
					'sub_audit_trail',
					'system',
					'Audit Trail Logger',
					'audit_system',
					'{"event_types": ["entity.created", "entity.updated", "entity.deleted"], "enable_full_payload": true}',
					'push',
					'system'
				),
				(
					'sub_webhook_relay',
					'system',
					'External Webhook Relay',
					'webhook_relay',
					'{"event_types": ["entity.updated"], "priority_entities": ["opportunity", "account"]}',
					'push',
					'system'
				)
			""")
			
			# Insert sample sync node
			await conn.execute("""
				INSERT INTO crm_sync_nodes (
					id, node_id, node_type, capabilities, max_capacity
				) VALUES 
				(
					'node_primary_sync',
					'sync-node-1',
					'primary_worker',
					'["event_processing", "conflict_resolution", "batch_processing", "real_time_sync"]',
					500
				)
			""")
			
			logger.info("✅ Real-time synchronization tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create real-time sync tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping real-time synchronization tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_sync_subscriptions CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_sync_batches CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_sync_nodes CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_sync_locks CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_sync_metrics CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_change_snapshots CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_sync_operations CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_sync_conflicts CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_sync_events CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_realtime_sync_configs CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS change_detection_mode CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS sync_status CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS conflict_resolution_strategy CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS sync_event_type CASCADE;")
			
			logger.info("✅ Real-time synchronization tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop real-time sync tables: {str(e)}")
			raise