"""Initial APG schema — apg_records JSONB store + 63 ORM tables.

Revision ID: 001_initial_schema
Create Date: 2026-06-17 11:08:54

Tables created:
- apg_records: shared JSONB store for all capabilities
- cr_*: composition registry (9 tables)
- es_*: event streaming (11 tables)
- sm_*: gateway/service-mesh (20 tables)
- so_oe_*: CRM order entry (9 tables)
- ds_*: GRC document control (5 tables)
- am_*: API management (9 tables)
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Shared JSONB store ────────────────────────────────────────────────
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS apg_records (
            id          TEXT        NOT NULL,
            collection  TEXT        NOT NULL,
            tenant_id   TEXT        NOT NULL DEFAULT 'default',
            data        JSONB       NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (collection, id)
        )
    """))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_apg_records_tenant ON apg_records (collection, tenant_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_apg_records_data_gin ON apg_records USING gin (data)"))
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION apg_set_updated_at()
        RETURNS TRIGGER LANGUAGE plpgsql AS $$
        BEGIN NEW.updated_at := now(); RETURN NEW; END;
        $$
    """))
    op.execute(sa.text("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_apg_records_updated_at') THEN
                CREATE TRIGGER trg_apg_records_updated_at
                    BEFORE UPDATE ON apg_records
                    FOR EACH ROW EXECUTE FUNCTION apg_set_updated_at();
            END IF;
        END $$
    """))

    # ── ORM model tables ──────────────────────────────────────────────────
    op.execute(sa.text("""
        CREATE TABLE cr_capabilities (
        	capability_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	capability_code VARCHAR(100) NOT NULL, 
        	capability_name VARCHAR(255) NOT NULL, 
        	description TEXT, 
        	version VARCHAR(50) NOT NULL, 
        	category VARCHAR(100) NOT NULL, 
        	subcategory VARCHAR(100), 
        	priority INTEGER, 
        	status VARCHAR(50) NOT NULL, 
        	composition_keywords JSON, 
        	provides_services JSON, 
        	data_models JSON, 
        	api_endpoints JSON, 
        	multi_tenant BOOLEAN NOT NULL, 
        	audit_enabled BOOLEAN NOT NULL, 
        	security_integration BOOLEAN NOT NULL, 
        	performance_optimized BOOLEAN, 
        	ai_enhanced BOOLEAN, 
        	target_users JSON, 
        	business_value TEXT, 
        	use_cases JSON, 
        	industry_focus JSON, 
        	file_path VARCHAR(500), 
        	module_path VARCHAR(500), 
        	documentation_path VARCHAR(500), 
        	repository_url VARCHAR(500), 
        	complexity_score FLOAT, 
        	quality_score FLOAT, 
        	popularity_score FLOAT, 
        	usage_count INTEGER, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE, 
        	created_by VARCHAR(36) NOT NULL, 
        	updated_by VARCHAR(36), 
        	metadata JSON, 
        	PRIMARY KEY (capability_id), 
        	CONSTRAINT uq_tenant_capability UNIQUE (tenant_id, capability_code)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_capability_status ON cr_capabilities (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_cr_capabilities_capability_code ON cr_capabilities (capability_code)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_capability_code ON cr_capabilities (capability_code)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_capability_tenant ON cr_capabilities (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_cr_capabilities_tenant_id ON cr_capabilities (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_capability_search ON cr_capabilities (capability_name, description)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_capability_category ON cr_capabilities (category)
    """))
    op.execute(sa.text("""
        CREATE TABLE cr_compositions (
        	composition_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	name VARCHAR(255) NOT NULL, 
        	description TEXT, 
        	composition_type VARCHAR(50) NOT NULL, 
        	version VARCHAR(50) NOT NULL, 
        	industry_template VARCHAR(100), 
        	deployment_strategy VARCHAR(100), 
        	validation_status VARCHAR(50) NOT NULL, 
        	validation_results JSON, 
        	validation_errors JSON, 
        	validation_warnings JSON, 
        	configuration JSON, 
        	environment_settings JSON, 
        	deployment_config JSON, 
        	estimated_complexity FLOAT, 
        	estimated_cost FLOAT, 
        	estimated_deployment_time VARCHAR(50), 
        	performance_metrics JSON, 
        	business_requirements JSON, 
        	compliance_requirements JSON, 
        	target_users JSON, 
        	is_template BOOLEAN, 
        	is_public BOOLEAN, 
        	shared_with_tenants JSON, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE, 
        	created_by VARCHAR(36) NOT NULL, 
        	updated_by VARCHAR(36), 
        	metadata JSON, 
        	PRIMARY KEY (composition_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_composition_tenant ON cr_compositions (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_composition_created ON cr_compositions (created_at)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_composition_status ON cr_compositions (validation_status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_cr_compositions_tenant_id ON cr_compositions (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_composition_type ON cr_compositions (composition_type)
    """))
    op.execute(sa.text("""
        CREATE TABLE cr_registry (
        	registry_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	name VARCHAR(255) NOT NULL, 
        	description TEXT, 
        	auto_discovery_enabled BOOLEAN, 
        	auto_validation_enabled BOOLEAN, 
        	marketplace_integration BOOLEAN, 
        	ai_recommendations BOOLEAN, 
        	discovery_paths JSON, 
        	excluded_paths JSON, 
        	scan_frequency_hours INTEGER, 
        	last_scan_date TIMESTAMP WITHOUT TIME ZONE, 
        	validation_rules JSON, 
        	quality_thresholds JSON, 
        	compliance_requirements JSON, 
        	cache_ttl_seconds INTEGER, 
        	max_composition_size INTEGER, 
        	max_dependency_depth INTEGER, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE, 
        	created_by VARCHAR(36) NOT NULL, 
        	updated_by VARCHAR(36), 
        	metadata JSON, 
        	PRIMARY KEY (registry_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_registry_tenant ON cr_registry (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_cr_registry_tenant_id ON cr_registry (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE cr_composition_capabilities (
        	comp_cap_id VARCHAR(36) NOT NULL, 
        	composition_id VARCHAR(36) NOT NULL, 
        	capability_id VARCHAR(36) NOT NULL, 
        	version_constraint VARCHAR(50), 
        	required BOOLEAN NOT NULL, 
        	load_order INTEGER, 
        	configuration JSON, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(36) NOT NULL, 
        	PRIMARY KEY (comp_cap_id), 
        	CONSTRAINT uq_composition_capability UNIQUE (composition_id, capability_id), 
        	FOREIGN KEY(composition_id) REFERENCES cr_compositions (composition_id), 
        	FOREIGN KEY(capability_id) REFERENCES cr_capabilities (capability_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_comp_cap_capability ON cr_composition_capabilities (capability_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_comp_cap_composition ON cr_composition_capabilities (composition_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE cr_dependencies (
        	dependency_id VARCHAR(36) NOT NULL, 
        	capability_id VARCHAR(36) NOT NULL, 
        	depends_on_id VARCHAR(36) NOT NULL, 
        	dependency_type VARCHAR(50) NOT NULL, 
        	version_constraint VARCHAR(50), 
        	version_min VARCHAR(50), 
        	version_max VARCHAR(50), 
        	version_exact VARCHAR(50), 
        	load_priority INTEGER, 
        	initialization_order INTEGER, 
        	optional_features JSON, 
        	conflict_resolution VARCHAR(100), 
        	alternative_capabilities JSON, 
        	fallback_strategy VARCHAR(100), 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE, 
        	created_by VARCHAR(36) NOT NULL, 
        	metadata JSON, 
        	PRIMARY KEY (dependency_id), 
        	CONSTRAINT uq_capability_dependency UNIQUE (capability_id, depends_on_id), 
        	FOREIGN KEY(capability_id) REFERENCES cr_capabilities (capability_id), 
        	FOREIGN KEY(depends_on_id) REFERENCES cr_capabilities (capability_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_dependency_capability ON cr_dependencies (capability_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_dependency_type ON cr_dependencies (dependency_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_dependency_depends_on ON cr_dependencies (depends_on_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE cr_health_metrics (
        	metric_id VARCHAR(36) NOT NULL, 
        	capability_id VARCHAR(36) NOT NULL, 
        	timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	health_score FLOAT, 
        	availability_pct FLOAT, 
        	performance_score FLOAT, 
        	error_rate_pct FLOAT, 
        	dependency_health_score FLOAT, 
        	missing_dependencies INTEGER, 
        	conflicting_dependencies INTEGER, 
        	documentation_completeness FLOAT, 
        	test_coverage_pct FLOAT, 
        	code_quality_score FLOAT, 
        	security_score FLOAT, 
        	metadata JSON, 
        	PRIMARY KEY (metric_id), 
        	FOREIGN KEY(capability_id) REFERENCES cr_capabilities (capability_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_health_capability ON cr_health_metrics (capability_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_health_timestamp ON cr_health_metrics (timestamp)
    """))
    op.execute(sa.text("""
        CREATE TABLE cr_metadata (
        	metadata_id VARCHAR(36) NOT NULL, 
        	capability_id VARCHAR(36) NOT NULL, 
        	metadata_type VARCHAR(100) NOT NULL, 
        	metadata_key VARCHAR(255) NOT NULL, 
        	metadata_value TEXT, 
        	metadata_json JSON, 
        	is_searchable BOOLEAN, 
        	is_public BOOLEAN, 
        	data_type VARCHAR(50), 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE, 
        	created_by VARCHAR(36) NOT NULL, 
        	updated_by VARCHAR(36), 
        	PRIMARY KEY (metadata_id), 
        	FOREIGN KEY(capability_id) REFERENCES cr_capabilities (capability_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_metadata_type ON cr_metadata (metadata_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_metadata_capability ON cr_metadata (capability_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE cr_usage_analytics (
        	usage_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	capability_id VARCHAR(36) NOT NULL, 
        	usage_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	usage_count INTEGER, 
        	composition_count INTEGER, 
        	deployment_count INTEGER, 
        	error_count INTEGER, 
        	avg_response_time_ms FLOAT, 
        	avg_memory_usage_mb FLOAT, 
        	avg_cpu_usage_pct FLOAT, 
        	unique_users INTEGER, 
        	total_sessions INTEGER, 
        	avg_session_duration FLOAT, 
        	metadata JSON, 
        	PRIMARY KEY (usage_id), 
        	FOREIGN KEY(capability_id) REFERENCES cr_capabilities (capability_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_usage_capability ON cr_usage_analytics (capability_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_usage_date ON cr_usage_analytics (usage_date)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_cr_usage_analytics_usage_date ON cr_usage_analytics (usage_date)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_cr_usage_analytics_tenant_id ON cr_usage_analytics (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_usage_tenant ON cr_usage_analytics (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE cr_versions (
        	version_id VARCHAR(36) NOT NULL, 
        	capability_id VARCHAR(36) NOT NULL, 
        	version_number VARCHAR(50) NOT NULL, 
        	major_version INTEGER NOT NULL, 
        	minor_version INTEGER NOT NULL, 
        	patch_version INTEGER NOT NULL, 
        	pre_release VARCHAR(50), 
        	build_metadata VARCHAR(100), 
        	release_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	release_notes TEXT, 
        	breaking_changes JSON, 
        	deprecations JSON, 
        	new_features JSON, 
        	compatible_versions JSON, 
        	incompatible_versions JSON, 
        	migration_path JSON, 
        	upgrade_instructions TEXT, 
        	api_changes JSON, 
        	backward_compatible BOOLEAN, 
        	forward_compatible BOOLEAN, 
        	quality_score FLOAT, 
        	test_coverage FLOAT, 
        	documentation_score FLOAT, 
        	security_audit_passed BOOLEAN, 
        	status VARCHAR(50), 
        	end_of_life_date TIMESTAMP WITHOUT TIME ZONE, 
        	support_level VARCHAR(50), 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(36) NOT NULL, 
        	metadata JSON, 
        	PRIMARY KEY (version_id), 
        	CONSTRAINT uq_capability_version UNIQUE (capability_id, version_number), 
        	FOREIGN KEY(capability_id) REFERENCES cr_capabilities (capability_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_version_number ON cr_versions (version_number)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_version_capability ON cr_versions (capability_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_cr_version_released ON cr_versions (release_date)
    """))
    op.execute(sa.text("""
        CREATE TABLE es_event_schemas (
        	schema_id VARCHAR(50) NOT NULL, 
        	event_type VARCHAR(200) NOT NULL, 
        	schema_version VARCHAR(20) NOT NULL, 
        	schema_name VARCHAR(200) NOT NULL, 
        	schema_description TEXT, 
        	json_schema JSONB NOT NULL, 
        	avro_schema JSONB, 
        	protobuf_schema TEXT, 
        	namespace VARCHAR(100) NOT NULL, 
        	compatibility_level VARCHAR(50) NOT NULL, 
        	schema_type VARCHAR(50) NOT NULL, 
        	parent_schema_id VARCHAR(50), 
        	evolution_strategy VARCHAR(50) NOT NULL, 
        	is_active BOOLEAN NOT NULL, 
        	is_deprecated BOOLEAN NOT NULL, 
        	deprecation_date TIMESTAMP WITH TIME ZONE, 
        	strict_validation BOOLEAN NOT NULL, 
        	allow_unknown_fields BOOLEAN NOT NULL, 
        	required_fields VARCHAR[] NOT NULL, 
        	optional_fields VARCHAR[] NOT NULL, 
        	usage_count BIGINT NOT NULL, 
        	last_used TIMESTAMP WITH TIME ZONE, 
        	validation_failures BIGINT NOT NULL, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	created_by VARCHAR(50) NOT NULL, 
        	updated_by VARCHAR(50) NOT NULL, 
        	PRIMARY KEY (schema_id), 
        	CONSTRAINT uk_es_schemas_type_version UNIQUE (event_type, schema_version), 
        	FOREIGN KEY(parent_schema_id) REFERENCES es_event_schemas (schema_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_event_schemas_tenant_id ON es_event_schemas (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_schemas_namespace_type ON es_event_schemas (namespace, event_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_schemas_active ON es_event_schemas (is_active, is_deprecated)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_event_schemas_event_type ON es_event_schemas (event_type)
    """))
    op.execute(sa.text("""
        CREATE TABLE es_schemas (
        	schema_id VARCHAR(100) NOT NULL, 
        	schema_name VARCHAR(200) NOT NULL, 
        	schema_version VARCHAR(20) NOT NULL, 
        	schema_definition JSONB NOT NULL, 
        	schema_format VARCHAR(20) NOT NULL, 
        	event_type VARCHAR(100) NOT NULL, 
        	compatibility_level VARCHAR(20) NOT NULL, 
        	is_active BOOLEAN NOT NULL, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	created_by VARCHAR(100) NOT NULL, 
        	PRIMARY KEY (schema_id), 
        	CONSTRAINT uq_schema_version_tenant UNIQUE (schema_name, schema_version, tenant_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_schemas_active ON es_schemas (is_active)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_schemas_tenant_id ON es_schemas (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_schemas_event_type ON es_schemas (event_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_schemas_event_type ON es_schemas (event_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_schemas_tenant ON es_schemas (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE es_streams (
        	stream_id VARCHAR(100) NOT NULL, 
        	stream_name VARCHAR(200) NOT NULL, 
        	stream_description TEXT, 
        	topic_name VARCHAR(200), 
        	bytewax_stream_name VARCHAR(200) NOT NULL, 
        	partitions INTEGER NOT NULL, 
        	replication_factor INTEGER NOT NULL, 
        	retention_time_ms BIGINT NOT NULL, 
        	retention_size_bytes BIGINT, 
        	cleanup_policy VARCHAR(20) NOT NULL, 
        	compression_type VARCHAR(20) NOT NULL, 
        	default_serialization VARCHAR(20) NOT NULL, 
        	event_category VARCHAR(100) NOT NULL, 
        	source_capability VARCHAR(100) NOT NULL, 
        	config_settings JSONB NOT NULL, 
        	status VARCHAR(20) NOT NULL, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	created_by VARCHAR(100) NOT NULL, 
        	PRIMARY KEY (stream_id), 
        	CONSTRAINT check_partitions_positive CHECK (partitions > 0), 
        	CONSTRAINT check_replication_positive CHECK (replication_factor > 0), 
        	CONSTRAINT check_retention_time_positive CHECK (retention_time_ms > 0), 
        	UNIQUE (stream_name)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_streams_capability ON es_streams (source_capability)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_streams_tenant_id ON es_streams (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_streams_bytewax_stream_name ON es_streams (bytewax_stream_name)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_streams_tenant ON es_streams (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_streams_status ON es_streams (status)
    """))
    op.execute(sa.text("""
        CREATE TABLE es_consumer_groups (
        	group_id VARCHAR(100) NOT NULL, 
        	group_name VARCHAR(200) NOT NULL, 
        	group_description TEXT, 
        	stream_id VARCHAR(100), 
        	session_timeout_ms INTEGER NOT NULL, 
        	heartbeat_interval_ms INTEGER NOT NULL, 
        	max_poll_interval_ms INTEGER NOT NULL, 
        	partition_assignment_strategy VARCHAR(50) NOT NULL, 
        	rebalance_timeout_ms INTEGER NOT NULL, 
        	active_consumers INTEGER NOT NULL, 
        	total_lag BIGINT NOT NULL, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	created_by VARCHAR(100) NOT NULL, 
        	PRIMARY KEY (group_id), 
        	CONSTRAINT uq_group_name_tenant UNIQUE (group_name, tenant_id), 
        	CONSTRAINT check_session_timeout_positive CHECK (session_timeout_ms > 0), 
        	CONSTRAINT check_heartbeat_positive CHECK (heartbeat_interval_ms > 0), 
        	CONSTRAINT check_active_consumers_positive CHECK (active_consumers >= 0), 
        	FOREIGN KEY(stream_id) REFERENCES es_streams (stream_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_consumer_groups_stream_id ON es_consumer_groups (stream_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_consumer_groups_tenant_id ON es_consumer_groups (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_consumer_groups_tenant ON es_consumer_groups (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE es_events (
        	event_id VARCHAR(30) NOT NULL, 
        	event_type VARCHAR(200) NOT NULL, 
        	event_version VARCHAR(20) NOT NULL, 
        	source_capability VARCHAR(100) NOT NULL, 
        	target_capability VARCHAR(100), 
        	aggregate_id VARCHAR(50) NOT NULL, 
        	aggregate_type VARCHAR(100) NOT NULL, 
        	sequence_number BIGINT NOT NULL, 
        	correlation_id VARCHAR(50), 
        	causation_id VARCHAR(30), 
        	event_timestamp TIMESTAMP WITH TIME ZONE NOT NULL, 
        	ingestion_timestamp TIMESTAMP WITH TIME ZONE NOT NULL, 
        	processed_timestamp TIMESTAMP WITH TIME ZONE, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	user_id VARCHAR(50), 
        	session_id VARCHAR(50), 
        	status VARCHAR(20) NOT NULL, 
        	priority VARCHAR(20) NOT NULL, 
        	retry_count INTEGER NOT NULL, 
        	max_retries INTEGER NOT NULL, 
        	payload JSONB NOT NULL, 
        	metadata JSONB NOT NULL, 
        	headers JSONB NOT NULL, 
        	schema_id VARCHAR(50), 
        	schema_version VARCHAR(20) NOT NULL, 
        	content_type VARCHAR(100) NOT NULL, 
        	serialization_format VARCHAR(20) NOT NULL, 
        	compression_type VARCHAR(20) NOT NULL, 
        	original_size INTEGER, 
        	compressed_size INTEGER, 
        	processing_duration_ms INTEGER, 
        	bytes_processed BIGINT, 
        	error_message TEXT, 
        	error_code VARCHAR(50), 
        	error_details JSONB, 
        	stream_id VARCHAR(100) NOT NULL, 
        	partition_key VARCHAR(200), 
        	offset_position BIGINT, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	created_by VARCHAR(50) NOT NULL, 
        	updated_by VARCHAR(50) NOT NULL, 
        	PRIMARY KEY (event_id), 
        	CONSTRAINT ck_es_events_retry_count_positive CHECK (retry_count >= 0), 
        	CONSTRAINT ck_es_events_max_retries_positive CHECK (max_retries >= 0), 
        	CONSTRAINT ck_es_events_sequence_positive CHECK (sequence_number > 0), 
        	CONSTRAINT ck_es_events_original_size_positive CHECK (original_size >= 0), 
        	CONSTRAINT ck_es_events_compressed_size_positive CHECK (compressed_size >= 0), 
        	FOREIGN KEY(stream_id) REFERENCES es_streams (stream_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_event_type ON es_events (event_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_events_aggregate ON es_events (aggregate_type, aggregate_id, sequence_number)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_status ON es_events (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_events_correlation ON es_events (correlation_id, event_timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_tenant_id ON es_events (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_sequence_number ON es_events (sequence_number)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_stream_id ON es_events (stream_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_source_capability ON es_events (source_capability)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_events_status_priority ON es_events (status, priority, event_timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_events_source_target ON es_events (source_capability, target_capability)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_events_stream_offset ON es_events (stream_id, offset_position)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_schema_id ON es_events (schema_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_priority ON es_events (priority)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_correlation_id ON es_events (correlation_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_target_capability ON es_events (target_capability)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_user_id ON es_events (user_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_causation_id ON es_events (causation_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_aggregate_id ON es_events (aggregate_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_events_tenant_type_timestamp ON es_events (tenant_id, event_type, event_timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_events_aggregate_type ON es_events (aggregate_type)
    """))
    op.execute(sa.text("""
        CREATE TABLE es_metrics (
        	metric_id VARCHAR(100) NOT NULL, 
        	metric_name VARCHAR(100) NOT NULL, 
        	metric_type VARCHAR(20) NOT NULL, 
        	stream_id VARCHAR(100), 
        	consumer_group_id VARCHAR(100), 
        	metric_value FLOAT NOT NULL, 
        	metric_unit VARCHAR(20), 
        	dimensions JSONB NOT NULL, 
        	time_bucket TIMESTAMP WITH TIME ZONE NOT NULL, 
        	aggregation_period VARCHAR(10) NOT NULL, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	PRIMARY KEY (metric_id), 
        	CONSTRAINT check_metric_type_valid CHECK (metric_type IN ('counter', 'gauge', 'histogram', 'timer')), 
        	FOREIGN KEY(stream_id) REFERENCES es_streams (stream_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_metrics_stream_time ON es_metrics (stream_id, time_bucket)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_metrics_group_time ON es_metrics (consumer_group_id, time_bucket)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_metrics_time_bucket ON es_metrics (time_bucket)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_metrics_stream_id ON es_metrics (stream_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_metrics_consumer_group_id ON es_metrics (consumer_group_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_metrics_metric_name ON es_metrics (metric_name)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_metrics_tenant_id ON es_metrics (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_metrics_tenant_time ON es_metrics (tenant_id, time_bucket)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_metrics_name_time ON es_metrics (metric_name, time_bucket)
    """))
    op.execute(sa.text("""
        CREATE TABLE es_stream_processors (
        	processor_id VARCHAR(50) NOT NULL, 
        	processor_name VARCHAR(200) NOT NULL, 
        	processor_type VARCHAR(50) NOT NULL, 
        	stream_id VARCHAR(100) NOT NULL, 
        	output_stream_id VARCHAR(100), 
        	description TEXT, 
        	processing_logic JSONB NOT NULL, 
        	configuration JSONB NOT NULL, 
        	filter_expression TEXT, 
        	transformation_function TEXT, 
        	aggregation_config JSONB, 
        	windowing_config JSONB, 
        	join_stream_id VARCHAR(100), 
        	join_condition TEXT, 
        	join_window_ms INTEGER, 
        	parallelism INTEGER NOT NULL, 
        	batch_size INTEGER NOT NULL, 
        	processing_timeout_ms INTEGER NOT NULL, 
        	checkpoint_interval_ms INTEGER NOT NULL, 
        	stateful BOOLEAN NOT NULL, 
        	state_store_config JSONB, 
        	changelog_stream VARCHAR(300), 
        	tenant_id VARCHAR(50) NOT NULL, 
        	owner_id VARCHAR(50) NOT NULL, 
        	status VARCHAR(20) NOT NULL, 
        	health_status VARCHAR(20) NOT NULL, 
        	last_checkpoint TIMESTAMP WITH TIME ZONE, 
        	messages_processed BIGINT NOT NULL, 
        	bytes_processed BIGINT NOT NULL, 
        	processing_errors BIGINT NOT NULL, 
        	output_messages BIGINT NOT NULL, 
        	throughput_msgs_sec INTEGER NOT NULL, 
        	latency_p95_ms INTEGER NOT NULL, 
        	cpu_usage_percent INTEGER NOT NULL, 
        	memory_usage_mb INTEGER NOT NULL, 
        	error_tolerance VARCHAR(20) NOT NULL, 
        	dead_letter_enabled BOOLEAN NOT NULL, 
        	dead_letter_stream VARCHAR(300), 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	created_by VARCHAR(50) NOT NULL, 
        	updated_by VARCHAR(50) NOT NULL, 
        	PRIMARY KEY (processor_id), 
        	CONSTRAINT ck_es_processors_parallelism_positive CHECK (parallelism > 0), 
        	CONSTRAINT ck_es_processors_batch_size_positive CHECK (batch_size > 0), 
        	CONSTRAINT ck_es_processors_messages_processed_non_negative CHECK (messages_processed >= 0), 
        	FOREIGN KEY(stream_id) REFERENCES es_streams (stream_id), 
        	FOREIGN KEY(output_stream_id) REFERENCES es_streams (stream_id), 
        	FOREIGN KEY(join_stream_id) REFERENCES es_streams (stream_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_processors_tenant_type ON es_stream_processors (tenant_id, processor_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_stream_processors_processor_name ON es_stream_processors (processor_name)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_stream_processors_stream_id ON es_stream_processors (stream_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_stream_processors_output_stream_id ON es_stream_processors (output_stream_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_stream_processors_owner_id ON es_stream_processors (owner_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_stream_processors_status ON es_stream_processors (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_processors_status ON es_stream_processors (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_processors_stream ON es_stream_processors (stream_id, status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_stream_processors_tenant_id ON es_stream_processors (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE es_audit_logs (
        	audit_id VARCHAR(100) NOT NULL, 
        	event_id VARCHAR(36), 
        	operation_type VARCHAR(50) NOT NULL, 
        	operation_status VARCHAR(20) NOT NULL, 
        	actor_type VARCHAR(20) NOT NULL, 
        	actor_id VARCHAR(100) NOT NULL, 
        	source_ip VARCHAR(45), 
        	user_agent VARCHAR(500), 
        	session_id VARCHAR(100), 
        	operation_details JSONB NOT NULL, 
        	error_message TEXT, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	PRIMARY KEY (audit_id), 
        	FOREIGN KEY(event_id) REFERENCES es_events (event_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_audit_logs_tenant_id ON es_audit_logs (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_audit_status ON es_audit_logs (operation_status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_audit_logs_event_id ON es_audit_logs (event_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_audit_logs_actor_id ON es_audit_logs (actor_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_audit_tenant_time ON es_audit_logs (tenant_id, created_at)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_audit_actor ON es_audit_logs (actor_id, created_at)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_audit_operation ON es_audit_logs (operation_type, created_at)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_audit_logs_operation_type ON es_audit_logs (operation_type)
    """))
    op.execute(sa.text("""
        CREATE TABLE es_event_processing_history (
        	history_id VARCHAR(50) NOT NULL, 
        	event_id VARCHAR(30) NOT NULL, 
        	processor_name VARCHAR(200) NOT NULL, 
        	processor_version VARCHAR(50) NOT NULL, 
        	processing_stage VARCHAR(100) NOT NULL, 
        	status VARCHAR(20) NOT NULL, 
        	started_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	completed_at TIMESTAMP WITH TIME ZONE, 
        	duration_ms INTEGER, 
        	input_data JSONB, 
        	output_data JSONB, 
        	transformation_applied JSONB, 
        	error_message TEXT, 
        	error_code VARCHAR(50), 
        	error_details JSONB, 
        	stack_trace TEXT, 
        	cpu_time_ms INTEGER, 
        	memory_used_mb INTEGER, 
        	io_operations INTEGER, 
        	retry_attempt INTEGER NOT NULL, 
        	retry_reason VARCHAR(200), 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	created_by VARCHAR(50) NOT NULL, 
        	PRIMARY KEY (history_id), 
        	CONSTRAINT ck_es_processing_history_retry_attempt_non_negative CHECK (retry_attempt >= 0), 
        	FOREIGN KEY(event_id) REFERENCES es_events (event_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_event_processing_history_status ON es_event_processing_history (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_processing_history_event_status ON es_event_processing_history (event_id, status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_processing_history_processor ON es_event_processing_history (processor_name, processing_stage)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_processing_history_timing ON es_event_processing_history (started_at, completed_at)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_event_processing_history_event_id ON es_event_processing_history (event_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE es_stream_assignments (
        	assignment_id VARCHAR(50) NOT NULL, 
        	event_id VARCHAR(30) NOT NULL, 
        	stream_id VARCHAR(100) NOT NULL, 
        	partition_id INTEGER NOT NULL, 
        	"offset" BIGINT NOT NULL, 
        	key VARCHAR(500), 
        	assignment_reason VARCHAR(100) NOT NULL, 
        	assignment_rules JSONB NOT NULL, 
        	published_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	consumed_count INTEGER NOT NULL, 
        	last_consumed_at TIMESTAMP WITH TIME ZONE, 
        	delivery_attempts INTEGER NOT NULL, 
        	successful_deliveries INTEGER NOT NULL, 
        	failed_deliveries INTEGER NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	created_by VARCHAR(50) NOT NULL, 
        	PRIMARY KEY (assignment_id), 
        	CONSTRAINT uk_es_stream_assignments_event_stream UNIQUE (event_id, stream_id), 
        	CONSTRAINT ck_es_stream_assignments_partition_id_non_negative CHECK (partition_id >= 0), 
        	CONSTRAINT ck_es_stream_assignments_offset_non_negative CHECK (offset >= 0), 
        	CONSTRAINT ck_es_stream_assignments_consumed_count_non_negative CHECK (consumed_count >= 0), 
        	FOREIGN KEY(event_id) REFERENCES es_events (event_id), 
        	FOREIGN KEY(stream_id) REFERENCES es_streams (stream_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_stream_assignments_published ON es_stream_assignments (published_at)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_stream_assignments_stream_partition ON es_stream_assignments (stream_id, partition_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_stream_assignments_stream_id ON es_stream_assignments (stream_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_stream_assignments_event_id ON es_stream_assignments (event_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_stream_assignments_offset ON es_stream_assignments (stream_id, "offset")
    """))
    op.execute(sa.text("""
        CREATE TABLE es_subscriptions (
        	subscription_id VARCHAR(100) NOT NULL, 
        	subscription_name VARCHAR(200) NOT NULL, 
        	subscription_description TEXT, 
        	stream_id VARCHAR(100) NOT NULL, 
        	consumer_group_id VARCHAR(100) NOT NULL, 
        	consumer_name VARCHAR(200) NOT NULL, 
        	event_type_patterns JSONB NOT NULL, 
        	filter_criteria JSONB NOT NULL, 
        	delivery_mode VARCHAR(20) NOT NULL, 
        	batch_size INTEGER NOT NULL, 
        	max_wait_time_ms INTEGER NOT NULL, 
        	start_position VARCHAR(20) NOT NULL, 
        	specific_offset BIGINT, 
        	retry_policy JSONB NOT NULL, 
        	dead_letter_enabled BOOLEAN NOT NULL, 
        	dead_letter_stream VARCHAR(200), 
        	webhook_url VARCHAR(500), 
        	webhook_headers JSONB, 
        	webhook_timeout_ms INTEGER, 
        	status VARCHAR(20) NOT NULL, 
        	last_consumed_offset BIGINT, 
        	last_consumed_at TIMESTAMP WITH TIME ZONE, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	created_by VARCHAR(100) NOT NULL, 
        	PRIMARY KEY (subscription_id), 
        	CONSTRAINT uq_subscription_name_tenant UNIQUE (subscription_name, tenant_id), 
        	CONSTRAINT check_batch_size_positive CHECK (batch_size > 0), 
        	CONSTRAINT check_wait_time_positive CHECK (max_wait_time_ms > 0), 
        	FOREIGN KEY(stream_id) REFERENCES es_streams (stream_id), 
        	FOREIGN KEY(consumer_group_id) REFERENCES es_consumer_groups (group_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_subscriptions_stream_id ON es_subscriptions (stream_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_subscriptions_tenant ON es_subscriptions (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_subscriptions_consumer_group ON es_subscriptions (consumer_group_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_subscriptions_consumer_group_id ON es_subscriptions (consumer_group_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_subscriptions_status ON es_subscriptions (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_es_subscriptions_stream_status ON es_subscriptions (stream_id, status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_es_subscriptions_tenant_id ON es_subscriptions (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_alerts (
        	alert_id VARCHAR(50) NOT NULL, 
        	alert_name VARCHAR(255) NOT NULL, 
        	condition TEXT NOT NULL, 
        	threshold FLOAT, 
        	severity VARCHAR(50) NOT NULL, 
        	enabled BOOLEAN, 
        	is_active BOOLEAN, 
        	last_triggered_at TIMESTAMP WITH TIME ZONE, 
        	last_resolved_at TIMESTAMP WITH TIME ZONE, 
        	trigger_count INTEGER, 
        	notification_channels JSONB, 
        	notification_template TEXT, 
        	description TEXT, 
        	metadata JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (alert_id), 
        	CONSTRAINT uq_alert_name_tenant UNIQUE (alert_name, tenant_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_alerts_severity ON sm_alerts (severity)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_alerts_tenant ON sm_alerts (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_alerts_active ON sm_alerts (is_active)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_alerts_enabled ON sm_alerts (enabled)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_autonomous_decisions (
        	decision_id VARCHAR(50) NOT NULL, 
        	decision_type VARCHAR(100) NOT NULL, 
        	trigger_event JSONB NOT NULL, 
        	analyzed_data JSONB NOT NULL, 
        	decision_rationale TEXT NOT NULL, 
        	actions_executed JSONB NOT NULL, 
        	rollback_plan JSONB NOT NULL, 
        	execution_status VARCHAR(50), 
        	execution_results JSONB, 
        	success_metrics JSONB, 
        	rollback_triggered BOOLEAN, 
        	decision_confidence FLOAT NOT NULL, 
        	feedback_score FLOAT, 
        	learning_data JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (decision_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_autonomous_decisions_status ON sm_autonomous_decisions (execution_status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_autonomous_decisions_tenant ON sm_autonomous_decisions (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_autonomous_decisions_type ON sm_autonomous_decisions (decision_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_autonomous_decisions_confidence ON sm_autonomous_decisions (decision_confidence)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_certificates (
        	certificate_id VARCHAR(50) NOT NULL, 
        	certificate_name VARCHAR(255) NOT NULL, 
        	common_name VARCHAR(255) NOT NULL, 
        	subject_alt_names JSONB, 
        	issuer VARCHAR(500), 
        	serial_number VARCHAR(100), 
        	certificate_pem TEXT NOT NULL, 
        	private_key_pem TEXT, 
        	ca_certificate_pem TEXT, 
        	not_before TIMESTAMP WITH TIME ZONE NOT NULL, 
        	not_after TIMESTAMP WITH TIME ZONE NOT NULL, 
        	status VARCHAR(50), 
        	auto_renew BOOLEAN, 
        	renewal_days_before INTEGER, 
        	metadata JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (certificate_id), 
        	CONSTRAINT uq_cert_name_tenant UNIQUE (certificate_name, tenant_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_certificates_not_after ON sm_certificates (not_after)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_certificates_status ON sm_certificates (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_certificates_common_name ON sm_certificates (common_name)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_certificates_tenant ON sm_certificates (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_collaborative_sessions (
        	session_id VARCHAR(50) NOT NULL, 
        	session_name VARCHAR(255) NOT NULL, 
        	problem_description TEXT NOT NULL, 
        	affected_services JSONB NOT NULL, 
        	session_type VARCHAR(50), 
        	active_participants JSONB, 
        	participant_roles JSONB, 
        	session_leader VARCHAR(255), 
        	shared_annotations JSONB, 
        	investigation_timeline JSONB, 
        	findings JSONB, 
        	resolution_actions JSONB, 
        	ai_suggestions JSONB, 
        	root_cause_analysis JSONB, 
        	automated_diagnostics JSONB, 
        	status VARCHAR(50), 
        	resolution_confidence FLOAT, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	started_at TIMESTAMP WITH TIME ZONE, 
        	ended_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (session_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_collaborative_sessions_status ON sm_collaborative_sessions (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_collaborative_sessions_tenant ON sm_collaborative_sessions (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_collaborative_sessions_type ON sm_collaborative_sessions (session_type)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_configurations (
        	config_id VARCHAR(50) NOT NULL, 
        	config_name VARCHAR(255) NOT NULL, 
        	config_type VARCHAR(50) NOT NULL, 
        	configuration JSONB NOT NULL, 
        	schema_version VARCHAR(20), 
        	enabled BOOLEAN, 
        	validated BOOLEAN, 
        	validation_errors JSONB, 
        	description TEXT, 
        	metadata JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (config_id), 
        	CONSTRAINT uq_config_name_tenant UNIQUE (config_name, tenant_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_configurations_enabled ON sm_configurations (enabled)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_configurations_type ON sm_configurations (config_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_configurations_tenant ON sm_configurations (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_federated_insights (
        	insight_id VARCHAR(50) NOT NULL, 
        	insight_type VARCHAR(100) NOT NULL, 
        	global_pattern JSONB NOT NULL, 
        	local_adaptation JSONB NOT NULL, 
        	aggregated_metrics JSONB NOT NULL, 
        	optimization_impact JSONB, 
        	deployment_clusters JSONB, 
        	adoption_rate FLOAT, 
        	model_version VARCHAR(50) NOT NULL, 
        	contribution_weight FLOAT, 
        	privacy_preserved BOOLEAN, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (insight_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_federated_insights_type ON sm_federated_insights (insight_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_federated_insights_tenant ON sm_federated_insights (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_federated_insights_model ON sm_federated_insights (model_version)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_intelligent_topology (
        	topology_id VARCHAR(50) NOT NULL, 
        	mesh_version VARCHAR(50) NOT NULL, 
        	topology_snapshot JSONB NOT NULL, 
        	service_dependencies JSONB NOT NULL, 
        	traffic_patterns JSONB, 
        	failure_predictions JSONB, 
        	optimization_recommendations JSONB, 
        	scaling_predictions JSONB, 
        	performance_insights JSONB, 
        	ml_model_version VARCHAR(50), 
        	prediction_confidence FLOAT, 
        	learning_feedback JSONB, 
        	active_viewers JSONB, 
        	collaborative_annotations JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (topology_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_intelligent_topology_confidence ON sm_intelligent_topology (prediction_confidence)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_intelligent_topology_tenant ON sm_intelligent_topology (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_intelligent_topology_version ON sm_intelligent_topology (mesh_version)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_nl_policies (
        	nl_policy_id VARCHAR(50) NOT NULL, 
        	policy_name VARCHAR(255) NOT NULL, 
        	natural_language_intent TEXT NOT NULL, 
        	processed_intent JSONB NOT NULL, 
        	compiled_rules JSONB NOT NULL, 
        	confidence_score FLOAT NOT NULL, 
        	affected_services JSONB, 
        	affected_routes JSONB, 
        	deployment_strategy VARCHAR(100), 
        	status VARCHAR(50), 
        	validation_results JSONB, 
        	compliance_mappings JSONB, 
        	ai_model_version VARCHAR(50), 
        	processing_metadata JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (nl_policy_id), 
        	CONSTRAINT uq_nl_policy_name_tenant UNIQUE (policy_name, tenant_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_nl_policies_confidence ON sm_nl_policies (confidence_score)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_nl_policies_tenant ON sm_nl_policies (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_nl_policies_status ON sm_nl_policies (status)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_predictive_alerts (
        	alert_id VARCHAR(50) NOT NULL, 
        	prediction_type VARCHAR(100) NOT NULL, 
        	predicted_event JSONB NOT NULL, 
        	prediction_confidence FLOAT NOT NULL, 
        	predicted_time_to_failure INTEGER, 
        	impact_assessment JSONB NOT NULL, 
        	suggested_actions JSONB, 
        	auto_remediation_enabled BOOLEAN, 
        	remediation_executed JSONB, 
        	prediction_accuracy FLOAT, 
        	actual_outcome JSONB, 
        	feedback_incorporated BOOLEAN, 
        	status VARCHAR(50), 
        	escalation_level INTEGER, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (alert_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_predictive_alerts_confidence ON sm_predictive_alerts (prediction_confidence)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_predictive_alerts_status ON sm_predictive_alerts (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_predictive_alerts_tenant ON sm_predictive_alerts (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_predictive_alerts_type ON sm_predictive_alerts (prediction_type)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_rate_limiters (
        	rate_limiter_id VARCHAR(50) NOT NULL, 
        	limiter_name VARCHAR(255) NOT NULL, 
        	requests_per_second INTEGER NOT NULL, 
        	burst_size INTEGER NOT NULL, 
        	window_size_seconds INTEGER, 
        	scope VARCHAR(50), 
        	key_expression VARCHAR(1000), 
        	rate_limit_response_code INTEGER, 
        	rate_limit_response_body TEXT, 
        	rate_limit_headers JSONB, 
        	enabled BOOLEAN, 
        	enforcement_mode VARCHAR(50), 
        	metadata JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (rate_limiter_id), 
        	CONSTRAINT uq_rate_limiter_name_tenant UNIQUE (limiter_name, tenant_id), 
        	CONSTRAINT ck_rl_positive_rps CHECK (requests_per_second > 0), 
        	CONSTRAINT ck_rl_positive_burst CHECK (burst_size > 0)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_rate_limiters_enabled ON sm_rate_limiters (enabled)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_rate_limiters_scope ON sm_rate_limiters (scope)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_rate_limiters_tenant ON sm_rate_limiters (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_security_policies (
        	security_policy_id VARCHAR(50) NOT NULL, 
        	policy_name VARCHAR(255) NOT NULL, 
        	policy_type VARCHAR(50) NOT NULL, 
        	rules JSONB NOT NULL, 
        	enforcement_mode VARCHAR(50), 
        	allowed_sources JSONB, 
        	denied_sources JSONB, 
        	allowed_methods JSONB, 
        	allowed_paths JSONB, 
        	require_authentication BOOLEAN, 
        	authentication_methods JSONB, 
        	enabled BOOLEAN, 
        	priority INTEGER, 
        	description TEXT, 
        	metadata JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (security_policy_id), 
        	CONSTRAINT uq_security_policy_name_tenant UNIQUE (policy_name, tenant_id), 
        	CONSTRAINT ck_security_policy_positive_priority CHECK (priority > 0)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_security_policies_priority ON sm_security_policies (priority)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_security_policies_enabled ON sm_security_policies (enabled)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_security_policies_type ON sm_security_policies (policy_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_security_policies_tenant ON sm_security_policies (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_services (
        	service_id VARCHAR(50) NOT NULL, 
        	service_name VARCHAR(255) NOT NULL, 
        	service_version VARCHAR(50) NOT NULL, 
        	namespace VARCHAR(255), 
        	description TEXT, 
        	tags JSONB, 
        	metadata JSONB, 
        	status VARCHAR(50) NOT NULL, 
        	health_status VARCHAR(50), 
        	last_health_check TIMESTAMP WITH TIME ZONE, 
        	configuration JSONB, 
        	environment VARCHAR(100), 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (service_id), 
        	CONSTRAINT uq_service_name_version_namespace_tenant UNIQUE (service_name, service_version, namespace, tenant_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_services_status ON sm_services (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_services_environment ON sm_services (environment)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_services_health ON sm_services (health_status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_services_tenant ON sm_services (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_traces (
        	trace_id VARCHAR(100) NOT NULL, 
        	span_id VARCHAR(50) NOT NULL, 
        	parent_span_id VARCHAR(50), 
        	service_name VARCHAR(255) NOT NULL, 
        	operation_name VARCHAR(255) NOT NULL, 
        	start_time TIMESTAMP WITH TIME ZONE NOT NULL, 
        	end_time TIMESTAMP WITH TIME ZONE, 
        	duration_ms FLOAT, 
        	status VARCHAR(50), 
        	error_message TEXT, 
        	http_method VARCHAR(10), 
        	http_url VARCHAR(1000), 
        	http_status_code INTEGER, 
        	user_agent VARCHAR(500), 
        	tags JSONB, 
        	logs JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	PRIMARY KEY (trace_id, span_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_traces_start_time ON sm_traces (start_time)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_traces_operation ON sm_traces (operation_name)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_traces_service_start ON sm_traces (service_name, start_time)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_traces_status ON sm_traces (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_traces_tenant ON sm_traces (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_endpoints (
        	endpoint_id VARCHAR(50) NOT NULL, 
        	service_id VARCHAR(50) NOT NULL, 
        	host VARCHAR(255) NOT NULL, 
        	port INTEGER NOT NULL, 
        	protocol VARCHAR(50) NOT NULL, 
        	path VARCHAR(500), 
        	weight INTEGER, 
        	enabled BOOLEAN, 
        	metadata JSONB, 
        	health_check_path VARCHAR(500), 
        	health_check_interval INTEGER, 
        	health_check_timeout INTEGER, 
        	healthy_threshold INTEGER, 
        	unhealthy_threshold INTEGER, 
        	tls_enabled BOOLEAN, 
        	tls_verify BOOLEAN, 
        	certificate_id VARCHAR(50), 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (endpoint_id), 
        	CONSTRAINT uq_endpoint_service_host_port_path UNIQUE (service_id, host, port, path), 
        	CONSTRAINT ck_endpoint_valid_port CHECK (port > 0 AND port <= 65535), 
        	FOREIGN KEY(service_id) REFERENCES sm_services (service_id), 
        	FOREIGN KEY(certificate_id) REFERENCES sm_certificates (certificate_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_endpoints_service ON sm_endpoints (service_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_endpoints_tenant ON sm_endpoints (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_endpoints_enabled ON sm_endpoints (enabled)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_load_balancers (
        	load_balancer_id VARCHAR(50) NOT NULL, 
        	load_balancer_name VARCHAR(255) NOT NULL, 
        	service_id VARCHAR(50), 
        	algorithm VARCHAR(50) NOT NULL, 
        	session_affinity BOOLEAN, 
        	session_affinity_cookie VARCHAR(100), 
        	health_check_enabled BOOLEAN, 
        	health_check_interval INTEGER, 
        	health_check_timeout INTEGER, 
        	healthy_threshold INTEGER, 
        	unhealthy_threshold INTEGER, 
        	circuit_breaker_enabled BOOLEAN, 
        	failure_threshold INTEGER, 
        	recovery_timeout INTEGER, 
        	half_open_requests INTEGER, 
        	max_connections INTEGER, 
        	max_pending_requests INTEGER, 
        	max_requests_per_connection INTEGER, 
        	connection_timeout_ms INTEGER, 
        	configuration JSONB, 
        	enabled BOOLEAN, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (load_balancer_id), 
        	CONSTRAINT uq_lb_name_tenant UNIQUE (load_balancer_name, tenant_id), 
        	CONSTRAINT ck_lb_positive_failure_threshold CHECK (failure_threshold > 0), 
        	CONSTRAINT ck_lb_positive_max_connections CHECK (max_connections > 0), 
        	FOREIGN KEY(service_id) REFERENCES sm_services (service_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_load_balancers_algorithm ON sm_load_balancers (algorithm)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_load_balancers_tenant ON sm_load_balancers (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_load_balancers_enabled ON sm_load_balancers (enabled)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_metrics (
        	metric_id VARCHAR(50) NOT NULL, 
        	service_id VARCHAR(50), 
        	metric_name VARCHAR(255) NOT NULL, 
        	metric_type VARCHAR(50) NOT NULL, 
        	labels JSONB, 
        	value FLOAT NOT NULL, 
        	timestamp TIMESTAMP WITH TIME ZONE NOT NULL, 
        	request_count INTEGER, 
        	error_count INTEGER, 
        	response_time_ms FLOAT, 
        	status_code INTEGER, 
        	metadata JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (metric_id), 
        	FOREIGN KEY(service_id) REFERENCES sm_services (service_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_metrics_tenant ON sm_metrics (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_metrics_service_timestamp ON sm_metrics (service_id, timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_metrics_timestamp ON sm_metrics (timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_metrics_name_timestamp ON sm_metrics (metric_name, timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_metrics_type ON sm_metrics (metric_type)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_routes (
        	route_id VARCHAR(50) NOT NULL, 
        	route_name VARCHAR(255) NOT NULL, 
        	service_id VARCHAR(50), 
        	match_type VARCHAR(50) NOT NULL, 
        	match_value VARCHAR(1000) NOT NULL, 
        	match_headers JSONB, 
        	match_query JSONB, 
        	destination_services JSONB NOT NULL, 
        	backup_services JSONB, 
        	timeout_ms INTEGER, 
        	retry_attempts INTEGER, 
        	retry_timeout_ms INTEGER, 
        	priority INTEGER, 
        	enabled BOOLEAN, 
        	request_headers_add JSONB, 
        	request_headers_remove JSONB, 
        	response_headers_add JSONB, 
        	response_headers_remove JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (route_id), 
        	CONSTRAINT uq_route_name_tenant UNIQUE (route_name, tenant_id), 
        	CONSTRAINT ck_route_positive_priority CHECK (priority > 0), 
        	FOREIGN KEY(service_id) REFERENCES sm_services (service_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_routes_enabled ON sm_routes (enabled)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_routes_tenant ON sm_routes (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_routes_priority ON sm_routes (priority)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_routes_match_type ON sm_routes (match_type)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_topology (
        	topology_id VARCHAR(50) NOT NULL, 
        	source_service_id VARCHAR(50) NOT NULL, 
        	target_service_id VARCHAR(50) NOT NULL, 
        	relationship_type VARCHAR(50), 
        	weight FLOAT, 
        	protocol VARCHAR(50), 
        	port INTEGER, 
        	endpoint_path VARCHAR(500), 
        	avg_response_time_ms FLOAT, 
        	request_count INTEGER, 
        	error_count INTEGER, 
        	status VARCHAR(50), 
        	last_communication_at TIMESTAMP WITH TIME ZONE, 
        	metadata JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (topology_id), 
        	CONSTRAINT uq_topology_source_target_tenant UNIQUE (source_service_id, target_service_id, tenant_id), 
        	FOREIGN KEY(source_service_id) REFERENCES sm_services (service_id), 
        	FOREIGN KEY(target_service_id) REFERENCES sm_services (service_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_topology_type ON sm_topology (relationship_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_topology_target ON sm_topology (target_service_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_topology_tenant ON sm_topology (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_topology_source ON sm_topology (source_service_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_health_checks (
        	health_check_id VARCHAR(50) NOT NULL, 
        	service_id VARCHAR(50) NOT NULL, 
        	endpoint_id VARCHAR(50), 
        	check_type VARCHAR(50), 
        	check_url VARCHAR(1000), 
        	check_interval INTEGER, 
        	check_timeout INTEGER, 
        	status VARCHAR(50) NOT NULL, 
        	response_time_ms FLOAT, 
        	status_code INTEGER, 
        	response_body TEXT, 
        	error_message TEXT, 
        	consecutive_successes INTEGER, 
        	consecutive_failures INTEGER, 
        	last_check_at TIMESTAMP WITH TIME ZONE, 
        	last_success_at TIMESTAMP WITH TIME ZONE, 
        	last_failure_at TIMESTAMP WITH TIME ZONE, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (health_check_id), 
        	FOREIGN KEY(service_id) REFERENCES sm_services (service_id), 
        	FOREIGN KEY(endpoint_id) REFERENCES sm_endpoints (endpoint_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_health_checks_last_check ON sm_health_checks (last_check_at)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_health_checks_status ON sm_health_checks (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_health_checks_service ON sm_health_checks (service_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_health_checks_tenant ON sm_health_checks (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE sm_policies (
        	policy_id VARCHAR(50) NOT NULL, 
        	policy_name VARCHAR(255) NOT NULL, 
        	policy_type VARCHAR(50) NOT NULL, 
        	route_id VARCHAR(50), 
        	configuration JSONB NOT NULL, 
        	enabled BOOLEAN, 
        	priority INTEGER, 
        	rate_limit_requests INTEGER, 
        	rate_limit_window_seconds INTEGER, 
        	rate_limit_burst INTEGER, 
        	auth_required BOOLEAN, 
        	auth_config JSONB, 
        	description TEXT, 
        	metadata JSONB, 
        	tenant_id VARCHAR(50) NOT NULL, 
        	created_by VARCHAR(255) NOT NULL, 
        	updated_by VARCHAR(255), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (policy_id), 
        	CONSTRAINT uq_policy_name_tenant UNIQUE (policy_name, tenant_id), 
        	CONSTRAINT ck_policy_positive_priority CHECK (priority > 0), 
        	FOREIGN KEY(route_id) REFERENCES sm_routes (route_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_policies_tenant ON sm_policies (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_policies_enabled ON sm_policies (enabled)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_policies_priority ON sm_policies (priority)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_sm_policies_type ON sm_policies (policy_type)
    """))
    op.execute(sa.text("""
        CREATE TABLE so_oe_customer (
        	customer_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	customer_number VARCHAR(20) NOT NULL, 
        	customer_name VARCHAR(200) NOT NULL, 
        	customer_type VARCHAR(50), 
        	contact_name VARCHAR(100), 
        	email VARCHAR(100), 
        	phone VARCHAR(50), 
        	mobile VARCHAR(50), 
        	fax VARCHAR(50), 
        	website VARCHAR(200), 
        	billing_address_line1 VARCHAR(100), 
        	billing_address_line2 VARCHAR(100), 
        	billing_city VARCHAR(50), 
        	billing_state_province VARCHAR(50), 
        	billing_postal_code VARCHAR(20), 
        	billing_country VARCHAR(50), 
        	shipping_address_line1 VARCHAR(100), 
        	shipping_address_line2 VARCHAR(100), 
        	shipping_city VARCHAR(50), 
        	shipping_state_province VARCHAR(50), 
        	shipping_postal_code VARCHAR(20), 
        	shipping_country VARCHAR(50), 
        	preferred_payment_method VARCHAR(50), 
        	preferred_shipping_method VARCHAR(50), 
        	payment_terms_code VARCHAR(20), 
        	price_level_id VARCHAR(36), 
        	credit_limit DECIMAL(15, 2), 
        	credit_hold BOOLEAN, 
        	credit_rating VARCHAR(10), 
        	credit_check_required BOOLEAN, 
        	tax_id VARCHAR(50), 
        	tax_exempt BOOLEAN, 
        	tax_exempt_number VARCHAR(50), 
        	default_tax_code VARCHAR(20), 
        	sales_rep_id VARCHAR(36), 
        	territory_id VARCHAR(36), 
        	customer_since DATE, 
        	is_active BOOLEAN, 
        	allow_backorders BOOLEAN, 
        	require_po_number BOOLEAN, 
        	auto_approve_orders BOOLEAN, 
        	order_approval_limit DECIMAL(15, 2), 
        	currency_code VARCHAR(3), 
        	current_balance DECIMAL(15, 2), 
        	ytd_orders DECIMAL(15, 2), 
        	last_order_date DATE, 
        	total_orders INTEGER, 
        	ar_customer_id VARCHAR(36), 
        	notes TEXT, 
        	internal_notes TEXT, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(100), 
        	updated_by VARCHAR(100), 
        	id VARCHAR(36) NOT NULL, 
        	PRIMARY KEY (id), 
        	CONSTRAINT uq_soe_customer_number_tenant UNIQUE (tenant_id, customer_number)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_customer_ar_customer_id ON so_oe_customer (ar_customer_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_customer_tenant_id ON so_oe_customer (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_customer_sales_rep_id ON so_oe_customer (sales_rep_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_customer_email ON so_oe_customer (email)
    """))
    op.execute(sa.text("""
        CREATE UNIQUE INDEX ix_so_oe_customer_customer_id ON so_oe_customer (customer_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_customer_customer_number ON so_oe_customer (customer_number)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_customer_customer_name ON so_oe_customer (customer_name)
    """))
    op.execute(sa.text("""
        CREATE TABLE so_oe_order_sequence (
        	sequence_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	sequence_name VARCHAR(100) NOT NULL, 
        	order_type VARCHAR(20) NOT NULL, 
        	prefix VARCHAR(10), 
        	suffix VARCHAR(10), 
        	number_length INTEGER, 
        	current_number INTEGER, 
        	increment_by INTEGER, 
        	reset_period VARCHAR(10), 
        	last_reset_date DATE, 
        	is_active BOOLEAN, 
        	zero_pad BOOLEAN, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(100), 
        	updated_by VARCHAR(100), 
        	id VARCHAR(36) NOT NULL, 
        	PRIMARY KEY (id), 
        	CONSTRAINT uq_soe_sequence_type_tenant UNIQUE (tenant_id, order_type)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_sequence_order_type ON so_oe_order_sequence (order_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_sequence_tenant_id ON so_oe_order_sequence (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE UNIQUE INDEX ix_so_oe_order_sequence_sequence_id ON so_oe_order_sequence (sequence_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE so_oe_price_level (
        	price_level_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	level_code VARCHAR(20) NOT NULL, 
        	level_name VARCHAR(100) NOT NULL, 
        	description TEXT, 
        	discount_percentage DECIMAL(5, 2), 
        	markup_percentage DECIMAL(5, 2), 
        	price_calculation_method VARCHAR(20), 
        	minimum_order_amount DECIMAL(15, 2), 
        	minimum_annual_volume DECIMAL(15, 2), 
        	customer_type VARCHAR(50), 
        	effective_date DATE, 
        	expiration_date DATE, 
        	is_active BOOLEAN, 
        	is_default BOOLEAN, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(100), 
        	updated_by VARCHAR(100), 
        	id VARCHAR(36) NOT NULL, 
        	PRIMARY KEY (id), 
        	CONSTRAINT uq_soe_price_level_code_tenant UNIQUE (tenant_id, level_code)
        )
    """))
    op.execute(sa.text("""
        CREATE UNIQUE INDEX ix_so_oe_price_level_price_level_id ON so_oe_price_level (price_level_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_price_level_level_code ON so_oe_price_level (level_code)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_price_level_tenant_id ON so_oe_price_level (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE so_oe_order_template (
        	template_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	template_name VARCHAR(200) NOT NULL, 
        	description TEXT, 
        	template_type VARCHAR(20), 
        	customer_id VARCHAR(36), 
        	is_active BOOLEAN, 
        	is_public BOOLEAN, 
        	usage_count INTEGER, 
        	last_used_date TIMESTAMP WITHOUT TIME ZONE, 
        	default_ship_to_id VARCHAR(36), 
        	default_requested_date_offset INTEGER, 
        	notes TEXT, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(100), 
        	updated_by VARCHAR(100), 
        	id VARCHAR(36) NOT NULL, 
        	PRIMARY KEY (id), 
        	FOREIGN KEY(customer_id) REFERENCES so_oe_customer (customer_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_template_customer_id ON so_oe_order_template (customer_id)
    """))
    op.execute(sa.text("""
        CREATE UNIQUE INDEX ix_so_oe_order_template_template_id ON so_oe_order_template (template_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_template_tenant_id ON so_oe_order_template (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE so_oe_ship_to_address (
        	ship_to_id VARCHAR(36) NOT NULL, 
        	customer_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	address_name VARCHAR(100) NOT NULL, 
        	contact_name VARCHAR(100), 
        	address_line1 VARCHAR(100) NOT NULL, 
        	address_line2 VARCHAR(100), 
        	city VARCHAR(50) NOT NULL, 
        	state_province VARCHAR(50) NOT NULL, 
        	postal_code VARCHAR(20) NOT NULL, 
        	country VARCHAR(50) NOT NULL, 
        	phone VARCHAR(50), 
        	email VARCHAR(100), 
        	preferred_carrier VARCHAR(50), 
        	preferred_service_level VARCHAR(50), 
        	delivery_instructions TEXT, 
        	is_validated BOOLEAN, 
        	validation_date TIMESTAMP WITHOUT TIME ZONE, 
        	validation_service VARCHAR(50), 
        	is_default BOOLEAN, 
        	is_active BOOLEAN, 
        	requires_appointment BOOLEAN, 
        	loading_dock_available BOOLEAN, 
        	latitude DECIMAL(10, 8), 
        	longitude DECIMAL(11, 8), 
        	timezone VARCHAR(50), 
        	tax_jurisdiction VARCHAR(100), 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(100), 
        	updated_by VARCHAR(100), 
        	id VARCHAR(36) NOT NULL, 
        	PRIMARY KEY (id), 
        	FOREIGN KEY(customer_id) REFERENCES so_oe_customer (customer_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_ship_to_address_customer_id ON so_oe_ship_to_address (customer_id)
    """))
    op.execute(sa.text("""
        CREATE UNIQUE INDEX ix_so_oe_ship_to_address_ship_to_id ON so_oe_ship_to_address (ship_to_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_ship_to_address_tenant_id ON so_oe_ship_to_address (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE so_oe_order_template_line (
        	template_line_id VARCHAR(36) NOT NULL, 
        	template_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	line_number INTEGER NOT NULL, 
        	description TEXT, 
        	item_id VARCHAR(36), 
        	item_code VARCHAR(50) NOT NULL, 
        	item_description VARCHAR(200), 
        	default_quantity DECIMAL(12, 4), 
        	minimum_quantity DECIMAL(12, 4), 
        	maximum_quantity DECIMAL(12, 4), 
        	is_required BOOLEAN, 
        	allow_quantity_change BOOLEAN, 
        	allow_substitution BOOLEAN, 
        	notes TEXT, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(100), 
        	updated_by VARCHAR(100), 
        	id VARCHAR(36) NOT NULL, 
        	PRIMARY KEY (id), 
        	FOREIGN KEY(template_id) REFERENCES so_oe_order_template (template_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_template_line_item_id ON so_oe_order_template_line (item_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_template_line_tenant_id ON so_oe_order_template_line (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_template_line_template_id ON so_oe_order_template_line (template_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_template_line_item_code ON so_oe_order_template_line (item_code)
    """))
    op.execute(sa.text("""
        CREATE UNIQUE INDEX ix_so_oe_order_template_line_template_line_id ON so_oe_order_template_line (template_line_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE so_oe_sales_order (
        	order_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	order_number VARCHAR(50) NOT NULL, 
        	description TEXT, 
        	order_type VARCHAR(20), 
        	customer_id VARCHAR(36) NOT NULL, 
        	ship_to_id VARCHAR(36), 
        	order_date DATE NOT NULL, 
        	requested_date DATE, 
        	promised_date DATE, 
        	shipped_date DATE, 
        	customer_po_number VARCHAR(50), 
        	quote_id VARCHAR(36), 
        	project_id VARCHAR(36), 
        	status VARCHAR(20), 
        	hold_status VARCHAR(20), 
        	hold_reason VARCHAR(200), 
        	requires_approval BOOLEAN, 
        	approved BOOLEAN, 
        	approved_by VARCHAR(36), 
        	approved_date TIMESTAMP WITHOUT TIME ZONE, 
        	approval_notes TEXT, 
        	subtotal_amount DECIMAL(15, 2), 
        	discount_amount DECIMAL(15, 2), 
        	tax_amount DECIMAL(15, 2), 
        	shipping_amount DECIMAL(15, 2), 
        	handling_amount DECIMAL(15, 2), 
        	total_amount DECIMAL(15, 2), 
        	price_level_id VARCHAR(36), 
        	currency_code VARCHAR(3), 
        	exchange_rate DECIMAL(10, 6), 
        	payment_method VARCHAR(50), 
        	payment_terms_code VARCHAR(20), 
        	credit_card_last_four VARCHAR(4), 
        	shipping_method VARCHAR(50), 
        	carrier VARCHAR(50), 
        	service_level VARCHAR(50), 
        	tracking_number VARCHAR(100), 
        	freight_terms VARCHAR(20), 
        	tax_exempt BOOLEAN, 
        	tax_exempt_number VARCHAR(50), 
        	sales_rep_id VARCHAR(36), 
        	territory_id VARCHAR(36), 
        	source_code VARCHAR(20), 
        	commission_rate DECIMAL(5, 2), 
        	commission_amount DECIMAL(15, 2), 
        	commission_paid BOOLEAN, 
        	warehouse_id VARCHAR(36), 
        	pick_list_printed BOOLEAN, 
        	pick_list_date TIMESTAMP WITHOUT TIME ZONE, 
        	packed BOOLEAN, 
        	packed_date TIMESTAMP WITHOUT TIME ZONE, 
        	documents_generated INTEGER, 
        	order_confirmation_sent BOOLEAN, 
        	exported_to_wms BOOLEAN, 
        	exported_to_ar BOOLEAN, 
        	wms_order_id VARCHAR(36), 
        	ar_invoice_id VARCHAR(36), 
        	picking_instructions TEXT, 
        	packing_instructions TEXT, 
        	shipping_instructions TEXT, 
        	notes TEXT, 
        	internal_notes TEXT, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(100), 
        	updated_by VARCHAR(100), 
        	id VARCHAR(36) NOT NULL, 
        	PRIMARY KEY (id), 
        	CONSTRAINT uq_soe_order_number_tenant UNIQUE (tenant_id, order_number), 
        	FOREIGN KEY(customer_id) REFERENCES so_oe_customer (customer_id), 
        	FOREIGN KEY(ship_to_id) REFERENCES so_oe_ship_to_address (ship_to_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_sales_rep_id ON so_oe_sales_order (sales_rep_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_requested_date ON so_oe_sales_order (requested_date)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_tenant_id ON so_oe_sales_order (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_ship_to_id ON so_oe_sales_order (ship_to_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_customer_id ON so_oe_sales_order (customer_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_promised_date ON so_oe_sales_order (promised_date)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_customer_po_number ON so_oe_sales_order (customer_po_number)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_quote_id ON so_oe_sales_order (quote_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_status ON so_oe_sales_order (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_soe_order_status_date ON so_oe_sales_order (status, order_date)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_order_number ON so_oe_sales_order (order_number)
    """))
    op.execute(sa.text("""
        CREATE UNIQUE INDEX ix_so_oe_sales_order_order_id ON so_oe_sales_order (order_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_warehouse_id ON so_oe_sales_order (warehouse_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_soe_order_customer_date ON so_oe_sales_order (customer_id, order_date)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_sales_order_order_date ON so_oe_sales_order (order_date)
    """))
    op.execute(sa.text("""
        CREATE TABLE so_oe_order_charge (
        	charge_id VARCHAR(36) NOT NULL, 
        	order_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	charge_type VARCHAR(20) NOT NULL, 
        	charge_code VARCHAR(20), 
        	description VARCHAR(200) NOT NULL, 
        	charge_amount DECIMAL(15, 2), 
        	calculation_method VARCHAR(20), 
        	calculation_base DECIMAL(15, 2), 
        	is_taxable BOOLEAN, 
        	tax_code VARCHAR(20), 
        	tax_rate DECIMAL(5, 2), 
        	tax_amount DECIMAL(15, 2), 
        	gl_account_id VARCHAR(36), 
        	is_automatic BOOLEAN, 
        	can_override BOOLEAN, 
        	notes TEXT, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(100), 
        	updated_by VARCHAR(100), 
        	id VARCHAR(36) NOT NULL, 
        	PRIMARY KEY (id), 
        	FOREIGN KEY(order_id) REFERENCES so_oe_sales_order (order_id)
        )
    """))
    op.execute(sa.text("""
        CREATE UNIQUE INDEX ix_so_oe_order_charge_charge_id ON so_oe_order_charge (charge_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_charge_tenant_id ON so_oe_order_charge (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_charge_order_id ON so_oe_order_charge (order_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE so_oe_order_line (
        	line_id VARCHAR(36) NOT NULL, 
        	order_id VARCHAR(36) NOT NULL, 
        	tenant_id VARCHAR(36) NOT NULL, 
        	line_number INTEGER NOT NULL, 
        	line_type VARCHAR(20), 
        	description TEXT, 
        	item_id VARCHAR(36), 
        	item_code VARCHAR(50), 
        	item_description VARCHAR(200), 
        	item_type VARCHAR(20), 
        	quantity_ordered DECIMAL(12, 4), 
        	quantity_allocated DECIMAL(12, 4), 
        	quantity_shipped DECIMAL(12, 4), 
        	quantity_backordered DECIMAL(12, 4), 
        	unit_of_measure VARCHAR(10), 
        	unit_conversion_factor DECIMAL(10, 4), 
        	unit_price DECIMAL(15, 4), 
        	list_price DECIMAL(15, 4), 
        	cost_price DECIMAL(15, 4), 
        	extended_amount DECIMAL(15, 2), 
        	discount_percentage DECIMAL(5, 2), 
        	discount_amount DECIMAL(15, 2), 
        	discount_reason_code VARCHAR(20), 
        	tax_code VARCHAR(20), 
        	tax_rate DECIMAL(5, 2), 
        	tax_amount DECIMAL(15, 2), 
        	is_taxable BOOLEAN, 
        	warehouse_id VARCHAR(36), 
        	location_id VARCHAR(36), 
        	lot_number VARCHAR(50), 
        	serial_number VARCHAR(50), 
        	requested_date DATE, 
        	promised_date DATE, 
        	shipped_date DATE, 
        	line_status VARCHAR(20), 
        	commission_rate DECIMAL(5, 2), 
        	commission_amount DECIMAL(15, 2), 
        	commissionable BOOLEAN, 
        	special_instructions TEXT, 
        	requires_special_handling BOOLEAN, 
        	hazardous_material BOOLEAN, 
        	vendor_id VARCHAR(36), 
        	vendor_item_code VARCHAR(50), 
        	drop_ship BOOLEAN, 
        	parent_line_id VARCHAR(36), 
        	kit_sequence INTEGER, 
        	inventory_allocated BOOLEAN, 
        	allocation_id VARCHAR(36), 
        	cost_center VARCHAR(20), 
        	department VARCHAR(20), 
        	project VARCHAR(20), 
        	notes TEXT, 
        	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
        	created_by VARCHAR(100), 
        	updated_by VARCHAR(100), 
        	id VARCHAR(36) NOT NULL, 
        	PRIMARY KEY (id), 
        	FOREIGN KEY(order_id) REFERENCES so_oe_sales_order (order_id), 
        	FOREIGN KEY(parent_line_id) REFERENCES so_oe_order_line (line_id)
        )
    """))
    op.execute(sa.text("""
        CREATE UNIQUE INDEX ix_so_oe_order_line_line_id ON so_oe_order_line (line_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_line_tenant_id ON so_oe_order_line (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_line_order_id ON so_oe_order_line (order_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_line_item_id ON so_oe_order_line (item_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_line_item_code ON so_oe_order_line (item_code)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_so_oe_order_line_warehouse_id ON so_oe_order_line (warehouse_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE ds_document_access (
        	access_id VARCHAR(50) NOT NULL, 
        	document_id VARCHAR(50) NOT NULL, 
        	accessed_by VARCHAR(100) NOT NULL, 
        	access_type VARCHAR(20) NOT NULL, 
        	ip_address VARCHAR(45), 
        	user_agent VARCHAR(500), 
        	referer VARCHAR(1000), 
        	accessed_at TIMESTAMP WITH TIME ZONE, 
        	tenant_id VARCHAR(100), 
        	PRIMARY KEY (access_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_access_time ON ds_document_access (accessed_at)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_access_user ON ds_document_access (accessed_by)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_access_tenant ON ds_document_access (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_access_document ON ds_document_access (document_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE ds_document_templates (
        	template_id VARCHAR(50) NOT NULL, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	name VARCHAR(255) NOT NULL, 
        	description TEXT, 
        	category VARCHAR(50) NOT NULL, 
        	document_type VARCHAR(50) NOT NULL, 
        	template_content TEXT NOT NULL, 
        	template_variables JSON, 
        	default_classification VARCHAR(20) NOT NULL, 
        	default_tags JSON, 
        	output_format VARCHAR(20) NOT NULL, 
        	usage_count INTEGER, 
        	last_used_at TIMESTAMP WITH TIME ZONE, 
        	last_used_by VARCHAR(100), 
        	is_active BOOLEAN, 
        	version VARCHAR(20), 
        	created_by VARCHAR(100) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	modified_by VARCHAR(100), 
        	modified_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (template_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_template_type ON ds_document_templates (document_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_template_category ON ds_document_templates (category)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_template_tenant ON ds_document_templates (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_template_active ON ds_document_templates (is_active)
    """))
    op.execute(sa.text("""
        CREATE TABLE ds_documents (
        	document_id VARCHAR(50) NOT NULL, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	title VARCHAR(500) NOT NULL, 
        	description TEXT, 
        	content TEXT, 
        	file_path VARCHAR(1000), 
        	file_size INTEGER, 
        	mime_type VARCHAR(100), 
        	file_hash VARCHAR(64), 
        	document_type VARCHAR(50) NOT NULL, 
        	classification VARCHAR(20) NOT NULL, 
        	tags JSON, 
        	custom_metadata JSON, 
        	extracted_text TEXT, 
        	extracted_entities JSON, 
        	content_summary TEXT, 
        	topics JSON, 
        	sentiment_analysis JSON, 
        	language_detection VARCHAR(10), 
        	confidence_scores JSON, 
        	status VARCHAR(20), 
        	processing_status VARCHAR(20), 
        	processing_started_at TIMESTAMP WITH TIME ZONE, 
        	processing_completed_at TIMESTAMP WITH TIME ZONE, 
        	processing_error TEXT, 
        	version_number INTEGER, 
        	parent_document_id VARCHAR(50), 
        	workflow_id VARCHAR(50), 
        	approval_status VARCHAR(50), 
        	collaborators JSON, 
        	current_editors JSON, 
        	access_permissions JSON, 
        	sharing_settings JSON, 
        	retention_date TIMESTAMP WITH TIME ZONE, 
        	compliance_tags JSON, 
        	view_count INTEGER, 
        	download_count INTEGER, 
        	last_accessed_at TIMESTAMP WITH TIME ZONE, 
        	last_accessed_by VARCHAR(100), 
        	created_by VARCHAR(100) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	modified_by VARCHAR(100), 
        	modified_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (document_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_document_status ON ds_documents (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_document_classification ON ds_documents (classification)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_document_created_by ON ds_documents (created_by)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_document_created_at ON ds_documents (created_at)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_document_processing_status ON ds_documents (processing_status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_document_type ON ds_documents (document_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_document_tenant ON ds_documents (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE ds_metric_summaries (
        	summary_id VARCHAR(50) NOT NULL, 
        	metric_name VARCHAR(255) NOT NULL, 
        	summary_type VARCHAR(20) NOT NULL, 
        	start_time TIMESTAMP WITH TIME ZONE NOT NULL, 
        	end_time TIMESTAMP WITH TIME ZONE NOT NULL, 
        	count INTEGER NOT NULL, 
        	sum_value FLOAT, 
        	min_value FLOAT, 
        	max_value FLOAT, 
        	avg_value FLOAT, 
        	median_value FLOAT, 
        	std_dev FLOAT, 
        	percentile_95 FLOAT, 
        	percentile_99 FLOAT, 
        	unique_sources JSON, 
        	unique_tags JSON, 
        	tenant_id VARCHAR(100), 
        	created_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (summary_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_summary_tenant ON ds_metric_summaries (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_summary_metric_type_time ON ds_metric_summaries (metric_name, summary_type, start_time)
    """))
    op.execute(sa.text("""
        CREATE TABLE ds_metrics (
        	metric_id VARCHAR(50) NOT NULL, 
        	metric_name VARCHAR(255) NOT NULL, 
        	metric_type VARCHAR(20) NOT NULL, 
        	value FLOAT NOT NULL, 
        	string_value VARCHAR(1000), 
        	tags JSON, 
        	timestamp TIMESTAMP WITH TIME ZONE NOT NULL, 
        	source VARCHAR(100) NOT NULL, 
        	tenant_id VARCHAR(100), 
        	partition_date VARCHAR(10) NOT NULL, 
        	PRIMARY KEY (metric_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_metric_timestamp ON ds_metrics (timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_metric_source ON ds_metrics (source)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_metric_name_timestamp ON ds_metrics (metric_name, timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_metric_type ON ds_metrics (metric_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_metric_partition ON ds_metrics (partition_date)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_metric_tenant ON ds_metrics (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE am_apis (
        	api_id VARCHAR(36) NOT NULL, 
        	api_name VARCHAR(200) NOT NULL, 
        	api_title VARCHAR(300) NOT NULL, 
        	api_description TEXT, 
        	version VARCHAR(50) NOT NULL, 
        	version_strategy VARCHAR(20) NOT NULL, 
        	protocol_type VARCHAR(20) NOT NULL, 
        	base_path VARCHAR(500) NOT NULL, 
        	upstream_url VARCHAR(1000) NOT NULL, 
        	status VARCHAR(20) NOT NULL, 
        	is_public BOOLEAN NOT NULL, 
        	documentation_url VARCHAR(1000), 
        	openapi_spec JSONB, 
        	graphql_schema TEXT, 
        	timeout_ms INTEGER NOT NULL, 
        	retry_attempts INTEGER NOT NULL, 
        	load_balancing_algorithm VARCHAR(30) NOT NULL, 
        	auth_type VARCHAR(20) NOT NULL, 
        	auth_config JSONB NOT NULL, 
        	default_rate_limit INTEGER, 
        	default_quota_limit INTEGER, 
        	category VARCHAR(100), 
        	tags JSONB NOT NULL, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	capability_id VARCHAR(100) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	created_by VARCHAR(100) NOT NULL, 
        	updated_by VARCHAR(100), 
        	PRIMARY KEY (api_id), 
        	CONSTRAINT uq_api_name_version_tenant UNIQUE (api_name, version, tenant_id), 
        	CONSTRAINT check_timeout_positive CHECK (timeout_ms > 0), 
        	CONSTRAINT check_retry_non_negative CHECK (retry_attempts >= 0)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_apis_api_name ON am_apis (api_name)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_apis_public ON am_apis (is_public)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_apis_tenant_capability ON am_apis (tenant_id, capability_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_apis_name_version ON am_apis (api_name, version)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_apis_capability_id ON am_apis (capability_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_apis_status ON am_apis (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_apis_tenant_id ON am_apis (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE TABLE am_consumers (
        	consumer_id VARCHAR(36) NOT NULL, 
        	consumer_name VARCHAR(200) NOT NULL, 
        	organization VARCHAR(300), 
        	contact_email VARCHAR(255) NOT NULL, 
        	contact_name VARCHAR(200), 
        	status VARCHAR(20) NOT NULL, 
        	approval_date TIMESTAMP WITH TIME ZONE, 
        	approved_by VARCHAR(100), 
        	allowed_apis JSONB NOT NULL, 
        	ip_whitelist JSONB NOT NULL, 
        	global_rate_limit INTEGER, 
        	global_quota_limit INTEGER, 
        	portal_access BOOLEAN NOT NULL, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	created_by VARCHAR(100) NOT NULL, 
        	PRIMARY KEY (consumer_id), 
        	CONSTRAINT uq_consumer_name_tenant UNIQUE (consumer_name, tenant_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_consumers_tenant ON am_consumers (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_consumers_tenant_id ON am_consumers (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_consumers_status ON am_consumers (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_consumers_email ON am_consumers (contact_email)
    """))
    op.execute(sa.text("""
        CREATE TABLE am_analytics (
        	metric_id VARCHAR(36) NOT NULL, 
        	api_id VARCHAR(36), 
        	endpoint_id VARCHAR(36), 
        	consumer_id VARCHAR(36), 
        	metric_name VARCHAR(100) NOT NULL, 
        	metric_type VARCHAR(20) NOT NULL, 
        	metric_value FLOAT NOT NULL, 
        	metric_unit VARCHAR(20), 
        	dimensions JSONB NOT NULL, 
        	timestamp TIMESTAMP WITH TIME ZONE NOT NULL, 
        	time_bucket TIMESTAMP WITH TIME ZONE NOT NULL, 
        	aggregation_period VARCHAR(10) NOT NULL, 
        	tenant_id VARCHAR(100) NOT NULL, 
        	PRIMARY KEY (metric_id), 
        	FOREIGN KEY(api_id) REFERENCES am_apis (api_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_analytics_endpoint_id ON am_analytics (endpoint_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_analytics_metric_name ON am_analytics (metric_name)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_analytics_tenant_time ON am_analytics (tenant_id, time_bucket)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_analytics_timestamp ON am_analytics (timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_analytics_consumer_id ON am_analytics (consumer_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_analytics_api_id ON am_analytics (api_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_analytics_time_bucket ON am_analytics (time_bucket)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_analytics_metric_time ON am_analytics (metric_name, time_bucket)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_analytics_api_time ON am_analytics (api_id, time_bucket)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_analytics_tenant_id ON am_analytics (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_analytics_consumer_time ON am_analytics (consumer_id, time_bucket)
    """))
    op.execute(sa.text("""
        CREATE TABLE am_api_keys (
        	key_id VARCHAR(36) NOT NULL, 
        	consumer_id VARCHAR(36) NOT NULL, 
        	key_name VARCHAR(200) NOT NULL, 
        	key_hash VARCHAR(255) NOT NULL, 
        	key_prefix VARCHAR(20) NOT NULL, 
        	scopes JSONB NOT NULL, 
        	allowed_apis JSONB NOT NULL, 
        	active BOOLEAN NOT NULL, 
        	expires_at TIMESTAMP WITH TIME ZONE, 
        	last_used_at TIMESTAMP WITH TIME ZONE, 
        	rate_limit_override INTEGER, 
        	quota_limit_override INTEGER, 
        	ip_restrictions JSONB NOT NULL, 
        	referer_restrictions JSONB NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	created_by VARCHAR(100) NOT NULL, 
        	PRIMARY KEY (key_id), 
        	CONSTRAINT uq_consumer_key_name UNIQUE (consumer_id, key_name), 
        	FOREIGN KEY(consumer_id) REFERENCES am_consumers (consumer_id), 
        	UNIQUE (key_hash)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_api_keys_consumer ON am_api_keys (consumer_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_api_keys_consumer_id ON am_api_keys (consumer_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_api_keys_prefix ON am_api_keys (key_prefix)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_api_keys_active ON am_api_keys (active)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_api_keys_expires ON am_api_keys (expires_at)
    """))
    op.execute(sa.text("""
        CREATE TABLE am_deployments (
        	deployment_id VARCHAR(36) NOT NULL, 
        	api_id VARCHAR(36) NOT NULL, 
        	deployment_name VARCHAR(200) NOT NULL, 
        	strategy VARCHAR(20) NOT NULL, 
        	environment VARCHAR(50) NOT NULL, 
        	from_version VARCHAR(50), 
        	to_version VARCHAR(50) NOT NULL, 
        	status VARCHAR(20) NOT NULL, 
        	progress_percentage INTEGER NOT NULL, 
        	config JSONB NOT NULL, 
        	traffic_percentage INTEGER NOT NULL, 
        	rollback_available BOOLEAN NOT NULL, 
        	rollback_reason TEXT, 
        	started_at TIMESTAMP WITH TIME ZONE, 
        	completed_at TIMESTAMP WITH TIME ZONE, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	created_by VARCHAR(100) NOT NULL, 
        	PRIMARY KEY (deployment_id), 
        	CONSTRAINT check_progress_percentage CHECK (progress_percentage >= 0 AND progress_percentage <= 100), 
        	CONSTRAINT check_traffic_percentage CHECK (traffic_percentage >= 0 AND traffic_percentage <= 100), 
        	FOREIGN KEY(api_id) REFERENCES am_apis (api_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_deployments_status ON am_deployments (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_deployments_api_id ON am_deployments (api_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_deployments_started ON am_deployments (started_at)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_deployments_api_env ON am_deployments (api_id, environment)
    """))
    op.execute(sa.text("""
        CREATE TABLE am_endpoints (
        	endpoint_id VARCHAR(36) NOT NULL, 
        	api_id VARCHAR(36) NOT NULL, 
        	path VARCHAR(500) NOT NULL, 
        	method VARCHAR(10) NOT NULL, 
        	operation_id VARCHAR(200), 
        	summary VARCHAR(300), 
        	description TEXT, 
        	request_schema JSONB, 
        	response_schema JSONB, 
        	parameters JSONB NOT NULL, 
        	auth_required BOOLEAN NOT NULL, 
        	scopes_required JSONB NOT NULL, 
        	rate_limit_override INTEGER, 
        	cache_enabled BOOLEAN NOT NULL, 
        	cache_ttl_seconds INTEGER, 
        	deprecated BOOLEAN NOT NULL, 
        	examples JSONB NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (endpoint_id), 
        	CONSTRAINT uq_endpoint_path_method UNIQUE (api_id, path, method), 
        	CONSTRAINT check_valid_http_method CHECK (method IN ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS')), 
        	FOREIGN KEY(api_id) REFERENCES am_apis (api_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_endpoints_api_id ON am_endpoints (api_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_endpoints_api_path ON am_endpoints (api_id, path, method)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_endpoints_deprecated ON am_endpoints (deprecated)
    """))
    op.execute(sa.text("""
        CREATE TABLE am_policies (
        	policy_id VARCHAR(36) NOT NULL, 
        	api_id VARCHAR(36) NOT NULL, 
        	policy_name VARCHAR(200) NOT NULL, 
        	policy_type VARCHAR(30) NOT NULL, 
        	policy_description TEXT, 
        	config JSONB NOT NULL, 
        	execution_order INTEGER NOT NULL, 
        	enabled BOOLEAN NOT NULL, 
        	conditions JSONB NOT NULL, 
        	applies_to_endpoints JSONB NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	created_by VARCHAR(100) NOT NULL, 
        	PRIMARY KEY (policy_id), 
        	CONSTRAINT check_execution_order_non_negative CHECK (execution_order >= 0), 
        	FOREIGN KEY(api_id) REFERENCES am_apis (api_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_policies_policy_type ON am_policies (policy_type)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_policies_enabled ON am_policies (enabled)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_policies_execution_order ON am_policies (execution_order)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_policies_api_id ON am_policies (api_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_policies_api_type ON am_policies (api_id, policy_type)
    """))
    op.execute(sa.text("""
        CREATE TABLE am_subscriptions (
        	subscription_id VARCHAR(36) NOT NULL, 
        	consumer_id VARCHAR(36) NOT NULL, 
        	api_id VARCHAR(36) NOT NULL, 
        	subscription_name VARCHAR(200) NOT NULL, 
        	plan_name VARCHAR(100), 
        	status VARCHAR(20) NOT NULL, 
        	starts_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	expires_at TIMESTAMP WITH TIME ZONE, 
        	rate_limit INTEGER, 
        	quota_limit INTEGER, 
        	burst_limit INTEGER, 
        	billing_model VARCHAR(20), 
        	price_per_request FLOAT, 
        	monthly_fee FLOAT, 
        	configuration JSONB NOT NULL, 
        	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        	updated_at TIMESTAMP WITH TIME ZONE, 
        	PRIMARY KEY (subscription_id), 
        	CONSTRAINT uq_consumer_api_subscription UNIQUE (consumer_id, api_id), 
        	FOREIGN KEY(consumer_id) REFERENCES am_consumers (consumer_id), 
        	FOREIGN KEY(api_id) REFERENCES am_apis (api_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_subscriptions_api_id ON am_subscriptions (api_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_subscriptions_consumer_api ON am_subscriptions (consumer_id, api_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_subscriptions_consumer_id ON am_subscriptions (consumer_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_subscriptions_status ON am_subscriptions (status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_subscriptions_expires ON am_subscriptions (expires_at)
    """))
    op.execute(sa.text("""
        CREATE TABLE am_usage_records (
        	record_id VARCHAR(36) NOT NULL, 
        	request_id VARCHAR(100) NOT NULL, 
        	consumer_id VARCHAR(36) NOT NULL, 
        	api_id VARCHAR(36) NOT NULL, 
        	endpoint_path VARCHAR(500) NOT NULL, 
        	method VARCHAR(10) NOT NULL, 
        	timestamp TIMESTAMP WITH TIME ZONE NOT NULL, 
        	response_status INTEGER NOT NULL, 
        	response_time_ms INTEGER NOT NULL, 
        	request_size_bytes INTEGER, 
        	response_size_bytes INTEGER, 
        	client_ip VARCHAR(45), 
        	user_agent VARCHAR(500), 
        	referer VARCHAR(1000), 
        	country_code VARCHAR(2), 
        	region VARCHAR(100), 
        	billable BOOLEAN NOT NULL, 
        	cost FLOAT, 
        	error_code VARCHAR(50), 
        	error_message VARCHAR(500), 
        	tenant_id VARCHAR(100) NOT NULL, 
        	PRIMARY KEY (record_id), 
        	CONSTRAINT check_valid_http_status CHECK (response_status >= 100 AND response_status < 600), 
        	CONSTRAINT check_response_time_non_negative CHECK (response_time_ms >= 0), 
        	UNIQUE (request_id), 
        	FOREIGN KEY(consumer_id) REFERENCES am_consumers (consumer_id)
        )
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_usage_tenant_time ON am_usage_records (tenant_id, timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_usage_records_api_id ON am_usage_records (api_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_usage_records_consumer_id ON am_usage_records (consumer_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_usage_records_tenant_id ON am_usage_records (tenant_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_usage_consumer_time ON am_usage_records (consumer_id, timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_usage_api_time ON am_usage_records (api_id, timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX ix_am_usage_records_timestamp ON am_usage_records (timestamp)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_usage_status ON am_usage_records (response_status)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_am_usage_billable ON am_usage_records (billable)
    """))


def downgrade() -> None:
    op.drop_table('am_usage_records')
    op.drop_table('am_subscriptions')
    op.drop_table('am_policies')
    op.drop_table('am_endpoints')
    op.drop_table('am_deployments')
    op.drop_table('am_api_keys')
    op.drop_table('am_analytics')
    op.drop_table('am_consumers')
    op.drop_table('am_apis')
    op.drop_table('ds_metrics')
    op.drop_table('ds_metric_summaries')
    op.drop_table('ds_documents')
    op.drop_table('ds_document_templates')
    op.drop_table('ds_document_access')
    op.drop_table('so_oe_order_line')
    op.drop_table('so_oe_order_charge')
    op.drop_table('so_oe_sales_order')
    op.drop_table('so_oe_order_template_line')
    op.drop_table('so_oe_ship_to_address')
    op.drop_table('so_oe_order_template')
    op.drop_table('so_oe_price_level')
    op.drop_table('so_oe_order_sequence')
    op.drop_table('so_oe_customer')
    op.drop_table('sm_policies')
    op.drop_table('sm_health_checks')
    op.drop_table('sm_topology')
    op.drop_table('sm_routes')
    op.drop_table('sm_metrics')
    op.drop_table('sm_load_balancers')
    op.drop_table('sm_endpoints')
    op.drop_table('sm_traces')
    op.drop_table('sm_services')
    op.drop_table('sm_security_policies')
    op.drop_table('sm_rate_limiters')
    op.drop_table('sm_predictive_alerts')
    op.drop_table('sm_nl_policies')
    op.drop_table('sm_intelligent_topology')
    op.drop_table('sm_federated_insights')
    op.drop_table('sm_configurations')
    op.drop_table('sm_collaborative_sessions')
    op.drop_table('sm_certificates')
    op.drop_table('sm_autonomous_decisions')
    op.drop_table('sm_alerts')
    op.drop_table('es_subscriptions')
    op.drop_table('es_stream_assignments')
    op.drop_table('es_event_processing_history')
    op.drop_table('es_audit_logs')
    op.drop_table('es_stream_processors')
    op.drop_table('es_metrics')
    op.drop_table('es_events')
    op.drop_table('es_consumer_groups')
    op.drop_table('es_streams')
    op.drop_table('es_schemas')
    op.drop_table('es_event_schemas')
    op.drop_table('cr_versions')
    op.drop_table('cr_usage_analytics')
    op.drop_table('cr_metadata')
    op.drop_table('cr_health_metrics')
    op.drop_table('cr_dependencies')
    op.drop_table('cr_composition_capabilities')
    op.drop_table('cr_registry')
    op.drop_table('cr_compositions')
    op.drop_table('cr_capabilities')
    op.drop_table("apg_records")
