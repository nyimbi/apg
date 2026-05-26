-- APG Connection Management Database Schema
-- Generated on: 2025-08-12 07:07:37.244145
-- Capability: Connection Management (Cn prefix)
-- Database: PostgreSQL


CREATE TABLE cn_connections (
	id UUID NOT NULL,
	tenant_id VARCHAR(100) NOT NULL,
	name VARCHAR(255) NOT NULL,
	description TEXT,
	connection_type connectiontype NOT NULL,
	status connectionstatus,
	singer_tap VARCHAR(100),
	singer_target VARCHAR(100),
	tap_config JSONB,
	target_config JSONB,
	credentials_encrypted BOOLEAN,
	credentials_key_id VARCHAR(100),
	sync_mode syncmode,
	sync_frequency VARCHAR(100),
	batch_size INTEGER,
	enabled BOOLEAN,
	last_sync TIMESTAMP WITH TIME ZONE,
	last_success TIMESTAMP WITH TIME ZONE,
	last_error TEXT,
	error_count INTEGER,
	records_processed INTEGER,
	tags JSONB,
	meta_data JSONB,
	created_at TIMESTAMP WITH TIME ZONE,
	updated_at TIMESTAMP WITH TIME ZONE,
	created_by VARCHAR(100),
	updated_by VARCHAR(100),
	PRIMARY KEY (id)
)

;


CREATE TABLE cn_data_flows (
	id UUID NOT NULL,
	tenant_id VARCHAR(100) NOT NULL,
	name VARCHAR(255) NOT NULL,
	description TEXT,
	source_connection_id UUID NOT NULL,
	target_connection_id UUID NOT NULL,
	status flowstatus,
	enabled BOOLEAN,
	schedule_expression VARCHAR(100),
	schedule_timezone VARCHAR(50),
	field_mappings JSONB,
	transformation_config JSONB,
	filter_config JSONB,
	current_state JSONB,
	last_state_update TIMESTAMP WITH TIME ZONE,
	last_execution TIMESTAMP WITH TIME ZONE,
	next_execution TIMESTAMP WITH TIME ZONE,
	execution_count INTEGER,
	success_count INTEGER,
	error_count INTEGER,
	records_processed INTEGER,
	avg_execution_time_seconds FLOAT,
	last_execution_time_seconds FLOAT,
	tags JSONB,
	meta_data JSONB,
	created_at TIMESTAMP WITH TIME ZONE,
	updated_at TIMESTAMP WITH TIME ZONE,
	created_by VARCHAR(100),
	updated_by VARCHAR(100),
	PRIMARY KEY (id),
	FOREIGN KEY(source_connection_id) REFERENCES cn_connections (id),
	FOREIGN KEY(target_connection_id) REFERENCES cn_connections (id)
)

;


CREATE TABLE cn_flow_executions (
	id UUID NOT NULL,
	flow_id UUID NOT NULL,
	started_at TIMESTAMP WITH TIME ZONE NOT NULL,
	completed_at TIMESTAMP WITH TIME ZONE,
	duration_seconds FLOAT,
	status VARCHAR(50) NOT NULL,
	records_processed INTEGER,
	records_failed INTEGER,
	initial_state JSONB,
	final_state JSONB,
	execution_logs TEXT,
	error_message TEXT,
	meta_data JSONB,
	PRIMARY KEY (id),
	FOREIGN KEY(flow_id) REFERENCES cn_data_flows (id)
)

;


CREATE TABLE cn_transformation_rules (
	id UUID NOT NULL,
	tenant_id VARCHAR(100) NOT NULL,
	name VARCHAR(255) NOT NULL,
	description TEXT,
	rule_type VARCHAR(50) NOT NULL,
	source_field VARCHAR(255) NOT NULL,
	target_field VARCHAR(255) NOT NULL,
	transformation_expression TEXT,
	rule_config JSONB,
	conditions JSONB,
	flow_id UUID,
	execution_order INTEGER,
	enabled BOOLEAN,
	created_at TIMESTAMP WITH TIME ZONE,
	updated_at TIMESTAMP WITH TIME ZONE,
	created_by VARCHAR(100),
	PRIMARY KEY (id),
	FOREIGN KEY(flow_id) REFERENCES cn_data_flows (id)
)

;


CREATE TABLE cn_singer_taps (
	id UUID NOT NULL,
	name VARCHAR(100) NOT NULL,
	package_name VARCHAR(100) NOT NULL,
	version VARCHAR(50),
	description TEXT,
	installation_status VARCHAR(50),
	installation_path VARCHAR(500),
	installation_date TIMESTAMP WITH TIME ZONE,
	config_schema JSONB,
	supported_features JSONB,
	documentation_url VARCHAR(500),
	repository_url VARCHAR(500),
	meta_data JSONB,
	created_at TIMESTAMP WITH TIME ZONE,
	updated_at TIMESTAMP WITH TIME ZONE,
	PRIMARY KEY (id)
)

;


CREATE TABLE cn_singer_targets (
	id UUID NOT NULL,
	name VARCHAR(100) NOT NULL,
	package_name VARCHAR(100) NOT NULL,
	version VARCHAR(50),
	description TEXT,
	installation_status VARCHAR(50),
	installation_path VARCHAR(500),
	installation_date TIMESTAMP WITH TIME ZONE,
	config_schema JSONB,
	supported_features JSONB,
	documentation_url VARCHAR(500),
	repository_url VARCHAR(500),
	meta_data JSONB,
	created_at TIMESTAMP WITH TIME ZONE,
	updated_at TIMESTAMP WITH TIME ZONE,
	PRIMARY KEY (id)
)

;


CREATE TABLE cn_lineage_nodes (
	id UUID NOT NULL,
	tenant_id VARCHAR(100) NOT NULL,
	name VARCHAR(255) NOT NULL,
	node_type lineagenodetype NOT NULL,
	connection_id UUID,
	external_id VARCHAR(255),
	schema_name VARCHAR(100),
	table_name VARCHAR(100),
	field_name VARCHAR(100),
	meta_data JSONB,
	properties JSONB,
	sensitive BOOLEAN,
	pii_classification VARCHAR(50),
	created_at TIMESTAMP WITH TIME ZONE,
	updated_at TIMESTAMP WITH TIME ZONE,
	PRIMARY KEY (id),
	FOREIGN KEY(connection_id) REFERENCES cn_connections (id)
)

;


CREATE TABLE cn_lineage_edges (
	id UUID NOT NULL,
	tenant_id VARCHAR(100) NOT NULL,
	source_node_id UUID NOT NULL,
	target_node_id UUID NOT NULL,
	relationship_type VARCHAR(50) NOT NULL,
	transformation_logic TEXT,
	flow_id UUID,
	meta_data JSONB,
	properties JSONB,
	confidence_score FLOAT,
	created_at TIMESTAMP WITH TIME ZONE,
	updated_at TIMESTAMP WITH TIME ZONE,
	PRIMARY KEY (id),
	FOREIGN KEY(source_node_id) REFERENCES cn_lineage_nodes (id),
	FOREIGN KEY(target_node_id) REFERENCES cn_lineage_nodes (id),
	FOREIGN KEY(flow_id) REFERENCES cn_data_flows (id)
)

;


CREATE TABLE cn_health_checks (
	id UUID NOT NULL,
	connection_id UUID NOT NULL,
	check_time TIMESTAMP WITH TIME ZONE NOT NULL,
	status VARCHAR(50) NOT NULL,
	latency_ms FLOAT,
	throughput_records_per_sec FLOAT,
	error_rate FLOAT,
	cpu_usage_percent FLOAT,
	memory_usage_percent FLOAT,
	check_results JSONB,
	error_details TEXT,
	meta_data JSONB,
	PRIMARY KEY (id),
	FOREIGN KEY(connection_id) REFERENCES cn_connections (id)
)

;
