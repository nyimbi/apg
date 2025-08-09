-- APG Workflow Orchestration Database Schema
-- Comprehensive PostgreSQL schema with multi-tenancy, audit trails, and performance optimization
-- © 2025 Datacraft. All rights reserved.

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schema for workflow orchestration
CREATE SCHEMA IF NOT EXISTS workflow_orchestration;

-- Set search path
SET search_path TO workflow_orchestration, public;

-- Workflow Templates table
CREATE TABLE workflow_templates (
	id VARCHAR(26) PRIMARY KEY, -- uuid7str
	name VARCHAR(200) NOT NULL,
	description TEXT DEFAULT '',
	category VARCHAR(100) DEFAULT 'general',
	tags TEXT[] DEFAULT '{}',
	industry VARCHAR(100),
	complexity_level VARCHAR(20) DEFAULT 'intermediate' CHECK (complexity_level IN ('beginner', 'intermediate', 'advanced', 'expert')),
	estimated_duration_hours NUMERIC(8,2) CHECK (estimated_duration_hours >= 0),
	template_data JSONB NOT NULL DEFAULT '{}',
	variables JSONB DEFAULT '{}',
	prerequisites TEXT[] DEFAULT '{}',
	outcomes TEXT[] DEFAULT '{}',
	created_by VARCHAR(26) NOT NULL,
	tenant_id VARCHAR(26) NOT NULL,
	is_public BOOLEAN DEFAULT FALSE,
	is_certified BOOLEAN DEFAULT FALSE,
	version VARCHAR(50) DEFAULT '1.0.0',
	created_at TIMESTAMPTZ DEFAULT NOW(),
	updated_at TIMESTAMPTZ DEFAULT NOW(),
	usage_count INTEGER DEFAULT 0 CHECK (usage_count >= 0),
	-- Audit fields
	created_by_name VARCHAR(200),
	updated_by VARCHAR(26),
	updated_by_name VARCHAR(200)
);

-- Workflows table (main workflow definitions)
CREATE TABLE workflows (
	id VARCHAR(26) PRIMARY KEY, -- uuid7str
	name VARCHAR(200) NOT NULL,
	description TEXT DEFAULT '',
	version VARCHAR(50) DEFAULT '1.0.0',
	-- APG multi-tenant integration
	tenant_id VARCHAR(26) NOT NULL,
	created_by VARCHAR(26) NOT NULL,
	owner_id VARCHAR(26) NOT NULL,
	team_id VARCHAR(26),
	-- Classification
	category VARCHAR(100) DEFAULT 'general',
	tags TEXT[] DEFAULT '{}',
	priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
	-- State management
	status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'paused', 'suspended', 'completed', 'failed', 'cancelled', 'archived')),
	is_template BOOLEAN DEFAULT FALSE,
	is_published BOOLEAN DEFAULT FALSE,
	is_public BOOLEAN DEFAULT FALSE,
	-- Execution configuration
	max_concurrent_instances INTEGER DEFAULT 10 CHECK (max_concurrent_instances BETWEEN 1 AND 1000),
	default_timeout_hours INTEGER CHECK (default_timeout_hours >= 1),
	auto_retry_failed BOOLEAN DEFAULT FALSE,
	max_retry_attempts INTEGER DEFAULT 3 CHECK (max_retry_attempts BETWEEN 0 AND 10),
	-- Parameters and variables
	input_parameters JSONB DEFAULT '{}',
	output_parameters JSONB DEFAULT '{}',
	variables JSONB DEFAULT '{}',
	-- SLA and performance
	sla_hours INTEGER CHECK (sla_hours >= 1),
	estimated_duration_hours NUMERIC(8,2) CHECK (estimated_duration_hours >= 0),
	-- Notifications and escalations
	notification_settings JSONB DEFAULT '{}',
	escalation_settings JSONB DEFAULT '{}',
	-- APG integrations
	required_capabilities TEXT[] DEFAULT '{}',
	integration_settings JSONB DEFAULT '{}',
	-- Compliance and audit
	compliance_requirements TEXT[] DEFAULT '{}',
	audit_level VARCHAR(20) DEFAULT 'basic' CHECK (audit_level IN ('none', 'basic', 'detailed', 'full')),
	-- Monitoring
	monitoring_enabled BOOLEAN DEFAULT TRUE,
	-- Timestamps
	created_at TIMESTAMPTZ DEFAULT NOW(),
	updated_at TIMESTAMPTZ DEFAULT NOW(),
	published_at TIMESTAMPTZ,
	archived_at TIMESTAMPTZ,
	-- Soft delete
	is_deleted BOOLEAN DEFAULT FALSE,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(26),
	-- Metadata
	metadata JSONB DEFAULT '{}',
	-- Audit fields
	created_by_name VARCHAR(200),
	owner_name VARCHAR(200),
	updated_by VARCHAR(26),
	updated_by_name VARCHAR(200)
);

-- Task Definitions table (embedded in workflows as JSONB, but also normalized for queries)
CREATE TABLE task_definitions (
	id VARCHAR(26) PRIMARY KEY, -- uuid7str
	workflow_id VARCHAR(26) NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
	name VARCHAR(200) NOT NULL,
	description TEXT DEFAULT '',
	task_type VARCHAR(20) NOT NULL CHECK (task_type IN ('automated', 'human', 'approval', 'notification', 'integration', 'conditional', 'parallel', 'subprocess', 'timer', 'script')),
	-- Assignment
	assigned_to VARCHAR(26),
	assigned_role VARCHAR(100),
	assigned_group VARCHAR(100),
	-- Timing
	estimated_duration_minutes INTEGER CHECK (estimated_duration_minutes >= 0),
	due_date_offset_hours INTEGER CHECK (due_date_offset_hours >= 0),
	timeout_minutes INTEGER CHECK (timeout_minutes >= 1),
	-- Priority and SLA
	priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
	sla_hours INTEGER CHECK (sla_hours >= 1),
	is_critical BOOLEAN DEFAULT FALSE,
	-- Dependencies (stored as array)
	dependencies TEXT[] DEFAULT '{}',
	-- Configuration
	conditions JSONB DEFAULT '[]',
	skip_conditions JSONB DEFAULT '[]',
	configuration JSONB DEFAULT '{}',
	input_parameters JSONB DEFAULT '{}',
	output_parameters TEXT[] DEFAULT '{}',
	-- Error handling
	max_retry_attempts INTEGER DEFAULT 3 CHECK (max_retry_attempts BETWEEN 0 AND 10),
	retry_delay_seconds INTEGER DEFAULT 60 CHECK (retry_delay_seconds >= 1),
	continue_on_failure BOOLEAN DEFAULT FALSE,
	-- Escalation and notifications
	escalation_rules JSONB DEFAULT '[]',
	notification_rules JSONB DEFAULT '[]',
	-- Metadata and positioning
	metadata JSONB DEFAULT '{}',
	tags TEXT[] DEFAULT '{}',
	position_x NUMERIC(10,2),
	position_y NUMERIC(10,2),
	-- Audit
	created_at TIMESTAMPTZ DEFAULT NOW(),
	updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Workflow Triggers table
CREATE TABLE workflow_triggers (
	id VARCHAR(26) PRIMARY KEY, -- uuid7str
	workflow_id VARCHAR(26) NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
	name VARCHAR(200) NOT NULL,
	trigger_type VARCHAR(20) NOT NULL CHECK (trigger_type IN ('manual', 'scheduled', 'event', 'api', 'webhook', 'condition', 'file', 'email')),
	is_enabled BOOLEAN DEFAULT TRUE,
	-- Scheduling
	cron_expression VARCHAR(100),
	schedule_timezone VARCHAR(50) DEFAULT 'UTC',
	-- Events
	event_source VARCHAR(100),
	event_types TEXT[] DEFAULT '{}',
	event_filters JSONB DEFAULT '{}',
	-- API/Webhook
	webhook_url TEXT,
	api_endpoints TEXT[] DEFAULT '{}',
	authentication_required BOOLEAN DEFAULT TRUE,
	-- Conditions
	condition_expression TEXT,
	condition_data_source VARCHAR(100),
	-- Files
	file_patterns TEXT[] DEFAULT '{}',
	file_directories TEXT[] DEFAULT '{}',
	-- Configuration
	configuration JSONB DEFAULT '{}',
	-- Rate limiting
	max_executions_per_hour INTEGER CHECK (max_executions_per_hour >= 1),
	concurrent_execution_limit INTEGER CHECK (concurrent_execution_limit >= 1),
	-- Timestamps
	created_at TIMESTAMPTZ DEFAULT NOW(),
	updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Workflow Instances table (active executions)
CREATE TABLE workflow_instances (
	id VARCHAR(26) PRIMARY KEY, -- uuid7str
	workflow_id VARCHAR(26) NOT NULL REFERENCES workflows(id),
	workflow_version VARCHAR(50) NOT NULL,
	-- APG context
	tenant_id VARCHAR(26) NOT NULL,
	started_by VARCHAR(26) NOT NULL,
	current_owner VARCHAR(26),
	-- Execution state
	status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('draft', 'active', 'paused', 'suspended', 'completed', 'failed', 'cancelled', 'archived')),
	current_tasks TEXT[] DEFAULT '{}',
	completed_tasks TEXT[] DEFAULT '{}',
	failed_tasks TEXT[] DEFAULT '{}',
	skipped_tasks TEXT[] DEFAULT '{}',
	-- Data
	input_data JSONB DEFAULT '{}',
	output_data JSONB DEFAULT '{}',
	variables JSONB DEFAULT '{}',
	context JSONB DEFAULT '{}',
	-- Timing
	started_at TIMESTAMPTZ DEFAULT NOW(),
	completed_at TIMESTAMPTZ,
	paused_at TIMESTAMPTZ,
	resumed_at TIMESTAMPTZ,
	duration_seconds NUMERIC(12,3) CHECK (duration_seconds >= 0),
	-- Progress
	progress_percentage NUMERIC(5,2) DEFAULT 0.0 CHECK (progress_percentage BETWEEN 0.0 AND 100.0),
	current_step VARCHAR(200),
	total_steps INTEGER DEFAULT 0 CHECK (total_steps >= 0),
	completed_steps INTEGER DEFAULT 0 CHECK (completed_steps >= 0),
	-- Error handling
	error_message TEXT,
	error_details JSONB,
	retry_count INTEGER DEFAULT 0 CHECK (retry_count >= 0),
	max_retries INTEGER DEFAULT 3 CHECK (max_retries >= 0),
	-- SLA
	sla_deadline TIMESTAMPTZ,
	is_sla_breached BOOLEAN DEFAULT FALSE,
	escalation_level INTEGER DEFAULT 0 CHECK (escalation_level >= 0),
	escalated_to VARCHAR(26),
	-- Trigger info
	trigger_source VARCHAR(100),
	trigger_data JSONB DEFAULT '{}',
	-- Relationships
	parent_instance_id VARCHAR(26) REFERENCES workflow_instances(id),
	child_instance_ids TEXT[] DEFAULT '{}',
	-- Audit trail (summary)
	audit_trail JSONB DEFAULT '[]',
	metadata JSONB DEFAULT '{}',
	-- Audit fields
	started_by_name VARCHAR(200),
	current_owner_name VARCHAR(200)
);

-- Task Executions table (individual task runs)
CREATE TABLE task_executions (
	id VARCHAR(26) PRIMARY KEY, -- uuid7str
	instance_id VARCHAR(26) NOT NULL REFERENCES workflow_instances(id) ON DELETE CASCADE,
	task_id VARCHAR(26) NOT NULL, -- References task_definitions.id
	task_name VARCHAR(200) NOT NULL,
	-- Assignment
	assigned_to VARCHAR(26),
	assigned_role VARCHAR(100),
	assigned_group VARCHAR(100),
	current_assignee VARCHAR(26),
	-- State
	status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'ready', 'assigned', 'in_progress', 'completed', 'failed', 'skipped', 'cancelled', 'escalated', 'expired')),
	priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
	-- Timing
	created_at TIMESTAMPTZ DEFAULT NOW(),
	assigned_at TIMESTAMPTZ,
	started_at TIMESTAMPTZ,
	completed_at TIMESTAMPTZ,
	due_date TIMESTAMPTZ,
	duration_seconds NUMERIC(12,3) CHECK (duration_seconds >= 0),
	-- Data
	input_data JSONB DEFAULT '{}',
	output_data JSONB DEFAULT '{}',
	result JSONB DEFAULT '{}',
	-- Error handling
	error_message TEXT,
	error_details JSONB,
	attempt_number INTEGER DEFAULT 1 CHECK (attempt_number >= 1),
	max_attempts INTEGER DEFAULT 3 CHECK (max_attempts >= 1),
	retry_at TIMESTAMPTZ,
	-- Progress
	progress_percentage NUMERIC(5,2) DEFAULT 0.0 CHECK (progress_percentage BETWEEN 0.0 AND 100.0),
	progress_message TEXT,
	-- Human task specifics
	comments JSONB DEFAULT '[]',
	attachments JSONB DEFAULT '[]',
	approval_decision VARCHAR(20) CHECK (approval_decision IN ('approve', 'reject', 'delegate')),
	approval_reason TEXT,
	-- Escalation
	escalation_level INTEGER DEFAULT 0 CHECK (escalation_level >= 0),
	escalated_at TIMESTAMPTZ,
	escalated_to VARCHAR(26),
	escalation_reason TEXT,
	-- SLA
	sla_deadline TIMESTAMPTZ,
	is_sla_breached BOOLEAN DEFAULT FALSE,
	sla_breach_time TIMESTAMPTZ,
	-- Audit
	created_by VARCHAR(26) NOT NULL,
	updated_by VARCHAR(26),
	audit_events JSONB DEFAULT '[]',
	metadata JSONB DEFAULT '{}',
	-- Audit names
	created_by_name VARCHAR(200),
	updated_by_name VARCHAR(200),
	assigned_to_name VARCHAR(200),
	current_assignee_name VARCHAR(200)
);

-- Workflow Connectors table
CREATE TABLE workflow_connectors (
	id VARCHAR(26) PRIMARY KEY, -- uuid7str
	name VARCHAR(200) NOT NULL,
	description TEXT DEFAULT '',
	connector_type VARCHAR(100) NOT NULL,
	-- APG integration
	tenant_id VARCHAR(26) NOT NULL,
	created_by VARCHAR(26) NOT NULL,
	-- Configuration
	connection_config JSONB NOT NULL,
	authentication_config JSONB DEFAULT '{}',
	-- State
	is_enabled BOOLEAN DEFAULT TRUE,
	is_validated BOOLEAN DEFAULT FALSE,
	last_test_at TIMESTAMPTZ,
	last_test_result TEXT,
	-- Rate limiting
	rate_limit_per_minute INTEGER CHECK (rate_limit_per_minute >= 1),
	daily_quota INTEGER CHECK (daily_quota >= 1),
	current_usage INTEGER DEFAULT 0 CHECK (current_usage >= 0),
	-- Error handling
	retry_configuration JSONB DEFAULT '{}',
	timeout_seconds INTEGER DEFAULT 30 CHECK (timeout_seconds BETWEEN 1 AND 300),
	-- Health monitoring
	health_check_enabled BOOLEAN DEFAULT TRUE,
	health_check_interval_minutes INTEGER DEFAULT 5 CHECK (health_check_interval_minutes BETWEEN 1 AND 60),
	last_health_check TIMESTAMPTZ,
	health_status VARCHAR(20) DEFAULT 'unknown' CHECK (health_status IN ('healthy', 'degraded', 'unhealthy', 'unknown')),
	-- Security
	encryption_enabled BOOLEAN DEFAULT TRUE,
	audit_level VARCHAR(20) DEFAULT 'basic' CHECK (audit_level IN ('none', 'basic', 'detailed')),
	-- Timestamps
	created_at TIMESTAMPTZ DEFAULT NOW(),
	updated_at TIMESTAMPTZ DEFAULT NOW(),
	metadata JSONB DEFAULT '{}',
	-- Audit fields
	created_by_name VARCHAR(200),
	updated_by VARCHAR(26),
	updated_by_name VARCHAR(200)
);

-- Workflow Audit Logs table (comprehensive audit trail)
CREATE TABLE workflow_audit_logs (
	id VARCHAR(26) PRIMARY KEY, -- uuid7str
	tenant_id VARCHAR(26) NOT NULL,
	-- Context
	workflow_id VARCHAR(26),
	instance_id VARCHAR(26),
	task_execution_id VARCHAR(26),
	-- Event details
	event_type VARCHAR(100) NOT NULL,
	event_category VARCHAR(20) NOT NULL CHECK (event_category IN ('workflow', 'instance', 'task', 'user', 'system', 'security', 'performance')),
	action VARCHAR(100) NOT NULL,
	resource_type VARCHAR(100) NOT NULL,
	resource_id VARCHAR(26) NOT NULL,
	-- User context
	user_id VARCHAR(26),
	session_id VARCHAR(100),
	ip_address INET,
	user_agent TEXT,
	-- Event data
	event_data JSONB DEFAULT '{}',
	previous_values JSONB,
	new_values JSONB,
	-- Result
	result VARCHAR(20) NOT NULL CHECK (result IN ('success', 'failure', 'partial')),
	error_message TEXT,
	impact_level VARCHAR(20) DEFAULT 'low' CHECK (impact_level IN ('low', 'medium', 'high', 'critical')),
	-- Compliance
	compliance_tags TEXT[] DEFAULT '{}',
	security_classification VARCHAR(20) DEFAULT 'internal' CHECK (security_classification IN ('public', 'internal', 'confidential', 'restricted')),
	retention_policy VARCHAR(50) DEFAULT 'standard',
	-- Timing
	timestamp TIMESTAMPTZ DEFAULT NOW(),
	duration_ms INTEGER CHECK (duration_ms >= 0),
	-- Tracing
	correlation_id VARCHAR(100),
	trace_id VARCHAR(100),
	parent_span_id VARCHAR(100),
	metadata JSONB DEFAULT '{}',
	-- Audit names for performance
	user_name VARCHAR(200),
	resource_name VARCHAR(200)
);

-- Performance indexes
-- Workflows
CREATE INDEX idx_workflows_tenant_id ON workflows(tenant_id);
CREATE INDEX idx_workflows_owner_id ON workflows(owner_id);
CREATE INDEX idx_workflows_status ON workflows(status);
CREATE INDEX idx_workflows_category ON workflows(category);
CREATE INDEX idx_workflows_created_at ON workflows(created_at DESC);
CREATE INDEX idx_workflows_updated_at ON workflows(updated_at DESC);
CREATE INDEX idx_workflows_tags ON workflows USING GIN(tags);
CREATE INDEX idx_workflows_is_deleted ON workflows(is_deleted) WHERE is_deleted = TRUE;
CREATE INDEX idx_workflows_tenant_status ON workflows(tenant_id, status);
CREATE INDEX idx_workflows_search ON workflows USING GIN(to_tsvector('english', name || ' ' || description));

-- Task Definitions
CREATE INDEX idx_task_definitions_workflow_id ON task_definitions(workflow_id);
CREATE INDEX idx_task_definitions_task_type ON task_definitions(task_type);
CREATE INDEX idx_task_definitions_assigned_to ON task_definitions(assigned_to);
CREATE INDEX idx_task_definitions_dependencies ON task_definitions USING GIN(dependencies);

-- Workflow Triggers
CREATE INDEX idx_workflow_triggers_workflow_id ON workflow_triggers(workflow_id);
CREATE INDEX idx_workflow_triggers_type ON workflow_triggers(trigger_type);
CREATE INDEX idx_workflow_triggers_enabled ON workflow_triggers(is_enabled) WHERE is_enabled = TRUE;

-- Workflow Instances
CREATE INDEX idx_workflow_instances_workflow_id ON workflow_instances(workflow_id);
CREATE INDEX idx_workflow_instances_tenant_id ON workflow_instances(tenant_id);
CREATE INDEX idx_workflow_instances_status ON workflow_instances(status);
CREATE INDEX idx_workflow_instances_started_by ON workflow_instances(started_by);
CREATE INDEX idx_workflow_instances_started_at ON workflow_instances(started_at DESC);
CREATE INDEX idx_workflow_instances_completed_at ON workflow_instances(completed_at DESC) WHERE completed_at IS NOT NULL;
CREATE INDEX idx_workflow_instances_sla_deadline ON workflow_instances(sla_deadline) WHERE sla_deadline IS NOT NULL;
CREATE INDEX idx_workflow_instances_tenant_status ON workflow_instances(tenant_id, status);
CREATE INDEX idx_workflow_instances_parent ON workflow_instances(parent_instance_id) WHERE parent_instance_id IS NOT NULL;

-- Task Executions
CREATE INDEX idx_task_executions_instance_id ON task_executions(instance_id);
CREATE INDEX idx_task_executions_task_id ON task_executions(task_id);
CREATE INDEX idx_task_executions_status ON task_executions(status);
CREATE INDEX idx_task_executions_assigned_to ON task_executions(assigned_to);
CREATE INDEX idx_task_executions_due_date ON task_executions(due_date) WHERE due_date IS NOT NULL;
CREATE INDEX idx_task_executions_sla_deadline ON task_executions(sla_deadline) WHERE sla_deadline IS NOT NULL;
CREATE INDEX idx_task_executions_created_at ON task_executions(created_at DESC);
CREATE INDEX idx_task_executions_escalated ON task_executions(escalation_level) WHERE escalation_level > 0;

-- Workflow Connectors
CREATE INDEX idx_workflow_connectors_tenant_id ON workflow_connectors(tenant_id);
CREATE INDEX idx_workflow_connectors_type ON workflow_connectors(connector_type);
CREATE INDEX idx_workflow_connectors_enabled ON workflow_connectors(is_enabled) WHERE is_enabled = TRUE;
CREATE INDEX idx_workflow_connectors_health ON workflow_connectors(health_status);

-- Audit Logs
CREATE INDEX idx_workflow_audit_logs_tenant_id ON workflow_audit_logs(tenant_id);
CREATE INDEX idx_workflow_audit_logs_timestamp ON workflow_audit_logs(timestamp DESC);
CREATE INDEX idx_workflow_audit_logs_workflow_id ON workflow_audit_logs(workflow_id) WHERE workflow_id IS NOT NULL;
CREATE INDEX idx_workflow_audit_logs_instance_id ON workflow_audit_logs(instance_id) WHERE instance_id IS NOT NULL;
CREATE INDEX idx_workflow_audit_logs_user_id ON workflow_audit_logs(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_workflow_audit_logs_event_type ON workflow_audit_logs(event_type);
CREATE INDEX idx_workflow_audit_logs_resource ON workflow_audit_logs(resource_type, resource_id);
CREATE INDEX idx_workflow_audit_logs_correlation ON workflow_audit_logs(correlation_id) WHERE correlation_id IS NOT NULL;

-- Composite indexes for common queries
CREATE INDEX idx_workflows_tenant_owner_status ON workflows(tenant_id, owner_id, status);
CREATE INDEX idx_instances_workflow_status_started ON workflow_instances(workflow_id, status, started_at DESC);
CREATE INDEX idx_tasks_instance_status_priority ON task_executions(instance_id, status, priority DESC);
CREATE INDEX idx_audit_tenant_timestamp_category ON workflow_audit_logs(tenant_id, timestamp DESC, event_category);

-- Partial indexes for active records
CREATE INDEX idx_workflows_active ON workflows(tenant_id, updated_at DESC) WHERE status IN ('active', 'draft') AND is_deleted = FALSE;
CREATE INDEX idx_instances_running ON workflow_instances(started_at DESC) WHERE status IN ('active', 'paused');
CREATE INDEX idx_tasks_pending ON task_executions(created_at DESC) WHERE status IN ('pending', 'ready', 'assigned');

-- Update triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
	NEW.updated_at = NOW();
	RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_workflows_updated_at BEFORE UPDATE ON workflows FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_task_definitions_updated_at BEFORE UPDATE ON task_definitions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_workflow_triggers_updated_at BEFORE UPDATE ON workflow_triggers FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_workflow_connectors_updated_at BEFORE UPDATE ON workflow_connectors FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Cleanup function for old audit logs (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
	deleted_count INTEGER;
BEGIN
	DELETE FROM workflow_audit_logs 
	WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL
	AND retention_policy = 'standard';
	
	GET DIAGNOSTICS deleted_count = ROW_COUNT;
	RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create scheduled job to run cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-workflow-audit-logs', '0 2 * * *', 'SELECT workflow_orchestration.cleanup_old_audit_logs();');

-- Views for common queries
CREATE VIEW active_workflows AS
SELECT w.*, 
	COALESCE(i.active_instances, 0) as active_instances,
	COALESCE(i.total_instances, 0) as total_instances
FROM workflows w
LEFT JOIN (
	SELECT workflow_id, 
		COUNT(*) FILTER (WHERE status = 'active') as active_instances,
		COUNT(*) as total_instances
	FROM workflow_instances 
	GROUP BY workflow_id
) i ON w.id = i.workflow_id
WHERE w.status = 'active' AND w.is_deleted = FALSE;

CREATE VIEW pending_tasks AS
SELECT te.id, te.instance_id, te.task_id, te.task_name,
	te.assigned_to, te.assigned_role, te.status, te.priority,
	te.created_at, te.due_date, te.sla_deadline,
	wi.workflow_id, w.name as workflow_name,
	CASE 
		WHEN te.due_date IS NOT NULL AND te.due_date < NOW() THEN TRUE
		WHEN te.sla_deadline IS NOT NULL AND te.sla_deadline < NOW() THEN TRUE
		ELSE FALSE
	END as is_overdue
FROM task_executions te
JOIN workflow_instances wi ON te.instance_id = wi.id
JOIN workflows w ON wi.workflow_id = w.id
WHERE te.status IN ('pending', 'ready', 'assigned', 'in_progress')
ORDER BY te.priority DESC, te.created_at ASC;

CREATE VIEW workflow_performance_metrics AS
SELECT w.id, w.name, w.tenant_id,
	COUNT(wi.id) as total_executions,
	COUNT(wi.id) FILTER (WHERE wi.status = 'completed') as successful_executions,
	COUNT(wi.id) FILTER (WHERE wi.status = 'failed') as failed_executions,
	ROUND(AVG(wi.duration_seconds), 2) as avg_duration_seconds,
	MAX(wi.completed_at) as last_execution_at,
	CASE 
		WHEN COUNT(wi.id) > 0 THEN 
			ROUND(COUNT(wi.id) FILTER (WHERE wi.status = 'completed')::NUMERIC / COUNT(wi.id) * 100, 2)
		ELSE 0
	END as success_rate_percentage
FROM workflows w
LEFT JOIN workflow_instances wi ON w.id = wi.workflow_id
WHERE w.is_deleted = FALSE
GROUP BY w.id, w.name, w.tenant_id;

-- Grant permissions (adjust based on APG's role structure)
-- GRANT USAGE ON SCHEMA workflow_orchestration TO apg_workflow_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA workflow_orchestration TO apg_workflow_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA workflow_orchestration TO apg_workflow_user;

COMMENT ON SCHEMA workflow_orchestration IS 'APG Workflow Orchestration capability database schema';
COMMENT ON TABLE workflows IS 'Main workflow definition storage with multi-tenant support';
COMMENT ON TABLE workflow_instances IS 'Active workflow execution instances with real-time status tracking';
COMMENT ON TABLE task_executions IS 'Individual task execution records with comprehensive audit trail';
COMMENT ON TABLE workflow_audit_logs IS 'Comprehensive audit logging for compliance and security';