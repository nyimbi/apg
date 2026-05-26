-- APG Import/Export (IMEX) Database Schema
-- Production-grade database schema for enterprise import/export operations
-- Supports multi-tenancy, ACID transactions, and high-performance operations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schema for IMEX capability
CREATE SCHEMA IF NOT EXISTS imex;
SET search_path TO imex, public;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    checksum TEXT
);

-- Insert initial migration record
INSERT INTO schema_migrations (version, description, checksum)
VALUES (1, 'Initial IMEX schema', 'sha256:initial_schema_v1')
ON CONFLICT (version) DO NOTHING;

-- Main import/export jobs table
CREATE TABLE IF NOT EXISTS imex_jobs (
    id VARCHAR(36) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    job_type VARCHAR(50) NOT NULL CHECK (job_type IN ('import', 'export', 'migration', 'sync', 'transform')),
    priority VARCHAR(20) NOT NULL DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),

    -- Configuration (stored as JSONB for efficient querying)
    source_config JSONB NOT NULL,
    target_config JSONB NOT NULL,
    schema_mapping JSONB,
    validation_rules JSONB NOT NULL DEFAULT '[]',
    transformation_steps JSONB NOT NULL DEFAULT '[]',
    schedule_config JSONB,

    -- Processing options
    validation_level VARCHAR(20) NOT NULL DEFAULT 'basic' CHECK (validation_level IN ('none', 'basic', 'strict', 'custom')),
    error_handling VARCHAR(30) NOT NULL DEFAULT 'log_and_continue' CHECK (error_handling IN ('fail_fast', 'skip_errors', 'log_and_continue', 'quarantine', 'custom')),
    parallel_processing BOOLEAN NOT NULL DEFAULT TRUE,
    max_workers INTEGER NOT NULL DEFAULT 4 CHECK (max_workers > 0 AND max_workers <= 32),
    memory_limit_mb INTEGER CHECK (memory_limit_mb IS NULL OR memory_limit_mb > 0),
    timeout_minutes INTEGER NOT NULL DEFAULT 60 CHECK (timeout_minutes > 0),

    -- Status and tracking
    status VARCHAR(20) NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'scheduled', 'queued', 'running', 'paused', 'completed', 'failed', 'cancelled')),
    execution_history JSONB NOT NULL DEFAULT '[]',
    last_run_at TIMESTAMP WITH TIME ZONE,
    next_run_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    tags JSONB NOT NULL DEFAULT '[]',
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- APG platform integration
    etlp_pipeline_id VARCHAR(36),
    audit_trail_id VARCHAR(36),
    notification_config JSONB NOT NULL DEFAULT '{}',

    -- Constraints
    CONSTRAINT imex_jobs_tenant_name_unique UNIQUE (tenant_id, name),
    CONSTRAINT imex_jobs_valid_schedule CHECK (
        (schedule_config IS NULL) OR
        (schedule_config->>'enabled' = 'false') OR
        (schedule_config->>'cron_expression' IS NOT NULL)
    )
);

-- Job execution tracking table
CREATE TABLE IF NOT EXISTS imex_executions (
    id VARCHAR(36) PRIMARY KEY,
    job_id VARCHAR(36) NOT NULL REFERENCES imex_jobs(id) ON DELETE CASCADE,
    execution_number INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'paused', 'completed', 'failed', 'cancelled')),

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Error handling
    error_message TEXT,
    error_details JSONB,

    -- Metrics and performance
    metrics JSONB NOT NULL DEFAULT '{}',

    -- Infrastructure
    log_file_path TEXT,
    worker_node VARCHAR(255),
    execution_config JSONB NOT NULL DEFAULT '{}',

    -- Constraints
    CONSTRAINT imex_executions_job_exec_unique UNIQUE (job_id, execution_number),
    CONSTRAINT imex_executions_timing_check CHECK (
        (started_at IS NULL AND completed_at IS NULL) OR
        (started_at IS NOT NULL AND (completed_at IS NULL OR completed_at >= started_at))
    ),
    CONSTRAINT imex_executions_status_timing CHECK (
        (status IN ('queued', 'paused') AND started_at IS NULL) OR
        (status = 'running' AND started_at IS NOT NULL AND completed_at IS NULL) OR
        (status IN ('completed', 'failed', 'cancelled') AND started_at IS NOT NULL AND completed_at IS NOT NULL)
    )
);

-- Data quality reports table
CREATE TABLE IF NOT EXISTS imex_quality_reports (
    id VARCHAR(36) PRIMARY KEY,
    job_id VARCHAR(36) NOT NULL REFERENCES imex_jobs(id) ON DELETE CASCADE,
    execution_id VARCHAR(36) NOT NULL REFERENCES imex_executions(id) ON DELETE CASCADE,

    -- Quality metrics
    total_records INTEGER NOT NULL CHECK (total_records >= 0),
    valid_records INTEGER NOT NULL CHECK (valid_records >= 0),
    invalid_records INTEGER NOT NULL CHECK (invalid_records >= 0),
    completeness_score NUMERIC(5,4) NOT NULL CHECK (completeness_score >= 0 AND completeness_score <= 1),
    consistency_score NUMERIC(5,4) NOT NULL CHECK (consistency_score >= 0 AND consistency_score <= 1),
    accuracy_score NUMERIC(5,4) NOT NULL CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
    overall_quality_score NUMERIC(5,4) NOT NULL CHECK (overall_quality_score >= 0 AND overall_quality_score <= 1),

    -- Issue analysis
    validation_issues JSONB NOT NULL DEFAULT '{}',
    field_quality_scores JSONB NOT NULL DEFAULT '{}',
    anomalies_detected JSONB NOT NULL DEFAULT '[]',
    recommendations JSONB NOT NULL DEFAULT '[]',

    -- Metadata
    generated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    generated_by VARCHAR(255) NOT NULL DEFAULT 'system',

    -- Constraints
    CONSTRAINT imex_quality_total_records_check CHECK (total_records = valid_records + invalid_records)
);

-- Workflow definitions table
CREATE TABLE IF NOT EXISTS imex_workflows (
    id VARCHAR(36) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',

    -- Workflow definition
    steps JSONB NOT NULL DEFAULT '[]',
    schedule_config JSONB,
    parallel_execution BOOLEAN NOT NULL DEFAULT FALSE,
    error_handling VARCHAR(30) NOT NULL DEFAULT 'fail_fast' CHECK (error_handling IN ('fail_fast', 'skip_errors', 'log_and_continue', 'quarantine', 'custom')),

    -- Status and execution
    status VARCHAR(20) NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'scheduled', 'queued', 'running', 'paused', 'completed', 'failed', 'cancelled')),
    last_execution_id VARCHAR(36),
    execution_history JSONB NOT NULL DEFAULT '[]',

    -- Metadata
    tags JSONB NOT NULL DEFAULT '[]',
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT imex_workflows_tenant_name_unique UNIQUE (tenant_id, name),
    CONSTRAINT imex_workflows_steps_not_empty CHECK (jsonb_array_length(steps) > 0)
);

-- Connection templates table
CREATE TABLE IF NOT EXISTS imex_connection_templates (
    id VARCHAR(36) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL,

    -- Template configuration
    source_template JSONB NOT NULL DEFAULT '{}',
    target_template JSONB NOT NULL DEFAULT '{}',
    schema_mapping_template JSONB,
    validation_template JSONB NOT NULL DEFAULT '[]',

    -- Usage statistics
    usage_count INTEGER NOT NULL DEFAULT 0 CHECK (usage_count >= 0),
    last_used_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT imex_templates_tenant_name_unique UNIQUE (tenant_id, name)
);

-- Monitoring alerts table
CREATE TABLE IF NOT EXISTS imex_monitoring_alerts (
    id VARCHAR(36) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Alert conditions
    metric_name VARCHAR(100) NOT NULL,
    threshold_value NUMERIC NOT NULL,
    comparison_operator VARCHAR(10) NOT NULL CHECK (comparison_operator IN ('gt', 'lt', 'eq', 'ne', 'gte', 'lte')),
    evaluation_window_minutes INTEGER NOT NULL DEFAULT 5 CHECK (evaluation_window_minutes > 0),

    -- Alert actions
    notification_channels JSONB NOT NULL DEFAULT '[]',
    webhook_urls JSONB NOT NULL DEFAULT '[]',
    auto_remediation_script TEXT,

    -- Status
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    last_triggered_at TIMESTAMP WITH TIME ZONE,
    trigger_count INTEGER NOT NULL DEFAULT 0 CHECK (trigger_count >= 0),

    -- Metadata
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT imex_alerts_tenant_name_unique UNIQUE (tenant_id, name)
);

-- Performance indexes for optimal query performance
-- Jobs table indexes
CREATE INDEX IF NOT EXISTS idx_imex_jobs_tenant_id ON imex_jobs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_imex_jobs_status ON imex_jobs(status);
CREATE INDEX IF NOT EXISTS idx_imex_jobs_job_type ON imex_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_imex_jobs_created_at ON imex_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_imex_jobs_last_run_at ON imex_jobs(last_run_at);
CREATE INDEX IF NOT EXISTS idx_imex_jobs_next_run_at ON imex_jobs(next_run_at);
CREATE INDEX IF NOT EXISTS idx_imex_jobs_created_by ON imex_jobs(created_by);

-- JSONB indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_imex_jobs_tags_gin ON imex_jobs USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_imex_jobs_source_config_gin ON imex_jobs USING GIN(source_config);
CREATE INDEX IF NOT EXISTS idx_imex_jobs_target_config_gin ON imex_jobs USING GIN(target_config);

-- Executions table indexes
CREATE INDEX IF NOT EXISTS idx_imex_executions_job_id ON imex_executions(job_id);
CREATE INDEX IF NOT EXISTS idx_imex_executions_status ON imex_executions(status);
CREATE INDEX IF NOT EXISTS idx_imex_executions_started_at ON imex_executions(started_at);
CREATE INDEX IF NOT EXISTS idx_imex_executions_completed_at ON imex_executions(completed_at);
CREATE INDEX IF NOT EXISTS idx_imex_executions_worker_node ON imex_executions(worker_node);

-- Quality reports indexes
CREATE INDEX IF NOT EXISTS idx_imex_quality_job_id ON imex_quality_reports(job_id);
CREATE INDEX IF NOT EXISTS idx_imex_quality_execution_id ON imex_quality_reports(execution_id);
CREATE INDEX IF NOT EXISTS idx_imex_quality_generated_at ON imex_quality_reports(generated_at);
CREATE INDEX IF NOT EXISTS idx_imex_quality_overall_score ON imex_quality_reports(overall_quality_score);

-- Workflow indexes
CREATE INDEX IF NOT EXISTS idx_imex_workflows_tenant_id ON imex_workflows(tenant_id);
CREATE INDEX IF NOT EXISTS idx_imex_workflows_status ON imex_workflows(status);
CREATE INDEX IF NOT EXISTS idx_imex_workflows_created_at ON imex_workflows(created_at);

-- Template indexes
CREATE INDEX IF NOT EXISTS idx_imex_templates_tenant_id ON imex_connection_templates(tenant_id);
CREATE INDEX IF NOT EXISTS idx_imex_templates_category ON imex_connection_templates(category);
CREATE INDEX IF NOT EXISTS idx_imex_templates_public ON imex_connection_templates(is_public);
CREATE INDEX IF NOT EXISTS idx_imex_templates_usage_count ON imex_connection_templates(usage_count);

-- Alert indexes
CREATE INDEX IF NOT EXISTS idx_imex_alerts_tenant_id ON imex_monitoring_alerts(tenant_id);
CREATE INDEX IF NOT EXISTS idx_imex_alerts_enabled ON imex_monitoring_alerts(enabled);
CREATE INDEX IF NOT EXISTS idx_imex_alerts_metric_name ON imex_monitoring_alerts(metric_name);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_imex_jobs_tenant_status ON imex_jobs(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_imex_jobs_tenant_type ON imex_jobs(tenant_id, job_type);
CREATE INDEX IF NOT EXISTS idx_imex_executions_job_number ON imex_executions(job_id, execution_number);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
DROP TRIGGER IF EXISTS update_imex_jobs_updated_at ON imex_jobs;
CREATE TRIGGER update_imex_jobs_updated_at
    BEFORE UPDATE ON imex_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_imex_workflows_updated_at ON imex_workflows;
CREATE TRIGGER update_imex_workflows_updated_at
    BEFORE UPDATE ON imex_workflows
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for job execution number auto-increment
CREATE OR REPLACE FUNCTION set_execution_number()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.execution_number IS NULL THEN
        SELECT COALESCE(MAX(execution_number), 0) + 1
        INTO NEW.execution_number
        FROM imex_executions
        WHERE job_id = NEW.job_id;
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for execution number auto-increment
DROP TRIGGER IF EXISTS set_imex_execution_number ON imex_executions;
CREATE TRIGGER set_imex_execution_number
    BEFORE INSERT ON imex_executions
    FOR EACH ROW EXECUTE FUNCTION set_execution_number();

-- Function to validate JSON schema for configurations
CREATE OR REPLACE FUNCTION validate_source_config()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate required fields in source_config
    IF NEW.source_config->>'source_type' IS NULL THEN
        RAISE EXCEPTION 'source_config must contain source_type';
    END IF;

    IF NEW.source_config->>'format' IS NULL THEN
        RAISE EXCEPTION 'source_config must contain format';
    END IF;

    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for source config validation
DROP TRIGGER IF EXISTS validate_imex_source_config ON imex_jobs;
CREATE TRIGGER validate_imex_source_config
    BEFORE INSERT OR UPDATE ON imex_jobs
    FOR EACH ROW EXECUTE FUNCTION validate_source_config();

-- Function to validate target config
CREATE OR REPLACE FUNCTION validate_target_config()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate required fields in target_config
    IF NEW.target_config->>'target_type' IS NULL THEN
        RAISE EXCEPTION 'target_config must contain target_type';
    END IF;

    IF NEW.target_config->>'format' IS NULL THEN
        RAISE EXCEPTION 'target_config must contain format';
    END IF;

    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for target config validation
DROP TRIGGER IF EXISTS validate_imex_target_config ON imex_jobs;
CREATE TRIGGER validate_imex_target_config
    BEFORE INSERT OR UPDATE ON imex_jobs
    FOR EACH ROW EXECUTE FUNCTION validate_target_config();

-- Function to update job execution history
CREATE OR REPLACE FUNCTION update_job_execution_history()
RETURNS TRIGGER AS $$
BEGIN
    -- Add execution ID to job's execution history
    UPDATE imex_jobs
    SET execution_history = execution_history || to_jsonb(NEW.id),
        last_run_at = CASE WHEN NEW.started_at IS NOT NULL THEN NEW.started_at ELSE last_run_at END
    WHERE id = NEW.job_id;

    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to update job execution history
DROP TRIGGER IF EXISTS update_imex_job_history ON imex_executions;
CREATE TRIGGER update_imex_job_history
    AFTER INSERT ON imex_executions
    FOR EACH ROW EXECUTE FUNCTION update_job_execution_history();

-- Views for common queries and reporting
CREATE OR REPLACE VIEW imex_job_summary AS
SELECT
    j.id,
    j.tenant_id,
    j.name,
    j.job_type,
    j.status,
    j.priority,
    j.created_at,
    j.last_run_at,
    jsonb_array_length(j.execution_history) as total_executions,
    CASE
        WHEN e.status = 'running' THEN 'running'
        WHEN e.status = 'completed' THEN 'completed'
        WHEN e.status = 'failed' THEN 'failed'
        ELSE 'idle'
    END as current_execution_status,
    e.started_at as current_execution_started,
    e.metrics->>'records_processed' as last_records_processed
FROM imex_jobs j
LEFT JOIN imex_executions e ON j.id = e.job_id
    AND e.execution_number = (
        SELECT MAX(execution_number)
        FROM imex_executions
        WHERE job_id = j.id
    );

-- View for execution metrics aggregation
CREATE OR REPLACE VIEW imex_execution_metrics AS
SELECT
    job_id,
    COUNT(*) as total_executions,
    COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_executions,
    AVG(EXTRACT(epoch FROM (completed_at - started_at))) FILTER (WHERE completed_at IS NOT NULL) as avg_duration_seconds,
    AVG((metrics->>'records_processed')::numeric) FILTER (WHERE metrics->>'records_processed' IS NOT NULL) as avg_records_processed,
    AVG((metrics->>'throughput_records_per_second')::numeric) FILTER (WHERE metrics->>'throughput_records_per_second' IS NOT NULL) as avg_throughput
FROM imex_executions
GROUP BY job_id;

-- View for data quality trends
CREATE OR REPLACE VIEW imex_quality_trends AS
SELECT
    job_id,
    DATE_TRUNC('day', generated_at) as date,
    AVG(overall_quality_score) as avg_quality_score,
    AVG(completeness_score) as avg_completeness,
    AVG(consistency_score) as avg_consistency,
    AVG(accuracy_score) as avg_accuracy,
    COUNT(*) as report_count
FROM imex_quality_reports
GROUP BY job_id, DATE_TRUNC('day', generated_at)
ORDER BY job_id, date;

-- Performance monitoring view
CREATE OR REPLACE VIEW imex_performance_summary AS
SELECT
    'jobs' as entity_type,
    COUNT(*) as total_count,
    COUNT(*) FILTER (WHERE status = 'running') as active_count,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as created_last_24h
FROM imex_jobs
UNION ALL
SELECT
    'executions' as entity_type,
    COUNT(*) as total_count,
    COUNT(*) FILTER (WHERE status = 'running') as active_count,
    COUNT(*) FILTER (WHERE started_at > NOW() - INTERVAL '24 hours') as created_last_24h
FROM imex_executions;

-- Grant permissions for application user
-- Note: This should be customized based on your security requirements
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'imex_app_user') THEN
        GRANT USAGE ON SCHEMA imex TO imex_app_user;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA imex TO imex_app_user;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA imex TO imex_app_user;
        GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA imex TO imex_app_user;
    END IF;
END
$$;

-- Comments for documentation
COMMENT ON SCHEMA imex IS 'APG Import/Export capability schema';
COMMENT ON TABLE imex_jobs IS 'Main table for import/export job definitions and configuration';
COMMENT ON TABLE imex_executions IS 'Job execution tracking with metrics and status';
COMMENT ON TABLE imex_quality_reports IS 'Data quality assessment reports for each execution';
COMMENT ON TABLE imex_workflows IS 'Multi-step workflow definitions';
COMMENT ON TABLE imex_connection_templates IS 'Reusable connection templates';
COMMENT ON TABLE imex_monitoring_alerts IS 'System monitoring and alerting configuration';

COMMENT ON COLUMN imex_jobs.source_config IS 'JSONB configuration for data source connection and format';
COMMENT ON COLUMN imex_jobs.target_config IS 'JSONB configuration for data target connection and format';
COMMENT ON COLUMN imex_jobs.schema_mapping IS 'JSONB field mapping configuration between source and target';
COMMENT ON COLUMN imex_jobs.execution_history IS 'JSONB array of execution IDs for this job';

COMMENT ON VIEW imex_job_summary IS 'Summary view of jobs with current execution status';
COMMENT ON VIEW imex_execution_metrics IS 'Aggregated execution metrics per job';
COMMENT ON VIEW imex_quality_trends IS 'Data quality trends over time per job';
COMMENT ON VIEW imex_performance_summary IS 'Overall system performance summary';

-- Final migration marker
INSERT INTO schema_migrations (version, description, checksum)
VALUES (2, 'Added indexes and views', 'sha256:indexes_views_v1')
ON CONFLICT (version) DO NOTHING;

-- Log successful schema creation
DO $$
BEGIN
    RAISE NOTICE 'APG IMEX schema created successfully';
    RAISE NOTICE 'Schema version: 1.0.0';
    RAISE NOTICE 'Tables created: %', (
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'imex'
    );
    RAISE NOTICE 'Indexes created: %', (
        SELECT COUNT(*)
        FROM pg_indexes
        WHERE schemaname = 'imex'
    );
END
$$;