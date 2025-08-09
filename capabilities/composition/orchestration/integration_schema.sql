-- APG Workflow Orchestration Integration Services Database Schema
-- © 2025 Datacraft. All rights reserved.

-- Integrations table
CREATE TABLE IF NOT EXISTS wo_integrations (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    endpoint TEXT NOT NULL,
    auth_type VARCHAR(50) NOT NULL DEFAULT 'none',
    auth_config JSONB,
    headers JSONB,
    timeout_seconds INTEGER DEFAULT 30,
    retry_attempts INTEGER DEFAULT 3,
    retry_delay_seconds INTEGER DEFAULT 5,
    tenant_id VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for integrations
CREATE INDEX IF NOT EXISTS idx_wo_integrations_tenant_id ON wo_integrations (tenant_id);
CREATE INDEX IF NOT EXISTS idx_wo_integrations_type ON wo_integrations (type);
CREATE INDEX IF NOT EXISTS idx_wo_integrations_active ON wo_integrations (is_active);
CREATE INDEX IF NOT EXISTS idx_wo_integrations_created_at ON wo_integrations (created_at);

-- Webhooks table
CREATE TABLE IF NOT EXISTS wo_webhooks (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    secret TEXT,
    events JSONB NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    retry_attempts INTEGER DEFAULT 3,
    retry_delay_seconds INTEGER DEFAULT 5,
    timeout_seconds INTEGER DEFAULT 30,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for webhooks
CREATE INDEX IF NOT EXISTS idx_wo_webhooks_tenant_id ON wo_webhooks (tenant_id);
CREATE INDEX IF NOT EXISTS idx_wo_webhooks_active ON wo_webhooks (is_active);
CREATE INDEX IF NOT EXISTS idx_wo_webhooks_created_at ON wo_webhooks (created_at);

-- Webhook deliveries table
CREATE TABLE IF NOT EXISTS wo_webhook_deliveries (
    id VARCHAR(36) PRIMARY KEY,
    webhook_id VARCHAR(36) NOT NULL REFERENCES wo_webhooks(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    last_attempt_at TIMESTAMP,
    response_code INTEGER,
    response_body TEXT,
    error_message TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for webhook deliveries
CREATE INDEX IF NOT EXISTS idx_wo_webhook_deliveries_webhook_id ON wo_webhook_deliveries (webhook_id);
CREATE INDEX IF NOT EXISTS idx_wo_webhook_deliveries_status ON wo_webhook_deliveries (status);
CREATE INDEX IF NOT EXISTS idx_wo_webhook_deliveries_event_type ON wo_webhook_deliveries (event_type);
CREATE INDEX IF NOT EXISTS idx_wo_webhook_deliveries_expires_at ON wo_webhook_deliveries (expires_at);
CREATE INDEX IF NOT EXISTS idx_wo_webhook_deliveries_created_at ON wo_webhook_deliveries (created_at);

-- Integration audit table
CREATE TABLE IF NOT EXISTS wo_integration_audit (
    id VARCHAR(36) PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL,
    integration_id VARCHAR(36),
    method VARCHAR(10) NOT NULL,
    endpoint TEXT,
    tenant_id VARCHAR(100) NOT NULL,
    workflow_instance_id VARCHAR(36),
    task_execution_id VARCHAR(36),
    request_payload JSONB,
    response_status INTEGER,
    response_data JSONB,
    duration_ms INTEGER,
    success BOOLEAN DEFAULT false,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for integration audit
CREATE INDEX IF NOT EXISTS idx_wo_integration_audit_request_id ON wo_integration_audit (request_id);
CREATE INDEX IF NOT EXISTS idx_wo_integration_audit_integration_id ON wo_integration_audit (integration_id);
CREATE INDEX IF NOT EXISTS idx_wo_integration_audit_tenant_id ON wo_integration_audit (tenant_id);
CREATE INDEX IF NOT EXISTS idx_wo_integration_audit_workflow_instance_id ON wo_integration_audit (workflow_instance_id);
CREATE INDEX IF NOT EXISTS idx_wo_integration_audit_task_execution_id ON wo_integration_audit (task_execution_id);
CREATE INDEX IF NOT EXISTS idx_wo_integration_audit_success ON wo_integration_audit (success);
CREATE INDEX IF NOT EXISTS idx_wo_integration_audit_created_at ON wo_integration_audit (created_at);

-- APG capability registry table
CREATE TABLE IF NOT EXISTS wo_apg_capabilities (
    id VARCHAR(36) PRIMARY KEY,
    capability_name VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(255) NOT NULL,
    description TEXT,
    endpoint TEXT NOT NULL,
    method VARCHAR(10) DEFAULT 'POST',
    input_schema JSONB,
    output_schema JSONB,
    input_mapping JSONB,
    output_mapping JSONB,
    auth_required BOOLEAN DEFAULT true,
    tenant_isolation BOOLEAN DEFAULT true,
    audit_enabled BOOLEAN DEFAULT true,
    rate_limit_per_minute INTEGER DEFAULT 100,
    timeout_seconds INTEGER DEFAULT 30,
    is_active BOOLEAN DEFAULT true,
    version VARCHAR(50) DEFAULT '1.0',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for APG capabilities
CREATE INDEX IF NOT EXISTS idx_wo_apg_capabilities_name ON wo_apg_capabilities (capability_name);
CREATE INDEX IF NOT EXISTS idx_wo_apg_capabilities_active ON wo_apg_capabilities (is_active);
CREATE INDEX IF NOT EXISTS idx_wo_apg_capabilities_created_at ON wo_apg_capabilities (created_at);

-- External system connections table
CREATE TABLE IF NOT EXISTS wo_external_connections (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    connection_type VARCHAR(50) NOT NULL,
    host VARCHAR(255),
    port INTEGER,
    database_name VARCHAR(255),
    username VARCHAR(255),
    password_encrypted TEXT,
    connection_string TEXT,
    ssl_enabled BOOLEAN DEFAULT false,
    ssl_cert_path TEXT,
    connection_timeout INTEGER DEFAULT 30,
    pool_size INTEGER DEFAULT 10,
    tenant_id VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_tested_at TIMESTAMP,
    test_result JSONB
);

-- Create indexes for external connections
CREATE INDEX IF NOT EXISTS idx_wo_external_connections_tenant_id ON wo_external_connections (tenant_id);
CREATE INDEX IF NOT EXISTS idx_wo_external_connections_type ON wo_external_connections (connection_type);
CREATE INDEX IF NOT EXISTS idx_wo_external_connections_active ON wo_external_connections (is_active);
CREATE INDEX IF NOT EXISTS idx_wo_external_connections_created_at ON wo_external_connections (created_at);

-- Integration metrics table
CREATE TABLE IF NOT EXISTS wo_integration_metrics (
    id VARCHAR(36) PRIMARY KEY,
    integration_id VARCHAR(36) NOT NULL,
    date DATE NOT NULL,
    hour INTEGER NOT NULL CHECK (hour >= 0 AND hour <= 23),
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    avg_response_time_ms INTEGER DEFAULT 0,
    min_response_time_ms INTEGER DEFAULT 0,
    max_response_time_ms INTEGER DEFAULT 0,
    total_bytes_sent BIGINT DEFAULT 0,
    total_bytes_received BIGINT DEFAULT 0,
    error_rate DECIMAL(5,2) DEFAULT 0.00,
    uptime_percentage DECIMAL(5,2) DEFAULT 100.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for integration metrics
CREATE UNIQUE INDEX IF NOT EXISTS idx_wo_integration_metrics_unique ON wo_integration_metrics (integration_id, date, hour);
CREATE INDEX IF NOT EXISTS idx_wo_integration_metrics_integration_id ON wo_integration_metrics (integration_id);
CREATE INDEX IF NOT EXISTS idx_wo_integration_metrics_date ON wo_integration_metrics (date);
CREATE INDEX IF NOT EXISTS idx_wo_integration_metrics_created_at ON wo_integration_metrics (created_at);

-- Rate limiting table
CREATE TABLE IF NOT EXISTS wo_rate_limits (
    id VARCHAR(36) PRIMARY KEY,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(36) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    window_start TIMESTAMP NOT NULL,
    window_duration_seconds INTEGER NOT NULL,
    request_count INTEGER DEFAULT 0,
    limit_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for rate limiting
CREATE UNIQUE INDEX IF NOT EXISTS idx_wo_rate_limits_unique ON wo_rate_limits (resource_type, resource_id, tenant_id, window_start);
CREATE INDEX IF NOT EXISTS idx_wo_rate_limits_tenant_id ON wo_rate_limits (tenant_id);
CREATE INDEX IF NOT EXISTS idx_wo_rate_limits_window_start ON wo_rate_limits (window_start);
CREATE INDEX IF NOT EXISTS idx_wo_rate_limits_created_at ON wo_rate_limits (created_at);

-- Security tokens table for integration authentication
CREATE TABLE IF NOT EXISTS wo_security_tokens (
    id VARCHAR(36) PRIMARY KEY,
    token_type VARCHAR(50) NOT NULL,
    token_name VARCHAR(255) NOT NULL,
    token_value_encrypted TEXT NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    integration_id VARCHAR(36),
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    permissions JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP
);

-- Create indexes for security tokens
CREATE INDEX IF NOT EXISTS idx_wo_security_tokens_tenant_id ON wo_security_tokens (tenant_id);
CREATE INDEX IF NOT EXISTS idx_wo_security_tokens_integration_id ON wo_security_tokens (integration_id);
CREATE INDEX IF NOT EXISTS idx_wo_security_tokens_type ON wo_security_tokens (token_type);
CREATE INDEX IF NOT EXISTS idx_wo_security_tokens_active ON wo_security_tokens (is_active);
CREATE INDEX IF NOT EXISTS idx_wo_security_tokens_expires_at ON wo_security_tokens (expires_at);
CREATE INDEX IF NOT EXISTS idx_wo_security_tokens_created_at ON wo_security_tokens (created_at);

-- Integration events table for webhook triggers
CREATE TABLE IF NOT EXISTS wo_integration_events (
    id VARCHAR(36) PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    event_source VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    workflow_instance_id VARCHAR(36),
    task_execution_id VARCHAR(36),
    integration_id VARCHAR(36),
    processed BOOLEAN DEFAULT false,
    webhook_triggered BOOLEAN DEFAULT false,
    webhook_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

-- Create indexes for integration events
CREATE INDEX IF NOT EXISTS idx_wo_integration_events_type ON wo_integration_events (event_type);
CREATE INDEX IF NOT EXISTS idx_wo_integration_events_source ON wo_integration_events (event_source);
CREATE INDEX IF NOT EXISTS idx_wo_integration_events_tenant_id ON wo_integration_events (tenant_id);
CREATE INDEX IF NOT EXISTS idx_wo_integration_events_workflow_instance_id ON wo_integration_events (workflow_instance_id);
CREATE INDEX IF NOT EXISTS idx_wo_integration_events_task_execution_id ON wo_integration_events (task_execution_id);
CREATE INDEX IF NOT EXISTS idx_wo_integration_events_processed ON wo_integration_events (processed);
CREATE INDEX IF NOT EXISTS idx_wo_integration_events_webhook_triggered ON wo_integration_events (webhook_triggered);
CREATE INDEX IF NOT EXISTS idx_wo_integration_events_created_at ON wo_integration_events (created_at);

-- Add foreign key constraints where appropriate
ALTER TABLE wo_webhook_deliveries 
ADD CONSTRAINT fk_wo_webhook_deliveries_webhook_id 
FOREIGN KEY (webhook_id) REFERENCES wo_webhooks(id) ON DELETE CASCADE;

ALTER TABLE wo_integration_audit 
ADD CONSTRAINT fk_wo_integration_audit_integration_id 
FOREIGN KEY (integration_id) REFERENCES wo_integrations(id) ON DELETE SET NULL;

ALTER TABLE wo_integration_metrics 
ADD CONSTRAINT fk_wo_integration_metrics_integration_id 
FOREIGN KEY (integration_id) REFERENCES wo_integrations(id) ON DELETE CASCADE;

ALTER TABLE wo_security_tokens 
ADD CONSTRAINT fk_wo_security_tokens_integration_id 
FOREIGN KEY (integration_id) REFERENCES wo_integrations(id) ON DELETE CASCADE;

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_wo_integrations_updated_at BEFORE UPDATE ON wo_integrations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_wo_webhooks_updated_at BEFORE UPDATE ON wo_webhooks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_wo_apg_capabilities_updated_at BEFORE UPDATE ON wo_apg_capabilities FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_wo_external_connections_updated_at BEFORE UPDATE ON wo_external_connections FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_wo_integration_metrics_updated_at BEFORE UPDATE ON wo_integration_metrics FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_wo_rate_limits_updated_at BEFORE UPDATE ON wo_rate_limits FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_wo_security_tokens_updated_at BEFORE UPDATE ON wo_security_tokens FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default APG capability configurations
INSERT INTO wo_apg_capabilities (
    id, capability_name, display_name, description, endpoint, method,
    input_mapping, output_mapping, auth_required, tenant_isolation, audit_enabled
) VALUES
(
    'apg-auth-rbac-001',
    'auth_rbac',
    'Authentication & RBAC',
    'APG Authentication and Role-Based Access Control capability integration',
    '/api/apg/auth/verify',
    'POST',
    '{"user_id": "user_id", "permission": "permission", "resource": "resource"}',
    '{"authorized": "authorized", "roles": "user_roles", "permissions": "user_permissions"}',
    true,
    true,
    true
),
(
    'apg-audit-compliance-001',
    'audit_compliance',
    'Audit & Compliance',
    'APG Audit and Compliance logging capability integration',
    '/api/apg/audit/log',
    'POST',
    '{"event": "audit_event", "resource": "resource_info", "metadata": "event_metadata"}',
    '{"audit_id": "audit_record_id", "status": "logging_status"}',
    true,
    true,
    false
),
(
    'apg-data-processing-001',
    'data_processing',
    'Data Processing',
    'APG Data Processing capability integration',
    '/api/apg/data/process',
    'POST',
    '{"data": "input_data", "config": "processing_config", "format": "output_format"}',
    '{"result": "processed_data", "metadata": "processing_metadata", "stats": "processing_stats"}',
    true,
    true,
    true
),
(
    'apg-real-time-collab-001',
    'real_time_collaboration',
    'Real-time Collaboration',
    'APG Real-time Collaboration capability integration',
    '/api/apg/collaboration/notify',
    'POST',
    '{"users": "target_users", "message": "notification_content", "type": "notification_type"}',
    '{"sent": "notification_sent", "recipients": "delivered_to", "errors": "delivery_errors"}',
    true,
    true,
    true
),
(
    'apg-financial-services-001',
    'financial_services',
    'Financial Services',
    'APG Financial Services capability integration',
    '/api/apg/financial/process',
    'POST',
    '{"transaction": "transaction_data", "validation": "validation_rules"}',
    '{"result": "processing_result", "compliance": "compliance_status", "audit_trail": "audit_info"}',
    true,
    true,
    true
),
(
    'apg-workflow-orchestration-001',
    'workflow_orchestration',
    'Workflow Orchestration',
    'APG Workflow Orchestration self-integration for nested workflows',
    '/api/apg/workflow/execute',
    'POST',
    '{"workflow_id": "target_workflow_id", "context": "execution_context", "priority": "execution_priority"}',
    '{"instance_id": "workflow_instance_id", "status": "execution_status"}',
    true,
    true,
    true
);

-- Create views for common queries
CREATE OR REPLACE VIEW wo_integration_status AS
SELECT 
    i.id,
    i.name,
    i.type,
    i.tenant_id,
    i.is_active,
    COUNT(ia.id) as total_requests,
    COUNT(CASE WHEN ia.success = true THEN 1 END) as successful_requests,
    COUNT(CASE WHEN ia.success = false THEN 1 END) as failed_requests,
    ROUND(AVG(ia.duration_ms), 2) as avg_response_time_ms,
    MAX(ia.created_at) as last_request_at
FROM wo_integrations i
LEFT JOIN wo_integration_audit ia ON i.id = ia.integration_id
    AND ia.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY i.id, i.name, i.type, i.tenant_id, i.is_active;

CREATE OR REPLACE VIEW wo_webhook_status AS
SELECT 
    w.id,
    w.name,
    w.tenant_id,
    w.is_active,
    COUNT(wd.id) as total_deliveries,
    COUNT(CASE WHEN wd.status = 'delivered' THEN 1 END) as successful_deliveries,
    COUNT(CASE WHEN wd.status = 'failed' THEN 1 END) as failed_deliveries,
    COUNT(CASE WHEN wd.status = 'pending' THEN 1 END) as pending_deliveries,
    MAX(wd.created_at) as last_delivery_at
FROM wo_webhooks w
LEFT JOIN wo_webhook_deliveries wd ON w.id = wd.webhook_id
    AND wd.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY w.id, w.name, w.tenant_id, w.is_active;

-- Comments for documentation
COMMENT ON TABLE wo_integrations IS 'Configuration for external system integrations';
COMMENT ON TABLE wo_webhooks IS 'Webhook subscription configurations';
COMMENT ON TABLE wo_webhook_deliveries IS 'Webhook delivery tracking and status';
COMMENT ON TABLE wo_integration_audit IS 'Audit trail for all integration requests and responses';
COMMENT ON TABLE wo_apg_capabilities IS 'Registry of available APG capabilities for integration';
COMMENT ON TABLE wo_external_connections IS 'External database and system connection configurations';
COMMENT ON TABLE wo_integration_metrics IS 'Hourly metrics for integration performance monitoring';
COMMENT ON TABLE wo_rate_limits IS 'Rate limiting tracking for integrations and capabilities';
COMMENT ON TABLE wo_security_tokens IS 'Encrypted security tokens for integration authentication';
COMMENT ON TABLE wo_integration_events IS 'Events that trigger webhook notifications and processing';

COMMENT ON VIEW wo_integration_status IS 'Real-time status and performance metrics for integrations';
COMMENT ON VIEW wo_webhook_status IS 'Real-time status and delivery metrics for webhooks';