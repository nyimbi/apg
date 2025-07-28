-- APG Financial Reporting - Revolutionary Database Schema
-- Enhanced multi-tenant database schema with AI-powered capabilities and performance optimization
-- © 2025 Datacraft. All rights reserved.
-- Author: Nyimbi Odero | APG Platform Architect

-- ==================================================================================
-- PERFORMANCE-OPTIMIZED TABLE SPACES AND PARTITIONING
-- ==================================================================================

-- Create specialized tablespaces for different data types
CREATE TABLESPACE fr_hot_data LOCATION '/var/lib/postgresql/fr_hot';     -- Frequently accessed data
CREATE TABLESPACE fr_warm_data LOCATION '/var/lib/postgresql/fr_warm';   -- Moderately accessed data  
CREATE TABLESPACE fr_cold_data LOCATION '/var/lib/postgresql/fr_cold';   -- Archival data
CREATE TABLESPACE fr_index_data LOCATION '/var/lib/postgresql/fr_index'; -- Index data

-- ==================================================================================
-- ENHANCED REPORT TEMPLATE TABLE WITH AI CAPABILITIES
-- ==================================================================================

-- Enhanced report template with AI capabilities
CREATE TABLE cf_fr_report_template (
    -- Identity
    id SERIAL PRIMARY KEY,
    template_id VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(36) NOT NULL,
    
    -- Template Information
    template_code VARCHAR(50) NOT NULL,
    template_name VARCHAR(200) NOT NULL,
    description TEXT,
    
    -- Template Classification
    statement_type VARCHAR(50) NOT NULL,
    category VARCHAR(50),
    format_type VARCHAR(50) NOT NULL,
    
    -- Revolutionary AI Enhancement Features
    ai_intelligence_level VARCHAR(20) DEFAULT 'enhanced',
    auto_narrative_generation BOOLEAN DEFAULT TRUE,
    predictive_insights_enabled BOOLEAN DEFAULT TRUE,
    adaptive_formatting BOOLEAN DEFAULT TRUE,
    natural_language_interface BOOLEAN DEFAULT TRUE,
    voice_activation_enabled BOOLEAN DEFAULT FALSE,
    
    -- Template Properties
    is_system BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    version VARCHAR(20) DEFAULT '1.0',
    
    -- Enhanced Formatting Options
    currency_type VARCHAR(20) DEFAULT 'single',
    show_percentages BOOLEAN DEFAULT FALSE,
    show_variances BOOLEAN DEFAULT FALSE,
    decimal_places INTEGER DEFAULT 2,
    thousands_separator BOOLEAN DEFAULT TRUE,
    
    -- Adaptive Layout Options
    page_orientation VARCHAR(20) DEFAULT 'portrait',
    font_size INTEGER DEFAULT 12,
    include_logo BOOLEAN DEFAULT TRUE,
    include_header BOOLEAN DEFAULT TRUE,
    include_footer BOOLEAN DEFAULT TRUE,
    dynamic_layout_optimization BOOLEAN DEFAULT TRUE,
    responsive_design_enabled BOOLEAN DEFAULT TRUE,
    accessibility_enhanced BOOLEAN DEFAULT TRUE,
    
    -- Intelligent Generation Options
    auto_generate BOOLEAN DEFAULT FALSE,
    generation_frequency VARCHAR(20),
    last_generated TIMESTAMP,
    ai_optimization_score DECIMAL(5,2) DEFAULT 0.0,
    usage_pattern_data JSONB,
    
    -- Real-Time Collaboration Features
    real_time_collaboration BOOLEAN DEFAULT TRUE,
    collaborative_session_id VARCHAR(36),
    conflict_resolution_mode VARCHAR(20) DEFAULT 'intelligent',
    version_control_enabled BOOLEAN DEFAULT TRUE,
    
    -- Advanced AI Configuration
    ai_model_preferences JSONB,
    natural_language_prompts JSONB,
    predictive_model_config JSONB,
    personalization_data JSONB,
    
    -- Performance and Analytics
    generation_performance_metrics JSONB,
    user_satisfaction_score DECIMAL(3,1) DEFAULT 0.0,
    usage_analytics JSONB,
    
    -- Metadata and Configuration
    configuration JSONB,
    
    -- Audit Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(36),
    updated_by VARCHAR(36)
) TABLESPACE fr_hot_data;

-- Partition by tenant_id for multi-tenant performance
CREATE TABLE cf_fr_report_template_partitioned (
    LIKE cf_fr_report_template INCLUDING ALL
) PARTITION BY HASH (tenant_id);

-- Create partitions for tenant distribution
CREATE TABLE cf_fr_report_template_p0 PARTITION OF cf_fr_report_template_partitioned FOR VALUES WITH (modulus 8, remainder 0);
CREATE TABLE cf_fr_report_template_p1 PARTITION OF cf_fr_report_template_partitioned FOR VALUES WITH (modulus 8, remainder 1);
CREATE TABLE cf_fr_report_template_p2 PARTITION OF cf_fr_report_template_partitioned FOR VALUES WITH (modulus 8, remainder 2);
CREATE TABLE cf_fr_report_template_p3 PARTITION OF cf_fr_report_template_partitioned FOR VALUES WITH (modulus 8, remainder 3);
CREATE TABLE cf_fr_report_template_p4 PARTITION OF cf_fr_report_template_partitioned FOR VALUES WITH (modulus 8, remainder 4);
CREATE TABLE cf_fr_report_template_p5 PARTITION OF cf_fr_report_template_partitioned FOR VALUES WITH (modulus 8, remainder 5);
CREATE TABLE cf_fr_report_template_p6 PARTITION OF cf_fr_report_template_partitioned FOR VALUES WITH (modulus 8, remainder 6);
CREATE TABLE cf_fr_report_template_p7 PARTITION OF cf_fr_report_template_partitioned FOR VALUES WITH (modulus 8, remainder 7);

-- ==================================================================================
-- REVOLUTIONARY AI-POWERED CONVERSATIONAL INTERFACE
-- ==================================================================================

CREATE TABLE cf_fr_conversational_interface (
    -- Identity
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    
    -- Conversation Context
    session_id VARCHAR(36) NOT NULL,
    conversation_type VARCHAR(50) DEFAULT 'report_creation',
    language_code VARCHAR(10) DEFAULT 'en-US',
    
    -- User Input
    user_query TEXT NOT NULL,
    query_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_modality VARCHAR(20) DEFAULT 'text',
    
    -- AI Processing
    intent_classification VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(4,3) DEFAULT 0.0,
    extracted_entities JSONB,
    context_understanding JSONB,
    
    -- AI Response
    ai_response TEXT,
    response_type VARCHAR(50) DEFAULT 'interactive',
    generated_artifacts JSONB,
    
    -- Learning and Optimization
    user_feedback_score DECIMAL(3,1),
    resolution_success BOOLEAN DEFAULT TRUE,
    learning_data JSONB,
    
    -- Performance Metrics
    processing_time_ms INTEGER,
    model_version VARCHAR(20),
    
    -- Audit Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) TABLESPACE fr_hot_data;

-- Time-series partitioning for conversation data
CREATE TABLE cf_fr_conversational_interface_partitioned (
    LIKE cf_fr_conversational_interface INCLUDING ALL
) PARTITION BY RANGE (query_timestamp);

-- Create monthly partitions for the last 12 months and future
CREATE TABLE cf_fr_conversational_interface_2024_01 PARTITION OF cf_fr_conversational_interface_partitioned 
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE cf_fr_conversational_interface_2024_02 PARTITION OF cf_fr_conversational_interface_partitioned 
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Continue for current year...
CREATE TABLE cf_fr_conversational_interface_2025_01 PARTITION OF cf_fr_conversational_interface_partitioned 
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE cf_fr_conversational_interface_2025_02 PARTITION OF cf_fr_conversational_interface_partitioned 
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- ==================================================================================
-- AI-POWERED INSIGHT GENERATION ENGINE
-- ==================================================================================

CREATE TABLE cf_fr_ai_insight_engine (
    -- Identity
    id SERIAL PRIMARY KEY,
    insight_id VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(36) NOT NULL,
    
    -- Insight Context
    source_report_id VARCHAR(36),
    source_data_type VARCHAR(50) NOT NULL,
    analysis_period VARCHAR(50),
    
    -- Insight Content
    insight_type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    narrative_explanation TEXT,
    
    -- AI Assessment
    confidence_level DECIMAL(4,3) NOT NULL,
    impact_score DECIMAL(4,1) NOT NULL,
    urgency_level VARCHAR(20) DEFAULT 'medium',
    accuracy_validation DECIMAL(4,3),
    
    -- Actionable Intelligence
    recommended_actions JSONB,
    risk_indicators JSONB,
    opportunity_indicators JSONB,
    
    -- Supporting Data
    supporting_metrics JSONB,
    data_sources JSONB,
    related_insights JSONB,
    
    -- ML Model Information
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    training_data_period VARCHAR(50),
    
    -- User Interaction
    user_views INTEGER DEFAULT 0,
    user_actions_taken JSONB,
    feedback_ratings JSONB,
    
    -- Status and Lifecycle
    insight_status VARCHAR(20) DEFAULT 'active',
    expiry_date TIMESTAMP,
    superseded_by VARCHAR(36),
    
    -- Performance tracking
    generation_time_ms INTEGER,
    
    -- Audit Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) TABLESPACE fr_warm_data;

-- ==================================================================================
-- PREDICTIVE ANALYTICS ENGINE
-- ==================================================================================

CREATE TABLE cf_fr_predictive_analytics (
    -- Identity
    id SERIAL PRIMARY KEY,
    prediction_id VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(36) NOT NULL,
    
    -- Prediction Context
    prediction_type VARCHAR(50) NOT NULL,
    target_metric VARCHAR(100) NOT NULL,
    prediction_horizon INTEGER NOT NULL,
    base_period DATE NOT NULL,
    
    -- Prediction Results
    predicted_value DECIMAL(20,4) NOT NULL,
    confidence_interval_lower DECIMAL(20,4),
    confidence_interval_upper DECIMAL(20,4),
    confidence_percentage DECIMAL(5,2) NOT NULL,
    
    -- Model Performance
    model_type VARCHAR(50) NOT NULL,
    model_accuracy_score DECIMAL(4,3),
    feature_importance JSONB,
    training_data_points INTEGER,
    
    -- Contributing Factors
    primary_drivers JSONB,
    risk_factors JSONB,
    seasonal_adjustments JSONB,
    external_factors JSONB,
    
    -- Validation and Tracking
    actual_value DECIMAL(20,4),
    prediction_error DECIMAL(15,4),
    validation_date TIMESTAMP,
    
    -- Alert Configuration
    variance_threshold DECIMAL(10,4),
    alert_triggered BOOLEAN DEFAULT FALSE,
    alert_recipients JSONB,
    
    -- Scenario Analysis
    best_case_scenario DECIMAL(20,4),
    worst_case_scenario DECIMAL(20,4),
    most_likely_scenario DECIMAL(20,4),
    scenario_probabilities JSONB,
    
    -- Model Metadata
    model_training_date TIMESTAMP,
    model_retrain_frequency VARCHAR(20) DEFAULT 'monthly',
    next_retrain_date TIMESTAMP,
    
    -- Audit Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) TABLESPACE fr_warm_data;

-- ==================================================================================
-- REAL-TIME COLLABORATION SYSTEM
-- ==================================================================================

CREATE TABLE cf_fr_real_time_collaboration (
    -- Identity
    id SERIAL PRIMARY KEY,
    collaboration_id VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(36) NOT NULL,
    
    -- Session Information
    session_id VARCHAR(36) NOT NULL,
    report_id VARCHAR(36) NOT NULL,
    session_type VARCHAR(50) DEFAULT 'collaborative_editing',
    
    -- Participants
    session_owner VARCHAR(36) NOT NULL,
    active_participants JSONB NOT NULL,
    total_participants JSONB,
    max_concurrent_users INTEGER DEFAULT 0,
    
    -- Session Timeline
    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_duration_minutes INTEGER,
    
    -- Collaboration Features
    real_time_sync_enabled BOOLEAN DEFAULT TRUE,
    conflict_resolution_mode VARCHAR(30) DEFAULT 'intelligent',
    version_control_enabled BOOLEAN DEFAULT TRUE,
    auto_save_interval INTEGER DEFAULT 30,
    
    -- Activity Tracking
    edit_operations JSONB,
    comment_threads JSONB,
    approval_requests JSONB,
    
    -- Conflict Management
    conflicts_detected INTEGER DEFAULT 0,
    conflicts_resolved INTEGER DEFAULT 0,
    conflict_resolution_log JSONB,
    
    -- AI Enhancement
    ai_suggestions_enabled BOOLEAN DEFAULT TRUE,
    ai_suggestions_provided JSONB,
    ai_suggestions_accepted INTEGER DEFAULT 0,
    
    -- Performance Metrics
    sync_latency_ms INTEGER,
    operation_count INTEGER DEFAULT 0,
    bandwidth_usage_mb DECIMAL(10,2),
    
    -- Quality Metrics
    user_satisfaction_scores JSONB,
    productivity_metrics JSONB,
    error_rate DECIMAL(5,4) DEFAULT 0.0,
    
    -- Audit Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) TABLESPACE fr_hot_data;

-- ==================================================================================
-- DATA QUALITY MONITORING SYSTEM
-- ==================================================================================

CREATE TABLE cf_fr_data_quality_monitor (
    -- Identity
    id SERIAL PRIMARY KEY,
    monitor_id VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(36) NOT NULL,
    
    -- Monitoring Context
    data_source VARCHAR(100) NOT NULL,
    monitoring_scope VARCHAR(50) NOT NULL,
    check_frequency VARCHAR(20) DEFAULT 'real_time',
    
    -- Quality Assessment
    overall_quality_score DECIMAL(5,2) NOT NULL,
    quality_trend VARCHAR(20) DEFAULT 'stable',
    last_assessment TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Quality Dimensions
    completeness_score DECIMAL(5,2) DEFAULT 100.0,
    accuracy_score DECIMAL(5,2) DEFAULT 100.0,
    consistency_score DECIMAL(5,2) DEFAULT 100.0,
    timeliness_score DECIMAL(5,2) DEFAULT 100.0,
    validity_score DECIMAL(5,2) DEFAULT 100.0,
    
    -- Issue Detection
    anomalies_detected INTEGER DEFAULT 0,
    critical_issues INTEGER DEFAULT 0,
    warning_issues INTEGER DEFAULT 0,
    info_issues INTEGER DEFAULT 0,
    
    -- Issue Details
    detected_anomalies JSONB,
    data_profiling_results JSONB,
    validation_rules_failed JSONB,
    
    -- AI-Powered Enhancement
    ml_anomaly_detection BOOLEAN DEFAULT TRUE,
    statistical_outlier_detection BOOLEAN DEFAULT TRUE,
    pattern_recognition_enabled BOOLEAN DEFAULT TRUE,
    ai_correction_suggestions JSONB,
    
    -- Auto-Correction
    auto_correction_enabled BOOLEAN DEFAULT FALSE,
    corrections_applied INTEGER DEFAULT 0,
    correction_success_rate DECIMAL(5,4) DEFAULT 0.0,
    correction_log JSONB,
    
    -- Performance Tracking
    monitoring_duration_ms INTEGER,
    records_processed INTEGER,
    processing_rate_per_second DECIMAL(10,2),
    
    -- Alerting
    alert_thresholds JSONB,
    alerts_triggered INTEGER DEFAULT 0,
    alert_recipients JSONB,
    
    -- Historical Tracking
    quality_history JSONB,
    improvement_suggestions JSONB,
    
    -- Audit Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) TABLESPACE fr_hot_data;

-- ==================================================================================
-- BLOCKCHAIN AUDIT TRAIL SYSTEM
-- ==================================================================================

CREATE TABLE cf_fr_blockchain_audit_trail (
    -- Identity
    id SERIAL PRIMARY KEY,
    audit_id VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(36) NOT NULL,
    
    -- Blockchain Reference
    block_hash VARCHAR(64) NOT NULL UNIQUE,
    transaction_hash VARCHAR(64) NOT NULL,
    block_number INTEGER NOT NULL,
    blockchain_network VARCHAR(50) DEFAULT 'ethereum',
    
    -- Audit Context
    audit_event_type VARCHAR(50) NOT NULL,
    source_entity_type VARCHAR(50) NOT NULL,
    source_entity_id VARCHAR(36) NOT NULL,
    
    -- Event Details
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(36) NOT NULL,
    user_role VARCHAR(50),
    event_description TEXT NOT NULL,
    
    -- Data Integrity
    data_hash VARCHAR(64) NOT NULL,
    previous_hash VARCHAR(64),
    merkle_root VARCHAR(64),
    
    -- Cryptographic Verification
    digital_signature TEXT,
    public_key TEXT,
    certificate_authority VARCHAR(100),
    
    -- Compliance and Regulatory
    regulatory_framework VARCHAR(50),
    compliance_status VARCHAR(20) DEFAULT 'compliant',
    retention_period_years INTEGER DEFAULT 7,
    
    -- Smart Contract Integration
    smart_contract_address VARCHAR(42),
    smart_contract_function VARCHAR(100),
    gas_used INTEGER,
    transaction_fee DECIMAL(18,8),
    
    -- Verification Status
    verification_status VARCHAR(20) DEFAULT 'pending',
    verification_timestamp TIMESTAMP,
    verification_attempts INTEGER DEFAULT 0,
    
    -- Performance Metrics
    blockchain_confirmation_time INTEGER,
    network_congestion_factor DECIMAL(5,2),
    
    -- Forensic Analysis
    forensic_markers JSONB,
    anomaly_indicators JSONB,
    
    -- Audit Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) TABLESPACE fr_cold_data;

-- Time-series partitioning for blockchain audit data
CREATE TABLE cf_fr_blockchain_audit_trail_partitioned (
    LIKE cf_fr_blockchain_audit_trail INCLUDING ALL
) PARTITION BY RANGE (event_timestamp);

-- ==================================================================================
-- PERFORMANCE-OPTIMIZED INDEXES
-- ==================================================================================

-- Report Template Indexes
CREATE INDEX CONCURRENTLY idx_report_template_tenant_id ON cf_fr_report_template (tenant_id) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_report_template_code ON cf_fr_report_template (template_code) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_report_template_type ON cf_fr_report_template (statement_type) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_report_template_ai_level ON cf_fr_report_template (ai_intelligence_level) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_report_template_active ON cf_fr_report_template (is_active) WHERE is_active = TRUE;

-- Conversational Interface Indexes
CREATE INDEX CONCURRENTLY idx_conversation_tenant_user ON cf_fr_conversational_interface (tenant_id, user_id) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_conversation_session ON cf_fr_conversational_interface (session_id) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_conversation_intent ON cf_fr_conversational_interface (intent_classification) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_conversation_timestamp ON cf_fr_conversational_interface (query_timestamp DESC) TABLESPACE fr_index_data;

-- AI Insight Engine Indexes
CREATE INDEX CONCURRENTLY idx_insight_tenant_type ON cf_fr_ai_insight_engine (tenant_id, insight_type) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_insight_confidence ON cf_fr_ai_insight_engine (confidence_level DESC) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_insight_impact ON cf_fr_ai_insight_engine (impact_score DESC) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_insight_status ON cf_fr_ai_insight_engine (insight_status) TABLESPACE fr_index_data;

-- Predictive Analytics Indexes  
CREATE INDEX CONCURRENTLY idx_prediction_tenant_type ON cf_fr_predictive_analytics (tenant_id, prediction_type) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_prediction_horizon ON cf_fr_predictive_analytics (prediction_horizon) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_prediction_accuracy ON cf_fr_predictive_analytics (model_accuracy_score DESC) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_prediction_base_period ON cf_fr_predictive_analytics (base_period DESC) TABLESPACE fr_index_data;

-- Real-Time Collaboration Indexes
CREATE INDEX CONCURRENTLY idx_collaboration_tenant_session ON cf_fr_real_time_collaboration (tenant_id, session_id) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_collaboration_report ON cf_fr_real_time_collaboration (report_id) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_collaboration_active ON cf_fr_real_time_collaboration (last_activity DESC) TABLESPACE fr_index_data;

-- Data Quality Monitor Indexes
CREATE INDEX CONCURRENTLY idx_quality_tenant_source ON cf_fr_data_quality_monitor (tenant_id, data_source) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_quality_score ON cf_fr_data_quality_monitor (overall_quality_score) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_quality_assessment ON cf_fr_data_quality_monitor (last_assessment DESC) TABLESPACE fr_index_data;

-- Blockchain Audit Trail Indexes
CREATE INDEX CONCURRENTLY idx_audit_tenant_event ON cf_fr_blockchain_audit_trail (tenant_id, audit_event_type) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_audit_entity ON cf_fr_blockchain_audit_trail (source_entity_type, source_entity_id) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_audit_timestamp ON cf_fr_blockchain_audit_trail (event_timestamp DESC) TABLESPACE fr_index_data;
CREATE INDEX CONCURRENTLY idx_audit_block ON cf_fr_blockchain_audit_trail (block_number DESC) TABLESPACE fr_index_data;

-- ==================================================================================
-- ADVANCED ANALYTICS VIEWS
-- ==================================================================================

-- Real-time performance analytics view
CREATE VIEW vw_fr_performance_analytics AS
SELECT 
    tenant_id,
    COUNT(*) as total_conversations,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(confidence_score) as avg_confidence,
    COUNT(*) FILTER (WHERE resolution_success = TRUE) * 100.0 / COUNT(*) as success_rate,
    DATE_TRUNC('hour', query_timestamp) as hour_bucket
FROM cf_fr_conversational_interface
WHERE query_timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY tenant_id, hour_bucket;

-- Data quality trend analysis view
CREATE VIEW vw_fr_quality_trends AS
SELECT 
    tenant_id,
    data_source,
    overall_quality_score,
    quality_trend,
    LAG(overall_quality_score) OVER (PARTITION BY tenant_id, data_source ORDER BY last_assessment) as previous_score,
    overall_quality_score - LAG(overall_quality_score) OVER (PARTITION BY tenant_id, data_source ORDER BY last_assessment) as score_change
FROM cf_fr_data_quality_monitor
ORDER BY tenant_id, data_source, last_assessment DESC;

-- ==================================================================================
-- STORED PROCEDURES FOR OPTIMIZATION
-- ==================================================================================

-- Function to automatically partition new months
CREATE OR REPLACE FUNCTION create_monthly_partitions(table_name TEXT, months_ahead INTEGER DEFAULT 3)
RETURNS VOID AS $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
    i INTEGER;
BEGIN
    FOR i IN 0..months_ahead LOOP
        start_date := DATE_TRUNC('month', CURRENT_DATE) + (i || ' months')::INTERVAL;
        end_date := start_date + INTERVAL '1 month';
        partition_name := table_name || '_' || TO_CHAR(start_date, 'YYYY_MM');
        
        EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
            partition_name, table_name || '_partitioned', start_date, end_date);
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to update AI model performance metrics
CREATE OR REPLACE FUNCTION update_ai_performance_metrics()
RETURNS VOID AS $$
BEGIN
    -- Update conversation interface success rates
    UPDATE cf_fr_conversational_interface 
    SET learning_data = jsonb_set(
        COALESCE(learning_data, '{}'),
        '{success_rate}',
        to_jsonb((
            SELECT COUNT(*) FILTER (WHERE resolution_success = TRUE) * 100.0 / COUNT(*)
            FROM cf_fr_conversational_interface ci2 
            WHERE ci2.tenant_id = cf_fr_conversational_interface.tenant_id 
            AND ci2.query_timestamp >= CURRENT_DATE - INTERVAL '7 days'
        ))
    )
    WHERE query_timestamp >= CURRENT_DATE - INTERVAL '1 day';
    
    -- Update insight engine accuracy
    UPDATE cf_fr_ai_insight_engine
    SET accuracy_validation = (
        SELECT AVG(user_feedback_score::NUMERIC / 10.0)
        FROM jsonb_array_elements_text(feedback_ratings) AS rating(user_feedback_score)
        WHERE rating.user_feedback_score ~ '^[0-9]+(\.[0-9]+)?$'
    )
    WHERE accuracy_validation IS NULL AND jsonb_array_length(feedback_ratings) > 0;
END;
$$ LANGUAGE plpgsql;

-- ==================================================================================
-- MAINTENANCE AND OPTIMIZATION PROCEDURES
-- ==================================================================================

-- Automated index maintenance
CREATE OR REPLACE FUNCTION maintain_indexes()
RETURNS VOID AS $$
BEGIN
    -- Reindex fragmented indexes
    REINDEX INDEX CONCURRENTLY idx_conversation_timestamp;
    REINDEX INDEX CONCURRENTLY idx_quality_assessment;
    REINDEX INDEX CONCURRENTLY idx_audit_timestamp;
    
    -- Update table statistics
    ANALYZE cf_fr_conversational_interface;
    ANALYZE cf_fr_ai_insight_engine;
    ANALYZE cf_fr_predictive_analytics;
    ANALYZE cf_fr_data_quality_monitor;
END;
$$ LANGUAGE plpgsql;

-- Schedule maintenance jobs
SELECT cron.schedule('maintain_fr_indexes', '0 2 * * *', 'SELECT maintain_indexes();');
SELECT cron.schedule('update_ai_metrics', '0 */6 * * *', 'SELECT update_ai_performance_metrics();');
SELECT cron.schedule('create_partitions', '0 0 1 * *', 'SELECT create_monthly_partitions(''cf_fr_conversational_interface'', 3);');

-- ==================================================================================
-- SECURITY AND ACCESS CONTROL
-- ==================================================================================

-- Row Level Security for multi-tenancy
ALTER TABLE cf_fr_report_template ENABLE ROW LEVEL SECURITY;
ALTER TABLE cf_fr_conversational_interface ENABLE ROW LEVEL SECURITY;
ALTER TABLE cf_fr_ai_insight_engine ENABLE ROW LEVEL SECURITY;
ALTER TABLE cf_fr_predictive_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE cf_fr_real_time_collaboration ENABLE ROW LEVEL SECURITY;
ALTER TABLE cf_fr_data_quality_monitor ENABLE ROW LEVEL SECURITY;
ALTER TABLE cf_fr_blockchain_audit_trail ENABLE ROW LEVEL SECURITY;

-- Tenant isolation policies
CREATE POLICY tenant_isolation_report_template ON cf_fr_report_template
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_conversation ON cf_fr_conversational_interface
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_insights ON cf_fr_ai_insight_engine
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_predictions ON cf_fr_predictive_analytics
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_collaboration ON cf_fr_real_time_collaboration
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_quality ON cf_fr_data_quality_monitor
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_audit ON cf_fr_blockchain_audit_trail
    USING (tenant_id = current_setting('app.current_tenant_id', true));

-- ==================================================================================
-- PERFORMANCE MONITORING AND ALERTS
-- ==================================================================================

-- Performance monitoring table
CREATE TABLE cf_fr_performance_monitor (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    threshold_value DECIMAL(15,4),
    alert_level VARCHAR(20) DEFAULT 'info',
    tenant_id VARCHAR(36),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX (metric_name, recorded_at),
    INDEX (tenant_id, recorded_at)
);

-- Function to record performance metrics
CREATE OR REPLACE FUNCTION record_performance_metric(
    p_metric_name VARCHAR(100),
    p_metric_value DECIMAL(15,4),
    p_threshold_value DECIMAL(15,4) DEFAULT NULL,
    p_tenant_id VARCHAR(36) DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    alert_level VARCHAR(20) := 'info';
BEGIN
    -- Determine alert level based on threshold
    IF p_threshold_value IS NOT NULL AND p_metric_value > p_threshold_value THEN
        alert_level := 'warning';
    END IF;
    
    -- Insert performance metric
    INSERT INTO cf_fr_performance_monitor (
        metric_name, metric_value, threshold_value, alert_level, tenant_id
    ) VALUES (
        p_metric_name, p_metric_value, p_threshold_value, alert_level, p_tenant_id
    );
    
    -- Trigger alert if threshold exceeded
    IF alert_level = 'warning' THEN
        PERFORM pg_notify('performance_alert', 
            json_build_object(
                'metric', p_metric_name,
                'value', p_metric_value,
                'threshold', p_threshold_value,
                'tenant_id', p_tenant_id
            )::text
        );
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ==================================================================================
-- DATABASE COMMENTS FOR DOCUMENTATION
-- ==================================================================================

COMMENT ON TABLE cf_fr_report_template IS 'Revolutionary AI-powered financial report templates with adaptive intelligence and natural language capabilities';
COMMENT ON TABLE cf_fr_conversational_interface IS 'Natural language processing interface for conversational financial reporting';
COMMENT ON TABLE cf_fr_ai_insight_engine IS 'AI-powered financial insight generation with machine learning analysis';
COMMENT ON TABLE cf_fr_predictive_analytics IS 'Advanced predictive analytics for financial forecasting and trend analysis';
COMMENT ON TABLE cf_fr_real_time_collaboration IS 'Real-time collaborative financial reporting with intelligent conflict resolution';
COMMENT ON TABLE cf_fr_data_quality_monitor IS 'Intelligent data quality monitoring with AI-powered anomaly detection';
COMMENT ON TABLE cf_fr_blockchain_audit_trail IS 'Blockchain-based immutable audit trail for regulatory compliance';

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA public TO fr_app_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fr_app_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO fr_app_role;