-- Payment Gateway Database Schema
-- Comprehensive PostgreSQL schema for APG Payment Gateway capability
-- Includes tables, indexes, constraints, and triggers for production deployment
-- 
-- © 2025 Datacraft. All rights reserved.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schema
CREATE SCHEMA IF NOT EXISTS payment_gateway;
SET search_path TO payment_gateway, public;

-- =====================================================
-- CORE PAYMENT TABLES
-- =====================================================

-- Merchants table
CREATE TABLE pg_merchants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    merchant_code VARCHAR(50) UNIQUE NOT NULL,
    business_name VARCHAR(255) NOT NULL,
    legal_name VARCHAR(255),
    contact_email VARCHAR(255) NOT NULL,
    contact_phone VARCHAR(50),
    website_url VARCHAR(500),
    business_type VARCHAR(100),
    country_code CHAR(2) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    timezone VARCHAR(50) DEFAULT 'UTC',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'inactive')),
    api_key_hash VARCHAR(255),
    webhook_url VARCHAR(500),
    webhook_secret VARCHAR(255),
    settings JSONB DEFAULT '{}',
    risk_profile VARCHAR(20) DEFAULT 'medium' CHECK (risk_profile IN ('low', 'medium', 'high')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,
    updated_by UUID
);

-- Payment methods table
CREATE TABLE pg_payment_methods (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID,
    merchant_id UUID REFERENCES pg_merchants(id),
    type VARCHAR(50) NOT NULL CHECK (type IN ('credit_card', 'debit_card', 'mpesa', 'paypal', 'bank_transfer', 'digital_wallet')),
    provider VARCHAR(50),
    token VARCHAR(255), -- Tokenized payment method
    details JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    is_default BOOLEAN DEFAULT FALSE,
    is_verified BOOLEAN DEFAULT FALSE,
    expires_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'expired', 'disabled')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Payment transactions table
CREATE TABLE pg_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    merchant_id UUID NOT NULL REFERENCES pg_merchants(id),
    customer_id UUID,
    payment_method_id UUID REFERENCES pg_payment_methods(id),
    parent_transaction_id UUID REFERENCES pg_transactions(id), -- For refunds/captures
    transaction_type VARCHAR(20) DEFAULT 'payment' CHECK (transaction_type IN ('payment', 'refund', 'capture', 'void')),
    amount BIGINT NOT NULL, -- Amount in smallest currency unit (cents)
    currency VARCHAR(3) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled', 'expired')),
    description TEXT,
    reference_id VARCHAR(255), -- Merchant reference
    processor_name VARCHAR(50),
    processor_transaction_id VARCHAR(255),
    processor_reference VARCHAR(255),
    payment_method_type VARCHAR(50) NOT NULL,
    gateway_fee BIGINT DEFAULT 0,
    processor_fee BIGINT DEFAULT 0,
    net_amount BIGINT,
    authorized_at TIMESTAMP WITH TIME ZONE,
    captured_at TIMESTAMP WITH TIME ZONE,
    settled_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    error_code VARCHAR(100),
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    risk_score DECIMAL(5,4),
    fraud_flags JSONB DEFAULT '[]',
    ip_address INET,
    user_agent TEXT,
    device_fingerprint VARCHAR(255),
    location_data JSONB,
    processing_time_ms INTEGER,
    retry_count INTEGER DEFAULT 0,
    webhook_delivered BOOLEAN DEFAULT FALSE,
    webhook_attempts INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Payment processors table
CREATE TABLE pg_processors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('mpesa', 'stripe', 'adyen', 'paypal', 'custom')),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance')),
    priority INTEGER DEFAULT 100,
    supported_methods JSONB NOT NULL DEFAULT '[]',
    supported_currencies JSONB NOT NULL DEFAULT '[]',
    supported_countries JSONB NOT NULL DEFAULT '[]',
    configuration JSONB DEFAULT '{}',
    credentials JSONB DEFAULT '{}', -- Encrypted
    webhook_config JSONB DEFAULT '{}',
    rate_limits JSONB DEFAULT '{}',
    fees JSONB DEFAULT '{}',
    health_status VARCHAR(20) DEFAULT 'unknown',
    last_health_check TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- FRAUD DETECTION TABLES
-- =====================================================

-- Fraud analysis table
CREATE TABLE pg_fraud_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID NOT NULL REFERENCES pg_transactions(id),
    risk_score DECIMAL(5,4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    analysis_result VARCHAR(20) NOT NULL CHECK (analysis_result IN ('approve', 'review', 'decline')),
    rules_triggered JSONB DEFAULT '[]',
    ml_model_scores JSONB DEFAULT '{}',
    behavioral_analysis JSONB DEFAULT '{}',
    device_analysis JSONB DEFAULT '{}',
    location_analysis JSONB DEFAULT '{}',
    velocity_analysis JSONB DEFAULT '{}',
    network_analysis JSONB DEFAULT '{}',
    analysis_duration_ms INTEGER,
    model_versions JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Fraud rules table
CREATE TABLE pg_fraud_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    rule_type VARCHAR(50) NOT NULL CHECK (rule_type IN ('velocity', 'amount', 'location', 'device', 'pattern', 'blacklist')),
    conditions JSONB NOT NULL,
    action VARCHAR(20) NOT NULL CHECK (action IN ('approve', 'review', 'decline', 'flag')),
    priority INTEGER DEFAULT 100,
    is_active BOOLEAN DEFAULT TRUE,
    merchant_id UUID REFERENCES pg_merchants(id), -- NULL for global rules
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Device fingerprints table
CREATE TABLE pg_device_fingerprints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fingerprint_hash VARCHAR(255) UNIQUE NOT NULL,
    device_info JSONB NOT NULL,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    transaction_count INTEGER DEFAULT 0,
    successful_transactions INTEGER DEFAULT 0,
    failed_transactions INTEGER DEFAULT 0,
    risk_score DECIMAL(5,4) DEFAULT 0.5,
    is_blacklisted BOOLEAN DEFAULT FALSE,
    blacklist_reason TEXT,
    metadata JSONB DEFAULT '{}'
);

-- =====================================================
-- ANALYTICS AND REPORTING TABLES
-- =====================================================

-- Transaction metrics (for fast analytics)
CREATE TABLE pg_transaction_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date_hour TIMESTAMP WITH TIME ZONE NOT NULL, -- Truncated to hour
    merchant_id UUID REFERENCES pg_merchants(id),
    processor_name VARCHAR(50),
    payment_method_type VARCHAR(50),
    currency VARCHAR(3),
    country_code CHAR(2),
    status VARCHAR(20),
    transaction_count INTEGER DEFAULT 0,
    total_amount BIGINT DEFAULT 0,
    successful_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    fraud_count INTEGER DEFAULT 0,
    avg_processing_time_ms INTEGER DEFAULT 0,
    avg_risk_score DECIMAL(5,4) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(date_hour, merchant_id, processor_name, payment_method_type, currency, status)
);

-- User behavior profiles (for smart completion)
CREATE TABLE pg_user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    preferred_payment_methods JSONB DEFAULT '[]',
    typical_amounts JSONB DEFAULT '[]',
    common_merchants JSONB DEFAULT '[]',
    geographic_patterns JSONB DEFAULT '{}',
    device_preferences JSONB DEFAULT '{}',
    time_patterns JSONB DEFAULT '{}',
    completion_history JSONB DEFAULT '[]',
    accuracy_score DECIMAL(5,4) DEFAULT 0.5,
    personalization_score DECIMAL(5,4) DEFAULT 0.0,
    last_interaction TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id)
);

-- Dashboard configurations
CREATE TABLE pg_dashboards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    view_type VARCHAR(50) NOT NULL,
    widgets JSONB NOT NULL DEFAULT '[]',
    layout JSONB DEFAULT '{}',
    theme VARCHAR(50) DEFAULT 'dark',
    auto_refresh BOOLEAN DEFAULT TRUE,
    refresh_rate_seconds INTEGER DEFAULT 30,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- WEBHOOK AND NOTIFICATIONS TABLES
-- =====================================================

-- Webhook events table
CREATE TABLE pg_webhook_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    merchant_id UUID NOT NULL REFERENCES pg_merchants(id),
    transaction_id UUID REFERENCES pg_transactions(id),
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'delivered', 'failed', 'cancelled')),
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 5,
    next_attempt_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE,
    response_status INTEGER,
    response_body TEXT,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Notification preferences
CREATE TABLE pg_notification_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    merchant_id UUID NOT NULL REFERENCES pg_merchants(id),
    event_type VARCHAR(100) NOT NULL,
    email_enabled BOOLEAN DEFAULT TRUE,
    webhook_enabled BOOLEAN DEFAULT TRUE,
    sms_enabled BOOLEAN DEFAULT FALSE,
    push_enabled BOOLEAN DEFAULT FALSE,
    threshold_amount BIGINT,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(merchant_id, event_type)
);

-- =====================================================
-- AUDIT AND LOGGING TABLES
-- =====================================================

-- Audit log table
CREATE TABLE pg_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100) NOT NULL,
    record_id UUID NOT NULL,
    action VARCHAR(20) NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    changed_fields JSONB,
    user_id UUID,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API access log
CREATE TABLE pg_api_access_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    merchant_id UUID REFERENCES pg_merchants(id),
    api_key_id UUID,
    endpoint VARCHAR(500) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    request_size INTEGER,
    response_size INTEGER,
    processing_time_ms INTEGER,
    ip_address INET,
    user_agent TEXT,
    request_id UUID,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Transaction indexes
CREATE INDEX idx_transactions_merchant_created ON pg_transactions(merchant_id, created_at DESC);
CREATE INDEX idx_transactions_customer_created ON pg_transactions(customer_id, created_at DESC);
CREATE INDEX idx_transactions_status_created ON pg_transactions(status, created_at DESC);
CREATE INDEX idx_transactions_processor_created ON pg_transactions(processor_name, created_at DESC);
CREATE INDEX idx_transactions_reference ON pg_transactions(reference_id);
CREATE INDEX idx_transactions_processor_ref ON pg_transactions(processor_transaction_id);
CREATE INDEX idx_transactions_amount_range ON pg_transactions(amount, created_at DESC);
CREATE INDEX idx_transactions_currency_date ON pg_transactions(currency, created_at DESC);

-- Payment method indexes
CREATE INDEX idx_payment_methods_customer ON pg_payment_methods(customer_id, is_default DESC);
CREATE INDEX idx_payment_methods_merchant ON pg_payment_methods(merchant_id, type);
CREATE INDEX idx_payment_methods_token ON pg_payment_methods(token) WHERE token IS NOT NULL;

-- Fraud analysis indexes
CREATE INDEX idx_fraud_analysis_transaction ON pg_fraud_analysis(transaction_id);
CREATE INDEX idx_fraud_analysis_risk_score ON pg_fraud_analysis(risk_score DESC, created_at DESC);
CREATE INDEX idx_fraud_analysis_result ON pg_fraud_analysis(analysis_result, created_at DESC);

-- Merchant indexes
CREATE INDEX idx_merchants_code ON pg_merchants(merchant_code);
CREATE INDEX idx_merchants_status ON pg_merchants(status, created_at DESC);
CREATE INDEX idx_merchants_country ON pg_merchants(country_code);

-- Analytics indexes
CREATE INDEX idx_transaction_metrics_date_merchant ON pg_transaction_metrics(date_hour DESC, merchant_id);
CREATE INDEX idx_transaction_metrics_processor ON pg_transaction_metrics(processor_name, date_hour DESC);
CREATE INDEX idx_transaction_metrics_method ON pg_transaction_metrics(payment_method_type, date_hour DESC);

-- User profile indexes
CREATE INDEX idx_user_profiles_user_id ON pg_user_profiles(user_id);
CREATE INDEX idx_user_profiles_last_interaction ON pg_user_profiles(last_interaction DESC);

-- Device fingerprint indexes
CREATE INDEX idx_device_fingerprints_hash ON pg_device_fingerprints(fingerprint_hash);
CREATE INDEX idx_device_fingerprints_risk ON pg_device_fingerprints(risk_score DESC);
CREATE INDEX idx_device_fingerprints_blacklist ON pg_device_fingerprints(is_blacklisted) WHERE is_blacklisted = TRUE;

-- Webhook indexes
CREATE INDEX idx_webhook_events_merchant_status ON pg_webhook_events(merchant_id, status, created_at DESC);
CREATE INDEX idx_webhook_events_next_attempt ON pg_webhook_events(next_attempt_at) WHERE status = 'pending';

-- Audit indexes
CREATE INDEX idx_audit_log_table_record ON pg_audit_log(table_name, record_id, created_at DESC);
CREATE INDEX idx_audit_log_user ON pg_audit_log(user_id, created_at DESC);
CREATE INDEX idx_api_access_log_merchant ON pg_api_access_log(merchant_id, created_at DESC);

-- =====================================================
-- TRIGGERS FOR AUDIT LOGGING
-- =====================================================

-- Function to handle audit logging
CREATE OR REPLACE FUNCTION pg_audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    -- Insert audit record
    INSERT INTO pg_audit_log (
        table_name,
        record_id,
        action,
        old_values,
        new_values,
        changed_fields,
        user_id
    ) VALUES (
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        TG_OP,
        CASE WHEN TG_OP = 'DELETE' THEN row_to_json(OLD) ELSE NULL END,
        CASE WHEN TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN row_to_json(NEW) ELSE NULL END,
        CASE 
            WHEN TG_OP = 'UPDATE' THEN 
                (SELECT json_object_agg(key, value) 
                 FROM json_each(row_to_json(NEW)) 
                 WHERE value IS DISTINCT FROM (row_to_json(OLD) ->> key)::json)
            ELSE NULL 
        END,
        current_setting('app.user_id', true)::UUID
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers to key tables
CREATE TRIGGER audit_trigger_merchants
    AFTER INSERT OR UPDATE OR DELETE ON pg_merchants
    FOR EACH ROW EXECUTE FUNCTION pg_audit_trigger_function();

CREATE TRIGGER audit_trigger_transactions
    AFTER INSERT OR UPDATE OR DELETE ON pg_transactions
    FOR EACH ROW EXECUTE FUNCTION pg_audit_trigger_function();

CREATE TRIGGER audit_trigger_payment_methods
    AFTER INSERT OR UPDATE OR DELETE ON pg_payment_methods
    FOR EACH ROW EXECUTE FUNCTION pg_audit_trigger_function();

-- =====================================================
-- FUNCTIONS AND STORED PROCEDURES
-- =====================================================

-- Function to update transaction metrics
CREATE OR REPLACE FUNCTION pg_update_transaction_metrics()
RETURNS TRIGGER AS $$
DECLARE
    metric_hour TIMESTAMP WITH TIME ZONE;
BEGIN
    -- Round to nearest hour
    metric_hour := date_trunc('hour', NEW.created_at);
    
    -- Insert or update metrics
    INSERT INTO pg_transaction_metrics (
        date_hour,
        merchant_id,
        processor_name,
        payment_method_type,
        currency,
        country_code,
        status,
        transaction_count,
        total_amount,
        successful_count,
        failed_count,
        fraud_count,
        avg_processing_time_ms,
        avg_risk_score
    ) VALUES (
        metric_hour,
        NEW.merchant_id,
        NEW.processor_name,
        NEW.payment_method_type,
        NEW.currency,
        COALESCE((NEW.location_data->>'country_code')::CHAR(2), 'XX'),
        NEW.status,
        1,
        NEW.amount,
        CASE WHEN NEW.status = 'completed' THEN 1 ELSE 0 END,
        CASE WHEN NEW.status = 'failed' THEN 1 ELSE 0 END,
        CASE WHEN NEW.risk_score > 0.8 THEN 1 ELSE 0 END,
        COALESCE(NEW.processing_time_ms, 0),
        COALESCE(NEW.risk_score, 0.5)
    )
    ON CONFLICT (date_hour, merchant_id, processor_name, payment_method_type, currency, status)
    DO UPDATE SET
        transaction_count = pg_transaction_metrics.transaction_count + 1,
        total_amount = pg_transaction_metrics.total_amount + NEW.amount,
        successful_count = pg_transaction_metrics.successful_count + 
            CASE WHEN NEW.status = 'completed' THEN 1 ELSE 0 END,
        failed_count = pg_transaction_metrics.failed_count + 
            CASE WHEN NEW.status = 'failed' THEN 1 ELSE 0 END,
        fraud_count = pg_transaction_metrics.fraud_count + 
            CASE WHEN NEW.risk_score > 0.8 THEN 1 ELSE 0 END,
        avg_processing_time_ms = (
            (pg_transaction_metrics.avg_processing_time_ms * (pg_transaction_metrics.transaction_count - 1)) + 
            COALESCE(NEW.processing_time_ms, 0)
        ) / pg_transaction_metrics.transaction_count,
        avg_risk_score = (
            (pg_transaction_metrics.avg_risk_score * (pg_transaction_metrics.transaction_count - 1)) + 
            COALESCE(NEW.risk_score, 0.5)
        ) / pg_transaction_metrics.transaction_count;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply metrics trigger
CREATE TRIGGER update_transaction_metrics_trigger
    AFTER INSERT ON pg_transactions
    FOR EACH ROW EXECUTE FUNCTION pg_update_transaction_metrics();

-- Function to clean old data
CREATE OR REPLACE FUNCTION pg_cleanup_old_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean old API access logs (keep 90 days)
    DELETE FROM pg_api_access_log 
    WHERE created_at < NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean old webhook events (keep successful for 30 days, failed for 90 days)
    DELETE FROM pg_webhook_events 
    WHERE (status = 'delivered' AND created_at < NOW() - INTERVAL '30 days')
       OR (status IN ('failed', 'cancelled') AND created_at < NOW() - INTERVAL '90 days');
    
    -- Clean old fraud analysis (keep 1 year)
    DELETE FROM pg_fraud_analysis 
    WHERE created_at < NOW() - INTERVAL '1 year';
    
    -- Clean old transaction metrics (keep 2 years)
    DELETE FROM pg_transaction_metrics 
    WHERE date_hour < NOW() - INTERVAL '2 years';
    
    -- Update device fingerprint stats
    UPDATE pg_device_fingerprints 
    SET last_seen = NOW() 
    WHERE last_seen < NOW() - INTERVAL '30 days';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Merchant summary view
CREATE VIEW pg_merchant_summary AS
SELECT 
    m.id,
    m.merchant_code,
    m.business_name,
    m.status,
    m.country_code,
    m.currency,
    COUNT(t.id) as total_transactions,
    COUNT(CASE WHEN t.status = 'completed' THEN 1 END) as successful_transactions,
    COUNT(CASE WHEN t.status = 'failed' THEN 1 END) as failed_transactions,
    COALESCE(SUM(CASE WHEN t.status = 'completed' THEN t.amount END), 0) as total_volume,
    COALESCE(AVG(t.risk_score), 0) as avg_risk_score,
    MAX(t.created_at) as last_transaction_at
FROM pg_merchants m
LEFT JOIN pg_transactions t ON m.id = t.merchant_id 
    AND t.created_at >= NOW() - INTERVAL '30 days'
GROUP BY m.id, m.merchant_code, m.business_name, m.status, m.country_code, m.currency;

-- Transaction summary view
CREATE VIEW pg_transaction_summary AS
SELECT 
    t.id,
    t.merchant_id,
    m.business_name as merchant_name,
    t.amount,
    t.currency,
    t.status,
    t.payment_method_type,
    t.processor_name,
    t.risk_score,
    t.processing_time_ms,
    t.created_at,
    fa.risk_level,
    fa.analysis_result
FROM pg_transactions t
JOIN pg_merchants m ON t.merchant_id = m.id
LEFT JOIN pg_fraud_analysis fa ON t.id = fa.transaction_id;

-- Daily metrics view
CREATE VIEW pg_daily_metrics AS
SELECT 
    DATE(date_hour) as date,
    merchant_id,
    processor_name,
    payment_method_type,
    currency,
    SUM(transaction_count) as daily_transactions,
    SUM(total_amount) as daily_volume,
    SUM(successful_count) as daily_successful,
    SUM(failed_count) as daily_failed,
    SUM(fraud_count) as daily_fraud,
    AVG(avg_processing_time_ms) as avg_processing_time,
    AVG(avg_risk_score) as avg_risk_score,
    CASE 
        WHEN SUM(transaction_count) > 0 
        THEN (SUM(successful_count)::DECIMAL / SUM(transaction_count)) * 100 
        ELSE 0 
    END as success_rate_percent
FROM pg_transaction_metrics
GROUP BY DATE(date_hour), merchant_id, processor_name, payment_method_type, currency;

-- =====================================================
-- SAMPLE DATA (FOR DEVELOPMENT)
-- =====================================================

-- Insert sample merchant
INSERT INTO pg_merchants (
    merchant_code, business_name, contact_email, country_code, currency, status
) VALUES (
    'TEST_MERCHANT_001', 'Test Business Ltd', 'test@example.com', 'KE', 'KES', 'active'
) ON CONFLICT (merchant_code) DO NOTHING;

-- Insert sample processors
INSERT INTO pg_processors (name, display_name, type, supported_methods, supported_currencies, supported_countries) VALUES
('mpesa', 'M-Pesa', 'mpesa', '["mpesa"]', '["KES"]', '["KE"]'),
('stripe', 'Stripe', 'stripe', '["credit_card", "debit_card"]', '["USD", "EUR", "GBP", "KES"]', '["US", "EU", "GB", "KE"]'),
('adyen', 'Adyen', 'adyen', '["credit_card", "debit_card", "digital_wallet"]', '["USD", "EUR", "GBP", "KES"]', '["US", "EU", "GB", "KE"]'),
('paypal', 'PayPal', 'paypal', '["paypal", "credit_card"]', '["USD", "EUR", "GBP"]', '["US", "EU", "GB"]')
ON CONFLICT (name) DO NOTHING;

-- =====================================================
-- PERMISSIONS AND SECURITY
-- =====================================================

-- Create roles
CREATE ROLE payment_gateway_read;
CREATE ROLE payment_gateway_write;
CREATE ROLE payment_gateway_admin;

-- Grant permissions
GRANT USAGE ON SCHEMA payment_gateway TO payment_gateway_read, payment_gateway_write, payment_gateway_admin;
GRANT SELECT ON ALL TABLES IN SCHEMA payment_gateway TO payment_gateway_read;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA payment_gateway TO payment_gateway_write;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA payment_gateway TO payment_gateway_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA payment_gateway TO payment_gateway_write, payment_gateway_admin;

-- Row Level Security policies would be added here for multi-tenant access

COMMENT ON SCHEMA payment_gateway IS 'APG Payment Gateway - Comprehensive payment processing schema';
COMMENT ON TABLE pg_transactions IS 'Core payment transactions with full audit trail';
COMMENT ON TABLE pg_merchants IS 'Merchant accounts and configurations';
COMMENT ON TABLE pg_fraud_analysis IS 'ML-powered fraud detection results';
COMMENT ON TABLE pg_transaction_metrics IS 'Pre-aggregated metrics for fast analytics';