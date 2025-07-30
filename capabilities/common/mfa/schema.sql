-- APG Multi-Factor Authentication (MFA) - Database Schema
-- 
-- Normalized PostgreSQL schema with performance indexes, multi-tenancy support,
-- and comprehensive constraints for revolutionary MFA capability.
--
-- Copyright © 2025 Datacraft
-- Author: Nyimbi Odero <nyimbi@gmail.com>
-- Website: www.datacraft.co.ke

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create custom types
CREATE TYPE mfa_method_type AS ENUM (
	'biometric_face',
	'biometric_voice', 
	'biometric_behavioral',
	'biometric_multi_modal',
	'token_totp',
	'token_hotp',
	'token_hardware',
	'sms',
	'email',
	'push_notification',
	'backup_codes',
	'delegation_token'
);

CREATE TYPE trust_level AS ENUM (
	'unknown',
	'low',
	'medium', 
	'high',
	'verified'
);

CREATE TYPE risk_level AS ENUM (
	'minimal',
	'low',
	'medium',
	'high',
	'critical'
);

CREATE TYPE authentication_status AS ENUM (
	'pending',
	'in_progress',
	'success',
	'failed',
	'expired',
	'revoked'
);

-- MFA User Profiles Table
CREATE TABLE mfa_user_profiles (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id VARCHAR(255) NOT NULL,
	user_id VARCHAR(255) NOT NULL,
	
	-- Trust and risk scoring
	base_risk_score DECIMAL(3,2) NOT NULL DEFAULT 0.50 CHECK (base_risk_score >= 0.0 AND base_risk_score <= 1.0),
	trust_score DECIMAL(3,2) NOT NULL DEFAULT 0.00 CHECK (trust_score >= 0.0 AND trust_score <= 1.0),
	behavioral_baseline JSONB DEFAULT '{}',
	
	-- Device and location trust
	trusted_devices TEXT[] DEFAULT '{}',
	trusted_locations JSONB DEFAULT '[]',
	device_trust_scores JSONB DEFAULT '{}',
	
	-- Authentication statistics
	total_authentications INTEGER DEFAULT 0,
	successful_authentications INTEGER DEFAULT 0,
	failed_authentications INTEGER DEFAULT 0,
	last_successful_auth TIMESTAMP WITH TIME ZONE,
	
	-- Security settings
	lockout_until TIMESTAMP WITH TIME ZONE,
	security_notifications_enabled BOOLEAN DEFAULT TRUE,
	adaptive_auth_enabled BOOLEAN DEFAULT TRUE,
	
	-- Compliance and privacy
	consent_given BOOLEAN DEFAULT FALSE,
	data_retention_days INTEGER DEFAULT 365,
	privacy_settings JSONB DEFAULT '{}',
	
	-- AI/ML insights
	ml_insights JSONB DEFAULT '{}',
	risk_patterns TEXT[] DEFAULT '{}',
	recommendation_engine_data JSONB DEFAULT '{}',
	
	-- Audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255) NOT NULL,
	updated_by VARCHAR(255) NOT NULL,
	
	-- Constraints
	CONSTRAINT unique_user_per_tenant UNIQUE (tenant_id, user_id)
);

-- Device Bindings Table
CREATE TABLE mfa_device_bindings (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id VARCHAR(255) NOT NULL,
	device_id VARCHAR(255) NOT NULL,
	device_type VARCHAR(100) NOT NULL,
	device_name VARCHAR(255) NOT NULL,
	device_fingerprint VARCHAR(512) NOT NULL,
	trust_level trust_level DEFAULT 'unknown',
	last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	is_active BOOLEAN DEFAULT TRUE,
	
	-- Security attributes
	is_rooted_jailbroken BOOLEAN,
	os_version VARCHAR(100),
	app_version VARCHAR(100),
	location_data JSONB,
	
	-- Audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255) NOT NULL,
	updated_by VARCHAR(255) NOT NULL,
	
	-- Constraints
	CONSTRAINT unique_device_per_tenant UNIQUE (tenant_id, device_id)
);

-- Biometric Templates Table
CREATE TABLE mfa_biometric_templates (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id VARCHAR(255) NOT NULL,
	biometric_type VARCHAR(50) NOT NULL,
	template_data TEXT NOT NULL, -- Encrypted biometric template
	template_version VARCHAR(20) NOT NULL,
	quality_score DECIMAL(3,2) NOT NULL CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
	enrollment_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	last_verified TIMESTAMP WITH TIME ZONE,
	verification_count INTEGER DEFAULT 0,
	
	-- Privacy and security
	is_encrypted BOOLEAN DEFAULT TRUE,
	encryption_method VARCHAR(50) DEFAULT 'aes256',
	can_be_reconstructed BOOLEAN DEFAULT FALSE,
	
	-- Audit fields  
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255) NOT NULL,
	updated_by VARCHAR(255) NOT NULL
);

-- MFA Methods Table
CREATE TABLE mfa_methods (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id VARCHAR(255) NOT NULL,
	user_id VARCHAR(255) NOT NULL,
	user_profile_id UUID NOT NULL REFERENCES mfa_user_profiles(id) ON DELETE CASCADE,
	method_type mfa_method_type NOT NULL,
	method_name VARCHAR(255) NOT NULL,
	is_primary BOOLEAN DEFAULT FALSE,
	is_active BOOLEAN DEFAULT TRUE,
	
	-- Method configuration
	method_config JSONB DEFAULT '{}',
	trust_level trust_level DEFAULT 'medium',
	
	-- Associated data
	device_binding_id UUID REFERENCES mfa_device_bindings(id) ON DELETE SET NULL,
	biometric_template_id UUID REFERENCES mfa_biometric_templates(id) ON DELETE SET NULL,
	backup_codes TEXT[],
	
	-- Usage statistics
	total_uses INTEGER DEFAULT 0,
	success_rate DECIMAL(3,2) DEFAULT 0.00 CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
	last_used TIMESTAMP WITH TIME ZONE,
	consecutive_failures INTEGER DEFAULT 0,
	
	-- Security settings
	max_failures_before_lockout INTEGER DEFAULT 5,
	lockout_duration_minutes INTEGER DEFAULT 30,
	requires_device_binding BOOLEAN DEFAULT TRUE,
	
	-- Audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255) NOT NULL,
	updated_by VARCHAR(255) NOT NULL
);

-- Risk Assessments Table
CREATE TABLE mfa_risk_assessments (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id VARCHAR(255) NOT NULL,
	user_id VARCHAR(255) NOT NULL,
	session_id VARCHAR(255) NOT NULL,
	
	-- Risk scoring
	overall_risk_score DECIMAL(3,2) NOT NULL CHECK (overall_risk_score >= 0.0 AND overall_risk_score <= 1.0),
	risk_level risk_level NOT NULL,
	confidence_level DECIMAL(3,2) NOT NULL CHECK (confidence_level >= 0.0 AND confidence_level <= 1.0),
	
	-- Risk factors (stored as JSONB for flexibility)
	risk_factors JSONB DEFAULT '[]',
	
	-- Context information
	device_context JSONB DEFAULT '{}',
	location_context JSONB DEFAULT '{}',
	behavioral_context JSONB DEFAULT '{}',
	temporal_context JSONB DEFAULT '{}',
	
	-- Recommendations
	recommended_auth_methods TEXT[],
	recommended_actions TEXT[],
	
	-- AI/ML metadata
	model_version VARCHAR(50) NOT NULL,
	processing_time_ms INTEGER NOT NULL,
	
	-- Audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255) NOT NULL,
	updated_by VARCHAR(255) NOT NULL
);

-- Authentication Events Table
CREATE TABLE mfa_auth_events (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id VARCHAR(255) NOT NULL,
	user_id VARCHAR(255) NOT NULL,
	session_id VARCHAR(255) NOT NULL,
	event_type VARCHAR(100) NOT NULL,
	
	-- Authentication details
	method_used mfa_method_type,
	method_id UUID REFERENCES mfa_methods(id) ON DELETE SET NULL,
	status authentication_status NOT NULL,
	
	-- Risk and trust
	risk_score DECIMAL(3,2) CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
	trust_score DECIMAL(3,2) CHECK (trust_score >= 0.0 AND trust_score <= 1.0),
	
	-- Context information
	device_info JSONB DEFAULT '{}',
	location_info JSONB DEFAULT '{}',
	network_info JSONB DEFAULT '{}',
	
	-- Timing and performance
	duration_ms INTEGER,
	retry_count INTEGER DEFAULT 0,
	
	-- Error information
	error_code VARCHAR(50),
	error_message TEXT,
	
	-- APG integration
	audit_trail_id UUID,
	notification_sent BOOLEAN DEFAULT FALSE,
	
	-- Audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255) NOT NULL,
	updated_by VARCHAR(255) NOT NULL
);

-- Recovery Methods Table
CREATE TABLE mfa_recovery_methods (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id VARCHAR(255) NOT NULL,
	user_id VARCHAR(255) NOT NULL,
	user_profile_id UUID NOT NULL REFERENCES mfa_user_profiles(id) ON DELETE CASCADE,
	recovery_type VARCHAR(100) NOT NULL,
	recovery_name VARCHAR(255) NOT NULL,
	is_active BOOLEAN DEFAULT TRUE,
	
	-- Recovery configuration
	recovery_config JSONB DEFAULT '{}',
	verification_required BOOLEAN DEFAULT TRUE,
	
	-- Security settings
	max_uses_per_day INTEGER DEFAULT 3,
	uses_today INTEGER DEFAULT 0,
	last_used TIMESTAMP WITH TIME ZONE,
	
	-- Emergency settings
	is_emergency_method BOOLEAN DEFAULT FALSE,
	requires_admin_approval BOOLEAN DEFAULT FALSE,
	
	-- Audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255) NOT NULL,
	updated_by VARCHAR(255) NOT NULL
);

-- Authentication Tokens Table
CREATE TABLE mfa_auth_tokens (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id VARCHAR(255) NOT NULL,
	user_id VARCHAR(255) NOT NULL,
	token_type VARCHAR(100) NOT NULL,
	token_value TEXT NOT NULL, -- Encrypted token value
	
	-- Token lifecycle
	issued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
	is_active BOOLEAN DEFAULT TRUE,
	is_single_use BOOLEAN DEFAULT FALSE,
	
	-- Usage tracking
	used_count INTEGER DEFAULT 0,
	max_uses INTEGER,
	last_used TIMESTAMP WITH TIME ZONE,
	
	-- Security context
	device_binding_required BOOLEAN DEFAULT TRUE,
	allowed_devices TEXT[] DEFAULT '{}',
	ip_restrictions TEXT[] DEFAULT '{}',
	
	-- Delegation (for collaborative authentication)
	is_delegation_token BOOLEAN DEFAULT FALSE,
	delegated_by VARCHAR(255),
	delegation_scope TEXT[] DEFAULT '{}',
	
	-- Audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255) NOT NULL,
	updated_by VARCHAR(255) NOT NULL
);

-- Create indexes for performance
CREATE INDEX idx_mfa_user_profiles_tenant_user ON mfa_user_profiles(tenant_id, user_id);
CREATE INDEX idx_mfa_user_profiles_trust_score ON mfa_user_profiles(trust_score);
CREATE INDEX idx_mfa_user_profiles_last_auth ON mfa_user_profiles(last_successful_auth);

CREATE INDEX idx_mfa_device_bindings_tenant_device ON mfa_device_bindings(tenant_id, device_id);
CREATE INDEX idx_mfa_device_bindings_trust_level ON mfa_device_bindings(trust_level);
CREATE INDEX idx_mfa_device_bindings_last_seen ON mfa_device_bindings(last_seen);

CREATE INDEX idx_mfa_biometric_templates_tenant_type ON mfa_biometric_templates(tenant_id, biometric_type);
CREATE INDEX idx_mfa_biometric_templates_quality ON mfa_biometric_templates(quality_score);

CREATE INDEX idx_mfa_methods_tenant_user ON mfa_methods(tenant_id, user_id);
CREATE INDEX idx_mfa_methods_user_profile ON mfa_methods(user_profile_id);
CREATE INDEX idx_mfa_methods_type ON mfa_methods(method_type);
CREATE INDEX idx_mfa_methods_active_primary ON mfa_methods(is_active, is_primary);

CREATE INDEX idx_mfa_risk_assessments_tenant_user ON mfa_risk_assessments(tenant_id, user_id);
CREATE INDEX idx_mfa_risk_assessments_session ON mfa_risk_assessments(session_id);
CREATE INDEX idx_mfa_risk_assessments_risk_level ON mfa_risk_assessments(risk_level);
CREATE INDEX idx_mfa_risk_assessments_created_at ON mfa_risk_assessments(created_at);

CREATE INDEX idx_mfa_auth_events_tenant_user ON mfa_auth_events(tenant_id, user_id);
CREATE INDEX idx_mfa_auth_events_session ON mfa_auth_events(session_id);
CREATE INDEX idx_mfa_auth_events_status ON mfa_auth_events(status);
CREATE INDEX idx_mfa_auth_events_created_at ON mfa_auth_events(created_at);
CREATE INDEX idx_mfa_auth_events_method ON mfa_auth_events(method_used);

CREATE INDEX idx_mfa_recovery_methods_tenant_user ON mfa_recovery_methods(tenant_id, user_id);
CREATE INDEX idx_mfa_recovery_methods_user_profile ON mfa_recovery_methods(user_profile_id);
CREATE INDEX idx_mfa_recovery_methods_type ON mfa_recovery_methods(recovery_type);

CREATE INDEX idx_mfa_auth_tokens_tenant_user ON mfa_auth_tokens(tenant_id, user_id);
CREATE INDEX idx_mfa_auth_tokens_expires_at ON mfa_auth_tokens(expires_at);
CREATE INDEX idx_mfa_auth_tokens_active ON mfa_auth_tokens(is_active);
CREATE INDEX idx_mfa_auth_tokens_delegation ON mfa_auth_tokens(is_delegation_token);

-- Create text search indexes
CREATE INDEX idx_mfa_user_profiles_search ON mfa_user_profiles USING gin(to_tsvector('english', user_id));
CREATE INDEX idx_mfa_auth_events_search ON mfa_auth_events USING gin(to_tsvector('english', event_type || ' ' || COALESCE(error_message, '')));

-- Create partial indexes for active records
CREATE INDEX idx_mfa_methods_active ON mfa_methods(tenant_id, user_id) WHERE is_active = TRUE;
CREATE INDEX idx_mfa_device_bindings_active ON mfa_device_bindings(tenant_id, device_id) WHERE is_active = TRUE;
CREATE INDEX idx_mfa_recovery_methods_active ON mfa_recovery_methods(tenant_id, user_id) WHERE is_active = TRUE;
CREATE INDEX idx_mfa_auth_tokens_active ON mfa_auth_tokens(tenant_id, user_id) WHERE is_active = TRUE AND expires_at > CURRENT_TIMESTAMP;

-- Create trigger for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language plpgsql;

-- Apply updated_at triggers to all tables
CREATE TRIGGER update_mfa_user_profiles_updated_at BEFORE UPDATE ON mfa_user_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_mfa_device_bindings_updated_at BEFORE UPDATE ON mfa_device_bindings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_mfa_biometric_templates_updated_at BEFORE UPDATE ON mfa_biometric_templates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_mfa_methods_updated_at BEFORE UPDATE ON mfa_methods FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_mfa_risk_assessments_updated_at BEFORE UPDATE ON mfa_risk_assessments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_mfa_auth_events_updated_at BEFORE UPDATE ON mfa_auth_events FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_mfa_recovery_methods_updated_at BEFORE UPDATE ON mfa_recovery_methods FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_mfa_auth_tokens_updated_at BEFORE UPDATE ON mfa_auth_tokens FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE VIEW mfa_active_user_methods AS
SELECT 
	m.id,
	m.tenant_id,
	m.user_id,
	m.method_type,
	m.method_name,
	m.is_primary,
	m.trust_level,
	m.success_rate,
	m.last_used,
	db.device_name,
	db.trust_level AS device_trust_level
FROM mfa_methods m
LEFT JOIN mfa_device_bindings db ON m.device_binding_id = db.id
WHERE m.is_active = TRUE;

CREATE VIEW mfa_user_security_overview AS
SELECT 
	up.id,
	up.tenant_id,
	up.user_id,
	up.trust_score,
	up.base_risk_score,
	up.total_authentications,
	up.successful_authentications,
	up.failed_authentications,
	up.last_successful_auth,
	up.lockout_until,
	COUNT(m.id) AS active_methods_count,
	COUNT(rm.id) AS recovery_methods_count
FROM mfa_user_profiles up
LEFT JOIN mfa_methods m ON up.id = m.user_profile_id AND m.is_active = TRUE
LEFT JOIN mfa_recovery_methods rm ON up.id = rm.user_profile_id AND rm.is_active = TRUE
GROUP BY up.id;

-- Row Level Security (RLS) for multi-tenancy
ALTER TABLE mfa_user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE mfa_device_bindings ENABLE ROW LEVEL SECURITY;
ALTER TABLE mfa_biometric_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE mfa_methods ENABLE ROW LEVEL SECURITY;
ALTER TABLE mfa_risk_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE mfa_auth_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE mfa_recovery_methods ENABLE ROW LEVEL SECURITY;
ALTER TABLE mfa_auth_tokens ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (example - adjust based on your tenant isolation strategy)
CREATE POLICY mfa_user_profiles_tenant_isolation ON mfa_user_profiles
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY mfa_device_bindings_tenant_isolation ON mfa_device_bindings
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY mfa_biometric_templates_tenant_isolation ON mfa_biometric_templates
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY mfa_methods_tenant_isolation ON mfa_methods
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY mfa_risk_assessments_tenant_isolation ON mfa_risk_assessments
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY mfa_auth_events_tenant_isolation ON mfa_auth_events
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY mfa_recovery_methods_tenant_isolation ON mfa_recovery_methods
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY mfa_auth_tokens_tenant_isolation ON mfa_auth_tokens
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));

-- Grant permissions (adjust based on your application roles)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO mfa_app_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mfa_app_role;
GRANT SELECT ON mfa_active_user_methods, mfa_user_security_overview TO mfa_readonly_role;

-- Create maintenance functions
CREATE OR REPLACE FUNCTION cleanup_expired_tokens()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM mfa_auth_tokens 
    WHERE expires_at < CURRENT_TIMESTAMP 
    AND is_active = FALSE;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_user_auth_statistics()
RETURNS VOID AS $$
BEGIN
    UPDATE mfa_user_profiles SET
        total_authentications = (
            SELECT COUNT(*)
            FROM mfa_auth_events e
            WHERE e.tenant_id = mfa_user_profiles.tenant_id
            AND e.user_id = mfa_user_profiles.user_id
        ),
        successful_authentications = (
            SELECT COUNT(*)
            FROM mfa_auth_events e
            WHERE e.tenant_id = mfa_user_profiles.tenant_id
            AND e.user_id = mfa_user_profiles.user_id
            AND e.status = 'success'
        ),
        failed_authentications = (
            SELECT COUNT(*)
            FROM mfa_auth_events e
            WHERE e.tenant_id = mfa_user_profiles.tenant_id
            AND e.user_id = mfa_user_profiles.user_id
            AND e.status = 'failed'
        ),
        last_successful_auth = (
            SELECT MAX(created_at)
            FROM mfa_auth_events e
            WHERE e.tenant_id = mfa_user_profiles.tenant_id
            AND e.user_id = mfa_user_profiles.user_id
            AND e.status = 'success'
        );
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE mfa_user_profiles IS 'Comprehensive MFA user profiles with AI-powered insights and behavioral baselines';
COMMENT ON TABLE mfa_device_bindings IS 'Device binding information for secure MFA method association';
COMMENT ON TABLE mfa_biometric_templates IS 'Encrypted biometric templates with privacy protection';
COMMENT ON TABLE mfa_methods IS 'Configured MFA methods with usage statistics and security settings';
COMMENT ON TABLE mfa_risk_assessments IS 'AI-powered risk assessments for adaptive authentication';
COMMENT ON TABLE mfa_auth_events IS 'Comprehensive authentication event audit trail';
COMMENT ON TABLE mfa_recovery_methods IS 'Account recovery methods with usage limits and security controls';
COMMENT ON TABLE mfa_auth_tokens IS 'Authentication tokens with delegation support for collaborative scenarios';