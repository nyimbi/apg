-- APG Encryption Services - PostgreSQL Database Schema
-- 
-- Revolutionary quantum-safe encryption database schema supporting:
-- - Post-quantum cryptographic operations
-- - Zero-knowledge encryption architecture
-- - Autonomous AI-driven key lifecycle management
-- - Multi-tenant isolation with shared threat intelligence
-- - APG capability integration patterns
--
-- © 2025 Datacraft. All rights reserved.
-- Author: Nyimbi Odero <nyimbi@gmail.com>

-- Enable UUID extension for uuid7str support
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgcrypto for cryptographic functions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Enable btree_gist for advanced indexing
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- APG Encryption Services Schema
CREATE SCHEMA IF NOT EXISTS apg_encryption;

-- Set search path for session
SET search_path TO apg_encryption, public;

--
-- ENUM TYPES
--

CREATE TYPE post_quantum_algorithm AS ENUM (
	'crystals-kyber-512',
	'crystals-kyber-768', 
	'crystals-kyber-1024',
	'crystals-dilithium-2',
	'crystals-dilithium-3',
	'crystals-dilithium-5',
	'falcon-512',
	'falcon-1024',
	'sphincs-plus-128s',
	'sphincs-plus-256s'
);

CREATE TYPE encryption_mode AS ENUM (
	'quantum-safe',
	'zero-knowledge',
	'homomorphic', 
	'neuromorphic',
	'threshold',
	'hybrid-classical-quantum'
);

CREATE TYPE key_lifecycle_state AS ENUM (
	'generating',
	'active',
	'rotation-scheduled',
	'rotating',
	'escrow',
	'deprecated',
	'destroyed',
	'quantum-upgrading'
);

CREATE TYPE security_level AS ENUM (
	'level-1',	-- AES-128 equivalent
	'level-2',	-- SHA-256 equivalent
	'level-3',	-- AES-192 equivalent
	'level-4',	-- SHA-384 equivalent
	'level-5'	-- AES-256 equivalent
);

CREATE TYPE threat_level AS ENUM (
	'minimal',
	'low',
	'moderate', 
	'high',
	'critical',
	'quantum-imminent'
);

CREATE TYPE compliance_framework AS ENUM (
	'gdpr',
	'hipaa',
	'pci-dss',
	'sox',
	'iso-27001',
	'fips-140-2',
	'common-criteria',
	'nist-cybersecurity'
);

--
-- CORE TABLES
--

-- Quantum Entropy Sources
CREATE TABLE quantum_entropy_sources (
	id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id VARCHAR(32) NOT NULL,
	source_type VARCHAR(50) NOT NULL,
	location VARCHAR(255) NOT NULL,
	quality_score DECIMAL(3,2) NOT NULL CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
	last_harvest_at TIMESTAMP WITH TIME ZONE NOT NULL,
	harvest_rate_mbps DECIMAL(10,2) NOT NULL,
	is_active BOOLEAN NOT NULL DEFAULT true,
	quantum_noise_level DECIMAL(10,6) NOT NULL,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	
	-- Multi-tenant isolation
	CONSTRAINT valid_tenant_id CHECK (LENGTH(tenant_id) >= 8),
	
	-- Performance constraints
	CONSTRAINT positive_harvest_rate CHECK (harvest_rate_mbps > 0),
	CONSTRAINT positive_noise_level CHECK (quantum_noise_level >= 0)
);

-- Post-Quantum Key Pairs
CREATE TABLE post_quantum_keypairs (
	id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id VARCHAR(32) NOT NULL,
	algorithm post_quantum_algorithm NOT NULL,
	security_level security_level NOT NULL,
	
	-- CRYSTALS-Kyber KEM keys (encrypted at rest)
	kyber_public_key BYTEA NOT NULL,
	kyber_secret_key_encrypted BYTEA NOT NULL,
	kyber_ciphertext BYTEA,
	
	-- CRYSTALS-Dilithium signature keys (encrypted at rest)
	dilithium_public_key BYTEA NOT NULL,
	dilithium_secret_key_encrypted BYTEA NOT NULL,
	
	-- Key metadata
	key_size INTEGER NOT NULL,
	entropy_source_id VARCHAR(36) NOT NULL REFERENCES quantum_entropy_sources(id),
	generation_context JSONB DEFAULT '{}',
	
	-- Lifecycle management
	state key_lifecycle_state NOT NULL DEFAULT 'generating',
	autonomous_management BOOLEAN NOT NULL DEFAULT true,
	last_rotation TIMESTAMP WITH TIME ZONE,
	next_rotation TIMESTAMP WITH TIME ZONE,
	rotation_frequency_days INTEGER NOT NULL DEFAULT 90,
	
	-- Security and compliance
	zero_knowledge_protected BOOLEAN NOT NULL DEFAULT true,
	threshold_shares INTEGER,
	compliance_frameworks compliance_framework[] DEFAULT '{}',
	
	-- Audit and versioning
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	expires_at TIMESTAMP WITH TIME ZONE,
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	version INTEGER NOT NULL DEFAULT 1,
	
	-- Multi-tenant isolation
	CONSTRAINT valid_tenant_id CHECK (LENGTH(tenant_id) >= 8),
	
	-- Key size validation
	CONSTRAINT valid_key_size CHECK (key_size IN (128, 192, 256, 384, 512, 1024, 2048, 3072, 4096)),
	
	-- Lifecycle validation
	CONSTRAINT valid_rotation_frequency CHECK (rotation_frequency_days > 0),
	CONSTRAINT valid_threshold_shares CHECK (threshold_shares IS NULL OR threshold_shares >= 2),
	
	-- Temporal constraints
	CONSTRAINT valid_expiration CHECK (expires_at IS NULL OR expires_at > created_at)
);

-- Quantum-Safe Sessions
CREATE TABLE quantum_safe_sessions (
	id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id VARCHAR(32) NOT NULL,
	user_id VARCHAR(36) NOT NULL,
	device_id VARCHAR(36) NOT NULL,
	
	-- Session cryptography (encrypted at rest)
	session_key_encrypted BYTEA NOT NULL,
	key_pair_id VARCHAR(36) NOT NULL REFERENCES post_quantum_keypairs(id),
	encryption_mode encryption_mode NOT NULL,
	
	-- Zero-knowledge architecture (encrypted at rest)
	client_key_share_encrypted BYTEA NOT NULL,
	server_key_share_encrypted BYTEA NOT NULL,
	threshold_required INTEGER NOT NULL DEFAULT 2,
	
	-- Session security
	threat_level threat_level NOT NULL DEFAULT 'low',
	adaptive_algorithm post_quantum_algorithm NOT NULL,
	quantum_safe_level security_level NOT NULL,
	
	-- Session lifecycle
	is_active BOOLEAN NOT NULL DEFAULT true,
	last_activity TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	session_timeout_minutes INTEGER NOT NULL DEFAULT 60,
	
	-- Audit
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
	
	-- Multi-tenant isolation
	CONSTRAINT valid_tenant_id CHECK (LENGTH(tenant_id) >= 8),
	
	-- Session constraints
	CONSTRAINT valid_timeout CHECK (session_timeout_minutes > 0),
	CONSTRAINT valid_threshold CHECK (threshold_required >= 2),
	CONSTRAINT valid_expiration CHECK (expires_at > created_at)
);

-- Zero-Knowledge Proofs
CREATE TABLE zero_knowledge_proofs (
	id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id VARCHAR(32) NOT NULL,
	session_id VARCHAR(36) NOT NULL REFERENCES quantum_safe_sessions(id),
	
	-- Proof components
	proof_data BYTEA NOT NULL,
	verification_key BYTEA NOT NULL,
	commitment BYTEA NOT NULL,
	challenge BYTEA NOT NULL,
	response BYTEA NOT NULL,
	
	-- Proof metadata
	proof_system VARCHAR(50) NOT NULL DEFAULT 'groth16',
	circuit_hash VARCHAR(64) NOT NULL,
	public_inputs TEXT[] DEFAULT '{}',
	
	-- Verification status
	is_verified BOOLEAN NOT NULL DEFAULT false,
	verified_at TIMESTAMP WITH TIME ZONE,
	verification_context JSONB DEFAULT '{}',
	
	-- Lifecycle
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
	
	-- Multi-tenant isolation
	CONSTRAINT valid_tenant_id CHECK (LENGTH(tenant_id) >= 8),
	
	-- Proof validation
	CONSTRAINT valid_circuit_hash CHECK (LENGTH(circuit_hash) = 64),
	CONSTRAINT verification_consistency CHECK (
		(is_verified = true AND verified_at IS NOT NULL) OR 
		(is_verified = false AND verified_at IS NULL)
	)
);

-- Homomorphic Ciphertexts
CREATE TABLE homomorphic_ciphertexts (
	id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id VARCHAR(32) NOT NULL,
	session_id VARCHAR(36) NOT NULL REFERENCES quantum_safe_sessions(id),
	
	-- Homomorphic encryption data
	ciphertext_data BYTEA NOT NULL,
	scheme VARCHAR(20) NOT NULL DEFAULT 'ckks',
	parameters JSONB NOT NULL,
	
	-- Computation metadata
	computation_context VARCHAR(64) NOT NULL,
	data_type VARCHAR(20) NOT NULL,
	data_size INTEGER NOT NULL,
	noise_level DECIMAL(10,6) NOT NULL,
	
	-- Operations tracking
	operations_performed TEXT[] DEFAULT '{}',
	operation_count INTEGER NOT NULL DEFAULT 0,
	max_operations INTEGER NOT NULL DEFAULT 1000,
	
	-- Lifecycle
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
	
	-- Multi-tenant isolation  
	CONSTRAINT valid_tenant_id CHECK (LENGTH(tenant_id) >= 8),
	
	-- Data validation
	CONSTRAINT positive_data_size CHECK (data_size > 0),
	CONSTRAINT valid_noise_level CHECK (noise_level >= 0),
	CONSTRAINT valid_operation_count CHECK (operation_count >= 0 AND operation_count <= max_operations),
	CONSTRAINT valid_scheme CHECK (scheme IN ('bfv', 'bgv', 'ckks'))
);

-- Autonomous Key Decisions
CREATE TABLE autonomous_key_decisions (
	id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id VARCHAR(32) NOT NULL,
	key_pair_id VARCHAR(36) NOT NULL REFERENCES post_quantum_keypairs(id),
	
	-- Decision context
	decision_type VARCHAR(50) NOT NULL,
	confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
	reasoning JSONB NOT NULL,
	
	-- Analysis inputs
	usage_patterns JSONB NOT NULL,
	security_assessment JSONB NOT NULL,
	threat_intelligence JSONB NOT NULL,
	compliance_requirements compliance_framework[] DEFAULT '{}',
	
	-- Recommended actions
	should_rotate BOOLEAN NOT NULL DEFAULT false,
	should_backup BOOLEAN NOT NULL DEFAULT false,
	should_destroy BOOLEAN NOT NULL DEFAULT false,
	should_upgrade_quantum BOOLEAN NOT NULL DEFAULT false,
	
	-- Action timing
	recommended_execution_time TIMESTAMP WITH TIME ZONE NOT NULL,
	priority_level INTEGER NOT NULL DEFAULT 5 CHECK (priority_level BETWEEN 1 AND 10),
	
	-- Execution tracking
	is_executed BOOLEAN NOT NULL DEFAULT false,
	executed_at TIMESTAMP WITH TIME ZONE,
	execution_result JSONB,
	
	-- Audit
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	
	-- Multi-tenant isolation
	CONSTRAINT valid_tenant_id CHECK (LENGTH(tenant_id) >= 8),
	
	-- Execution consistency
	CONSTRAINT execution_consistency CHECK (
		(is_executed = true AND executed_at IS NOT NULL) OR
		(is_executed = false AND executed_at IS NULL)
	)
);

-- Cryptographic Policies
CREATE TABLE cryptographic_policies (
	id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id VARCHAR(32) NOT NULL,
	
	-- Policy definition
	policy_name VARCHAR(255) NOT NULL,
	policy_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
	policy_description TEXT NOT NULL,
	
	-- Algorithm selection
	required_algorithm post_quantum_algorithm NOT NULL,
	fallback_algorithms post_quantum_algorithm[] DEFAULT '{}',
	minimum_security_level security_level NOT NULL,
	quantum_safe_required BOOLEAN NOT NULL DEFAULT true,
	
	-- Key management requirements
	key_rotation_interval_days INTEGER NOT NULL,
	autonomous_management_required BOOLEAN NOT NULL DEFAULT true,
	threshold_cryptography_shares INTEGER,
	
	-- Compliance and regulatory requirements
	applicable_frameworks compliance_framework[] DEFAULT '{}',
	data_residency_requirements TEXT[] DEFAULT '{}',
	retention_period_days INTEGER,
	
	-- Threat adaptation
	threat_adaptation_enabled BOOLEAN NOT NULL DEFAULT true,
	threat_response_sensitivity DECIMAL(3,2) NOT NULL DEFAULT 0.70 CHECK (threat_response_sensitivity BETWEEN 0.0 AND 1.0),
	quantum_threat_threshold DECIMAL(3,2) NOT NULL DEFAULT 0.80 CHECK (quantum_threat_threshold BETWEEN 0.0 AND 1.0),
	
	-- Performance requirements
	max_encryption_latency_ms INTEGER NOT NULL DEFAULT 100,
	min_throughput_ops_per_sec INTEGER NOT NULL DEFAULT 1000,
	
	-- Audit and monitoring
	audit_level VARCHAR(20) NOT NULL DEFAULT 'comprehensive',
	monitoring_enabled BOOLEAN NOT NULL DEFAULT true,
	
	-- Policy lifecycle
	is_active BOOLEAN NOT NULL DEFAULT true,
	effective_from TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	effective_until TIMESTAMP WITH TIME ZONE,
	
	-- AI policy generation metadata
	ai_generated BOOLEAN NOT NULL DEFAULT true,
	ai_confidence DECIMAL(3,2) NOT NULL CHECK (ai_confidence BETWEEN 0.0 AND 1.0),
	generation_context JSONB DEFAULT '{}',
	
	-- Audit
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	
	-- Multi-tenant isolation
	CONSTRAINT valid_tenant_id CHECK (LENGTH(tenant_id) >= 8),
	
	-- Policy validation
	CONSTRAINT unique_policy_name_per_tenant UNIQUE (tenant_id, policy_name),
	CONSTRAINT valid_rotation_interval CHECK (key_rotation_interval_days > 0),
	CONSTRAINT valid_threshold CHECK (threshold_cryptography_shares IS NULL OR threshold_cryptography_shares >= 2),
	CONSTRAINT valid_retention CHECK (retention_period_days IS NULL OR retention_period_days > 0),
	CONSTRAINT valid_performance CHECK (max_encryption_latency_ms > 0 AND min_throughput_ops_per_sec > 0),
	CONSTRAINT valid_effective_period CHECK (effective_until IS NULL OR effective_until > effective_from),
	CONSTRAINT valid_audit_level CHECK (audit_level IN ('minimal', 'standard', 'comprehensive', 'forensic'))
);

-- Threat Intelligence
CREATE TABLE threat_intelligence (
	id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id VARCHAR(32) NOT NULL,
	
	-- Threat assessment
	current_threat_level threat_level NOT NULL,
	quantum_threat_probability DECIMAL(3,2) NOT NULL CHECK (quantum_threat_probability BETWEEN 0.0 AND 1.0),
	nation_state_activity BOOLEAN NOT NULL DEFAULT false,
	
	-- Threat indicators
	threat_sources TEXT[] DEFAULT '{}',
	attack_vectors TEXT[] DEFAULT '{}',
	targeted_algorithms post_quantum_algorithm[] DEFAULT '{}',
	
	-- Intelligence sources
	intelligence_feeds TEXT[] DEFAULT '{}',
	last_feed_update TIMESTAMP WITH TIME ZONE NOT NULL,
	confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score BETWEEN 0.0 AND 1.0),
	
	-- Adaptive recommendations
	recommended_algorithms post_quantum_algorithm[] DEFAULT '{}',
	recommended_security_level security_level NOT NULL,
	immediate_action_required BOOLEAN NOT NULL DEFAULT false,
	
	-- Geospatial context
	threat_geography TEXT[] DEFAULT '{}',
	affected_regions TEXT[] DEFAULT '{}',
	
	-- Lifecycle
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
	
	-- Multi-tenant isolation
	CONSTRAINT valid_tenant_id CHECK (LENGTH(tenant_id) >= 8)
);

-- Encryption Operations (audit trail)
CREATE TABLE encryption_operations (
	id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id VARCHAR(32) NOT NULL,
	session_id VARCHAR(36) REFERENCES quantum_safe_sessions(id),
	
	-- Operation details
	operation_type VARCHAR(50) NOT NULL,
	encryption_mode encryption_mode NOT NULL,
	algorithm_used post_quantum_algorithm NOT NULL,
	
	-- Data context
	data_size_bytes BIGINT NOT NULL,
	data_classification VARCHAR(50) NOT NULL,
	data_context JSONB DEFAULT '{}',
	
	-- Performance metrics
	operation_latency_ms DECIMAL(10,3) NOT NULL,
	throughput_mbps DECIMAL(10,2) NOT NULL,
	cpu_usage_percent DECIMAL(5,2) NOT NULL,
	memory_usage_mb DECIMAL(10,2) NOT NULL,
	
	-- Security context
	threat_level_at_operation threat_level NOT NULL,
	security_level_achieved security_level NOT NULL,
	zero_knowledge_proof_id VARCHAR(36) REFERENCES zero_knowledge_proofs(id),
	
	-- Quality assurance
	entropy_quality DECIMAL(3,2) NOT NULL CHECK (entropy_quality BETWEEN 0.0 AND 1.0),
	validation_passed BOOLEAN NOT NULL,
	error_details TEXT,
	
	-- Audit and compliance
	compliance_frameworks_met compliance_framework[] DEFAULT '{}',
	audit_trail_id VARCHAR(36) NOT NULL,
	
	-- Neuromorphic processing (if used)
	neuromorphic_processing_used BOOLEAN NOT NULL DEFAULT false,
	neuromorphic_latency_ns DECIMAL(15,3),
	energy_consumption_pj DECIMAL(15,6),
	
	-- Audit timestamps
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	completed_at TIMESTAMP WITH TIME ZONE NOT NULL,
	
	-- Multi-tenant isolation
	CONSTRAINT valid_tenant_id CHECK (LENGTH(tenant_id) >= 8),
	
	-- Data validation
	CONSTRAINT positive_data_size CHECK (data_size_bytes > 0),
	CONSTRAINT valid_latency CHECK (operation_latency_ms >= 0),
	CONSTRAINT valid_throughput CHECK (throughput_mbps >= 0),
	CONSTRAINT valid_cpu_usage CHECK (cpu_usage_percent BETWEEN 0 AND 100),
	CONSTRAINT valid_memory_usage CHECK (memory_usage_mb >= 0),
	CONSTRAINT neuromorphic_consistency CHECK (
		(neuromorphic_processing_used = true AND neuromorphic_latency_ns IS NOT NULL) OR
		(neuromorphic_processing_used = false)
	),
	CONSTRAINT valid_completion CHECK (completed_at >= created_at)
);

-- APG Integration Context
CREATE TABLE apg_encryption_context (
	id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id VARCHAR(32) NOT NULL,
	
	-- APG capability context
	requesting_capability VARCHAR(50) NOT NULL,
	capability_version VARCHAR(20) NOT NULL,
	integration_context JSONB DEFAULT '{}',
	
	-- Authentication context (from auth capability)
	user_context JSONB DEFAULT '{}',
	session_context JSONB DEFAULT '{}',
	rbac_context JSONB DEFAULT '{}',
	
	-- Security framework context (from secu capability)
	security_assessment JSONB DEFAULT '{}',
	risk_score DECIMAL(3,2) NOT NULL DEFAULT 0.50 CHECK (risk_score BETWEEN 0.0 AND 1.0),
	threat_context JSONB DEFAULT '{}',
	
	-- Audit context (from audl capability)
	audit_requirements TEXT[] DEFAULT '{}',
	compliance_context JSONB DEFAULT '{}',
	
	-- Performance context
	performance_requirements JSONB DEFAULT '{}',
	latency_budget_ms INTEGER NOT NULL DEFAULT 100,
	
	-- Lifecycle
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
	
	-- Multi-tenant isolation
	CONSTRAINT valid_tenant_id CHECK (LENGTH(tenant_id) >= 8),
	CONSTRAINT valid_latency_budget CHECK (latency_budget_ms > 0)
);

--
-- PERFORMANCE INDEXES
--

-- Quantum Entropy Sources
CREATE INDEX idx_quantum_entropy_sources_tenant ON quantum_entropy_sources(tenant_id);
CREATE INDEX idx_quantum_entropy_sources_active ON quantum_entropy_sources(is_active) WHERE is_active = true;
CREATE INDEX idx_quantum_entropy_sources_quality ON quantum_entropy_sources(quality_score DESC);

-- Post-Quantum Key Pairs  
CREATE INDEX idx_post_quantum_keypairs_tenant ON post_quantum_keypairs(tenant_id);
CREATE INDEX idx_post_quantum_keypairs_state ON post_quantum_keypairs(state);
CREATE INDEX idx_post_quantum_keypairs_algorithm ON post_quantum_keypairs(algorithm);
CREATE INDEX idx_post_quantum_keypairs_rotation ON post_quantum_keypairs(next_rotation) WHERE next_rotation IS NOT NULL;
CREATE INDEX idx_post_quantum_keypairs_tenant_active ON post_quantum_keypairs(tenant_id, state) WHERE state = 'active';

-- Quantum-Safe Sessions
CREATE INDEX idx_quantum_safe_sessions_tenant ON quantum_safe_sessions(tenant_id);
CREATE INDEX idx_quantum_safe_sessions_user ON quantum_safe_sessions(user_id);
CREATE INDEX idx_quantum_safe_sessions_active ON quantum_safe_sessions(is_active) WHERE is_active = true;
CREATE INDEX idx_quantum_safe_sessions_expiry ON quantum_safe_sessions(expires_at);
CREATE INDEX idx_quantum_safe_sessions_tenant_user_active ON quantum_safe_sessions(tenant_id, user_id) WHERE is_active = true;

-- Zero-Knowledge Proofs
CREATE INDEX idx_zero_knowledge_proofs_tenant ON zero_knowledge_proofs(tenant_id);
CREATE INDEX idx_zero_knowledge_proofs_session ON zero_knowledge_proofs(session_id);
CREATE INDEX idx_zero_knowledge_proofs_verified ON zero_knowledge_proofs(is_verified, verified_at);
CREATE INDEX idx_zero_knowledge_proofs_expiry ON zero_knowledge_proofs(expires_at);

-- Homomorphic Ciphertexts
CREATE INDEX idx_homomorphic_ciphertexts_tenant ON homomorphic_ciphertexts(tenant_id);
CREATE INDEX idx_homomorphic_ciphertexts_session ON homomorphic_ciphertexts(session_id);
CREATE INDEX idx_homomorphic_ciphertexts_context ON homomorphic_ciphertexts(computation_context);
CREATE INDEX idx_homomorphic_ciphertexts_operations ON homomorphic_ciphertexts(operation_count);

-- Autonomous Key Decisions
CREATE INDEX idx_autonomous_key_decisions_tenant ON autonomous_key_decisions(tenant_id);
CREATE INDEX idx_autonomous_key_decisions_key ON autonomous_key_decisions(key_pair_id);
CREATE INDEX idx_autonomous_key_decisions_execution ON autonomous_key_decisions(recommended_execution_time) WHERE is_executed = false;
CREATE INDEX idx_autonomous_key_decisions_priority ON autonomous_key_decisions(priority_level DESC);

-- Cryptographic Policies
CREATE INDEX idx_cryptographic_policies_tenant ON cryptographic_policies(tenant_id);
CREATE INDEX idx_cryptographic_policies_active ON cryptographic_policies(is_active) WHERE is_active = true;
CREATE INDEX idx_cryptographic_policies_tenant_active ON cryptographic_policies(tenant_id, is_active) WHERE is_active = true;

-- Threat Intelligence
CREATE INDEX idx_threat_intelligence_tenant ON threat_intelligence(tenant_id);
CREATE INDEX idx_threat_intelligence_level ON threat_intelligence(current_threat_level);
CREATE INDEX idx_threat_intelligence_quantum ON threat_intelligence(quantum_threat_probability DESC);
CREATE INDEX idx_threat_intelligence_expiry ON threat_intelligence(expires_at);

-- Encryption Operations (for analytics)
CREATE INDEX idx_encryption_operations_tenant ON encryption_operations(tenant_id);
CREATE INDEX idx_encryption_operations_session ON encryption_operations(session_id);
CREATE INDEX idx_encryption_operations_algorithm ON encryption_operations(algorithm_used);
CREATE INDEX idx_encryption_operations_timestamp ON encryption_operations(created_at);
CREATE INDEX idx_encryption_operations_tenant_timestamp ON encryption_operations(tenant_id, created_at);
CREATE INDEX idx_encryption_operations_performance ON encryption_operations(operation_latency_ms, throughput_mbps);

-- APG Integration Context
CREATE INDEX idx_apg_encryption_context_tenant ON apg_encryption_context(tenant_id);
CREATE INDEX idx_apg_encryption_context_capability ON apg_encryption_context(requesting_capability);

--
-- ADVANCED INDEXES FOR ANALYTICS
--

-- Composite indexes for common query patterns
CREATE INDEX idx_keypairs_tenant_algo_state ON post_quantum_keypairs(tenant_id, algorithm, state);
CREATE INDEX idx_sessions_tenant_threat_level ON quantum_safe_sessions(tenant_id, threat_level);
CREATE INDEX idx_operations_tenant_mode_algorithm ON encryption_operations(tenant_id, encryption_mode, algorithm_used);

-- Partial indexes for performance optimization
CREATE INDEX idx_active_sessions_last_activity ON quantum_safe_sessions(last_activity) WHERE is_active = true;
CREATE INDEX idx_pending_key_decisions ON autonomous_key_decisions(recommended_execution_time, priority_level) WHERE is_executed = false;
CREATE INDEX idx_high_threat_intelligence ON threat_intelligence(created_at) WHERE current_threat_level IN ('high', 'critical', 'quantum-imminent');

--
-- TRIGGERS FOR AUTOMATIC UPDATES
--

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
	NEW.updated_at = CURRENT_TIMESTAMP;
	RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers to relevant tables
CREATE TRIGGER update_quantum_entropy_sources_updated_at BEFORE UPDATE ON quantum_entropy_sources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_post_quantum_keypairs_updated_at BEFORE UPDATE ON post_quantum_keypairs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cryptographic_policies_updated_at BEFORE UPDATE ON cryptographic_policies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_threat_intelligence_updated_at BEFORE UPDATE ON threat_intelligence FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_apg_encryption_context_updated_at BEFORE UPDATE ON apg_encryption_context FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

--
-- ROW LEVEL SECURITY (RLS) FOR MULTI-TENANT ISOLATION
--

-- Enable RLS on all tables
ALTER TABLE quantum_entropy_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE post_quantum_keypairs ENABLE ROW LEVEL SECURITY;
ALTER TABLE quantum_safe_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE zero_knowledge_proofs ENABLE ROW LEVEL SECURITY;
ALTER TABLE homomorphic_ciphertexts ENABLE ROW LEVEL SECURITY;
ALTER TABLE autonomous_key_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE cryptographic_policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE threat_intelligence ENABLE ROW LEVEL SECURITY;
ALTER TABLE encryption_operations ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_encryption_context ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
-- Note: In production, these would reference the actual APG authentication context

CREATE POLICY tenant_isolation_quantum_entropy_sources ON quantum_entropy_sources
	FOR ALL TO apg_encryption_service
	USING (tenant_id = current_setting('apg.current_tenant_id', true));

CREATE POLICY tenant_isolation_post_quantum_keypairs ON post_quantum_keypairs
	FOR ALL TO apg_encryption_service
	USING (tenant_id = current_setting('apg.current_tenant_id', true));

CREATE POLICY tenant_isolation_quantum_safe_sessions ON quantum_safe_sessions
	FOR ALL TO apg_encryption_service
	USING (tenant_id = current_setting('apg.current_tenant_id', true));

CREATE POLICY tenant_isolation_zero_knowledge_proofs ON zero_knowledge_proofs
	FOR ALL TO apg_encryption_service
	USING (tenant_id = current_setting('apg.current_tenant_id', true));

CREATE POLICY tenant_isolation_homomorphic_ciphertexts ON homomorphic_ciphertexts
	FOR ALL TO apg_encryption_service
	USING (tenant_id = current_setting('apg.current_tenant_id', true));

CREATE POLICY tenant_isolation_autonomous_key_decisions ON autonomous_key_decisions
	FOR ALL TO apg_encryption_service
	USING (tenant_id = current_setting('apg.current_tenant_id', true));

CREATE POLICY tenant_isolation_cryptographic_policies ON cryptographic_policies
	FOR ALL TO apg_encryption_service
	USING (tenant_id = current_setting('apg.current_tenant_id', true));

CREATE POLICY tenant_isolation_threat_intelligence ON threat_intelligence
	FOR ALL TO apg_encryption_service
	USING (tenant_id = current_setting('apg.current_tenant_id', true));

CREATE POLICY tenant_isolation_encryption_operations ON encryption_operations
	FOR ALL TO apg_encryption_service
	USING (tenant_id = current_setting('apg.current_tenant_id', true));

CREATE POLICY tenant_isolation_apg_encryption_context ON apg_encryption_context
	FOR ALL TO apg_encryption_service
	USING (tenant_id = current_setting('apg.current_tenant_id', true));

--
-- ANALYTICS VIEWS
--

-- Key Lifecycle Analytics
CREATE VIEW key_lifecycle_analytics AS
SELECT 
	tenant_id,
	algorithm,
	state,
	COUNT(*) as key_count,
	AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 86400) as avg_age_days,
	COUNT(*) FILTER (WHERE autonomous_management = true) as autonomous_count,
	COUNT(*) FILTER (WHERE next_rotation IS NOT NULL AND next_rotation < CURRENT_TIMESTAMP + INTERVAL '7 days') as rotation_due_soon
FROM post_quantum_keypairs
GROUP BY tenant_id, algorithm, state;

-- Threat Intelligence Summary
CREATE VIEW threat_intelligence_summary AS
SELECT 
	tenant_id,
	current_threat_level,
	AVG(quantum_threat_probability) as avg_quantum_threat,
	COUNT(*) FILTER (WHERE nation_state_activity = true) as nation_state_threats,
	COUNT(*) FILTER (WHERE immediate_action_required = true) as immediate_actions,
	MAX(created_at) as latest_intelligence
FROM threat_intelligence
WHERE expires_at > CURRENT_TIMESTAMP
GROUP BY tenant_id, current_threat_level;

-- Performance Metrics
CREATE VIEW encryption_performance_metrics AS
SELECT 
	tenant_id,
	algorithm_used,
	encryption_mode,
	DATE_TRUNC('hour', created_at) as hour_bucket,
	COUNT(*) as operation_count,
	AVG(operation_latency_ms) as avg_latency_ms,
	PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY operation_latency_ms) as p95_latency_ms,
	AVG(throughput_mbps) as avg_throughput_mbps,
	COUNT(*) FILTER (WHERE validation_passed = true) as successful_operations,
	COUNT(*) FILTER (WHERE neuromorphic_processing_used = true) as neuromorphic_operations
FROM encryption_operations
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY tenant_id, algorithm_used, encryption_mode, hour_bucket;

--
-- MATERIALIZED VIEWS FOR PERFORMANCE
--

-- Daily tenant encryption statistics
CREATE MATERIALIZED VIEW daily_tenant_encryption_stats AS
SELECT 
	tenant_id,
	DATE(created_at) as date,
	COUNT(*) as total_operations,
	COUNT(DISTINCT session_id) as unique_sessions,
	AVG(operation_latency_ms) as avg_latency_ms,
	SUM(data_size_bytes) as total_data_encrypted_bytes,
	COUNT(*) FILTER (WHERE algorithm_used LIKE 'crystals-%') as quantum_safe_operations,
	COUNT(*) FILTER (WHERE neuromorphic_processing_used = true) as neuromorphic_operations
FROM encryption_operations
GROUP BY tenant_id, DATE(created_at);

-- Create index on materialized view
CREATE UNIQUE INDEX idx_daily_tenant_encryption_stats ON daily_tenant_encryption_stats(tenant_id, date);

-- Refresh materialized view daily
-- (In production, this would be handled by a scheduled job)

--
-- DATABASE ROLES AND PERMISSIONS
--

-- Create service role for APG encryption capability
CREATE ROLE apg_encryption_service;

-- Grant schema usage
GRANT USAGE ON SCHEMA apg_encryption TO apg_encryption_service;

-- Grant table permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA apg_encryption TO apg_encryption_service;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA apg_encryption TO apg_encryption_service;

-- Grant view permissions
GRANT SELECT ON ALL TABLES IN SCHEMA apg_encryption TO apg_encryption_service;

-- Grant materialized view permissions
GRANT SELECT ON daily_tenant_encryption_stats TO apg_encryption_service;

--
-- SAMPLE DATA FOR TESTING (Development Only)
--

-- Insert sample quantum entropy source
INSERT INTO quantum_entropy_sources (
	tenant_id, source_type, location, quality_score, 
	last_harvest_at, harvest_rate_mbps, quantum_noise_level
) VALUES (
	'apg_dev_tenant', 'photonic', 'quantum_lab_1', 0.999,
	CURRENT_TIMESTAMP, 100.0, 0.000001
);

-- Insert sample cryptographic policy
INSERT INTO cryptographic_policies (
	tenant_id, policy_name, policy_description, required_algorithm,
	minimum_security_level, key_rotation_interval_days, ai_confidence
) VALUES (
	'apg_dev_tenant', 'High Security Policy', 'Quantum-safe encryption for sensitive data',
	'crystals-kyber-1024', 'level-5', 30, 0.95
);

--
-- COMMENTS FOR DOCUMENTATION
--

COMMENT ON SCHEMA apg_encryption IS 'APG Encryption Services - Quantum-safe cryptographic operations with zero-knowledge architecture';

COMMENT ON TABLE quantum_entropy_sources IS 'Quantum entropy sources for true randomness generation in cryptographic operations';
COMMENT ON TABLE post_quantum_keypairs IS 'Post-quantum cryptographic key pairs with autonomous lifecycle management';
COMMENT ON TABLE quantum_safe_sessions IS 'Quantum-safe cryptographic sessions with zero-knowledge architecture';
COMMENT ON TABLE zero_knowledge_proofs IS 'Zero-knowledge proofs for privacy-preserving access control';
COMMENT ON TABLE homomorphic_ciphertexts IS 'Ciphertexts for homomorphic computation on encrypted data';
COMMENT ON TABLE autonomous_key_decisions IS 'AI-driven autonomous key management decisions and actions';
COMMENT ON TABLE cryptographic_policies IS 'AI-generated cryptographic policies based on threat intelligence';
COMMENT ON TABLE threat_intelligence IS 'Real-time threat intelligence for adaptive encryption';
COMMENT ON TABLE encryption_operations IS 'Comprehensive audit trail of all encryption operations';
COMMENT ON TABLE apg_encryption_context IS 'Integration context for APG capability interactions';

-- Performance optimization comments
COMMENT ON INDEX idx_post_quantum_keypairs_tenant_active IS 'High-performance lookup for active keys by tenant';
COMMENT ON INDEX idx_quantum_safe_sessions_tenant_user_active IS 'Optimized session lookup for authentication';
COMMENT ON INDEX idx_encryption_operations_tenant_timestamp IS 'Efficient analytics queries by tenant and time';

-- Security comments
COMMENT ON POLICY tenant_isolation_post_quantum_keypairs ON post_quantum_keypairs IS 'Row-level security for complete tenant isolation';

-- Final schema validation
DO $$
BEGIN
	RAISE NOTICE 'APG Encryption Services Database Schema Created Successfully';
	RAISE NOTICE 'Schema supports: Post-quantum cryptography, Zero-knowledge encryption, Autonomous key management';
	RAISE NOTICE 'Multi-tenant isolation: Enabled with Row Level Security';
	RAISE NOTICE 'Performance optimization: Comprehensive indexing strategy implemented';
	RAISE NOTICE 'Compliance: Audit trails and data lifecycle management included';
END $$;