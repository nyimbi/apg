-- APG Crawler Capability Database Schema
-- Multi-tenant enterprise web intelligence platform
-- Copyright © 2025 Datacraft (nyimbi@gmail.com)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "vector"; -- For vector embeddings (pgvector)
CREATE EXTENSION IF NOT EXISTS "hstore"; -- For key-value storage

-- Create schema for crawler capability
CREATE SCHEMA IF NOT EXISTS crawler;

-- Set search path
SET search_path TO crawler, public;

-- =====================================================
-- CORE CRAWLER MANAGEMENT TABLES
-- =====================================================

-- Crawl targets define what to crawl and how
CREATE TABLE cr_crawl_targets (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	name VARCHAR(500) NOT NULL,
	description TEXT,
	target_urls TEXT[] NOT NULL DEFAULT '{}',
	target_type VARCHAR(50) NOT NULL DEFAULT 'web_crawl',
	data_schema JSONB,
	business_context JSONB NOT NULL DEFAULT '{}',
	stealth_requirements JSONB NOT NULL DEFAULT '{}',
	quality_requirements JSONB NOT NULL DEFAULT '{}',
	scheduling_config JSONB NOT NULL DEFAULT '{}',
	collaboration_config JSONB NOT NULL DEFAULT '{}',
	-- RAG/GraphRAG Integration Fields
	rag_integration_enabled BOOLEAN NOT NULL DEFAULT true,
	graphrag_integration_enabled BOOLEAN NOT NULL DEFAULT false,
	knowledge_graph_target VARCHAR(255),
	content_fingerprinting BOOLEAN NOT NULL DEFAULT true,
	markdown_storage BOOLEAN NOT NULL DEFAULT true,
	status VARCHAR(50) NOT NULL DEFAULT 'active',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255),
	updated_by VARCHAR(255)
);

-- Crawl pipelines define processing workflows
CREATE TABLE cr_crawl_pipelines (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	name VARCHAR(500) NOT NULL,
	description TEXT,
	visual_config JSONB NOT NULL DEFAULT '{}',
	processing_stages JSONB NOT NULL DEFAULT '[]',
	optimization_settings JSONB NOT NULL DEFAULT '{}',
	performance_metrics JSONB NOT NULL DEFAULT '{}',
	deployment_status VARCHAR(50) NOT NULL DEFAULT 'draft',
	monitoring_config JSONB NOT NULL DEFAULT '{}',
	target_id UUID REFERENCES cr_crawl_targets(id) ON DELETE CASCADE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255),
	updated_by VARCHAR(255)
);

-- Data source configurations
CREATE TABLE cr_data_sources (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	name VARCHAR(500) NOT NULL,
	source_type VARCHAR(100) NOT NULL, -- web, api, database, social_media, news, etc.
	connection_config JSONB NOT NULL DEFAULT '{}',
	authentication_config JSONB NOT NULL DEFAULT '{}',
	rate_limits JSONB NOT NULL DEFAULT '{}',
	stealth_config JSONB NOT NULL DEFAULT '{}',
	capabilities JSONB NOT NULL DEFAULT '{}',
	health_status VARCHAR(50) NOT NULL DEFAULT 'unknown',
	last_health_check TIMESTAMP WITH TIME ZONE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255),
	updated_by VARCHAR(255)
);

-- =====================================================
-- DATA EXTRACTION & PROCESSING TABLES
-- =====================================================

-- Extracted datasets from crawling operations
CREATE TABLE cr_extracted_datasets (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	crawl_target_id UUID NOT NULL REFERENCES cr_crawl_targets(id) ON DELETE CASCADE,
	pipeline_id UUID REFERENCES cr_crawl_pipelines(id) ON DELETE SET NULL,
	dataset_name VARCHAR(500) NOT NULL,
	extraction_method VARCHAR(100) NOT NULL,
	source_urls TEXT[] NOT NULL DEFAULT '{}',
	record_count INTEGER NOT NULL DEFAULT 0,
	quality_metrics JSONB NOT NULL DEFAULT '{}',
	validation_status VARCHAR(50) NOT NULL DEFAULT 'pending',
	consensus_score DECIMAL(5,4) DEFAULT 0.0,
	data_schema JSONB,
	metadata JSONB NOT NULL DEFAULT '{}',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255),
	updated_by VARCHAR(255)
);

-- Individual data records within datasets
CREATE TABLE cr_data_records (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	dataset_id UUID NOT NULL REFERENCES cr_extracted_datasets(id) ON DELETE CASCADE,
	record_index INTEGER NOT NULL,
	source_url TEXT NOT NULL,
	extracted_data JSONB NOT NULL DEFAULT '{}',
	raw_content TEXT,
	processed_content TEXT,
	-- RAG/GraphRAG Content Fields
	cleaned_content TEXT,
	markdown_content TEXT,
	content_fingerprint VARCHAR(64) DEFAULT '',
	content_processing_stage VARCHAR(50) NOT NULL DEFAULT 'raw_extracted',
	vector_embeddings vector(1536), -- Vector embeddings for RAG
	rag_chunk_ids TEXT[] DEFAULT '{}',
	graphrag_node_id VARCHAR(255),
	knowledge_graph_entities TEXT[] DEFAULT '{}',
	content_type VARCHAR(100),
	language VARCHAR(10),
	quality_score DECIMAL(5,4) DEFAULT 0.0,
	confidence_score DECIMAL(5,4) DEFAULT 0.0,
	validation_status VARCHAR(50) NOT NULL DEFAULT 'pending',
	business_entities JSONB NOT NULL DEFAULT '[]',
	semantic_tags JSONB NOT NULL DEFAULT '[]',
	extraction_metadata JSONB NOT NULL DEFAULT '{}',
	rag_metadata JSONB NOT NULL DEFAULT '{}',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Business entities extracted from content
CREATE TABLE cr_business_entities (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	record_id UUID NOT NULL REFERENCES cr_data_records(id) ON DELETE CASCADE,
	entity_type VARCHAR(100) NOT NULL,
	entity_name VARCHAR(1000) NOT NULL,
	entity_value TEXT NOT NULL,
	confidence_score DECIMAL(5,4) NOT NULL,
	context_window TEXT,
	start_position INTEGER,
	end_position INTEGER,
	semantic_properties JSONB NOT NULL DEFAULT '{}',
	business_relevance DECIMAL(5,4) DEFAULT 0.0,
	validation_status VARCHAR(50) NOT NULL DEFAULT 'pending',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- COLLABORATIVE VALIDATION TABLES
-- =====================================================

-- Validation sessions for collaborative review
CREATE TABLE cr_validation_sessions (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	dataset_id UUID NOT NULL REFERENCES cr_extracted_datasets(id) ON DELETE CASCADE,
	session_name VARCHAR(500) NOT NULL,
	description TEXT,
	validation_schema JSONB NOT NULL DEFAULT '{}',
	session_status VARCHAR(50) NOT NULL DEFAULT 'active',
	consensus_threshold DECIMAL(5,4) NOT NULL DEFAULT 0.8,
	quality_threshold DECIMAL(5,4) NOT NULL DEFAULT 0.7,
	validator_count INTEGER NOT NULL DEFAULT 0,
	completion_percentage DECIMAL(5,2) NOT NULL DEFAULT 0.0,
	consensus_metrics JSONB NOT NULL DEFAULT '{}',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255),
	updated_by VARCHAR(255)
);

-- Validator profiles and assignments
CREATE TABLE cr_validators (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	session_id UUID NOT NULL REFERENCES cr_validation_sessions(id) ON DELETE CASCADE,
	user_id VARCHAR(255) NOT NULL,
	validator_name VARCHAR(500) NOT NULL,
	validator_role VARCHAR(100) NOT NULL,
	expertise_areas JSONB NOT NULL DEFAULT '[]',
	validation_permissions JSONB NOT NULL DEFAULT '{}',
	assignment_status VARCHAR(50) NOT NULL DEFAULT 'active',
	validation_stats JSONB NOT NULL DEFAULT '{}',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Validation feedback and annotations
CREATE TABLE cr_validation_feedback (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	session_id UUID NOT NULL REFERENCES cr_validation_sessions(id) ON DELETE CASCADE,
	validator_id UUID NOT NULL REFERENCES cr_validators(id) ON DELETE CASCADE,
	record_id UUID NOT NULL REFERENCES cr_data_records(id) ON DELETE CASCADE,
	feedback_type VARCHAR(50) NOT NULL, -- approve, reject, modify, comment
	quality_rating INTEGER CHECK (quality_rating >= 1 AND quality_rating <= 5),
	accuracy_rating INTEGER CHECK (accuracy_rating >= 1 AND accuracy_rating <= 5),
	completeness_rating INTEGER CHECK (completeness_rating >= 1 AND completeness_rating <= 5),
	suggested_changes JSONB,
	comments TEXT,
	validation_tags JSONB NOT NULL DEFAULT '[]',
	confidence_level DECIMAL(5,4) NOT NULL DEFAULT 1.0,
	processing_time_seconds INTEGER,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Consensus tracking for validated data
CREATE TABLE cr_validation_consensus (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	session_id UUID NOT NULL REFERENCES cr_validation_sessions(id) ON DELETE CASCADE,
	record_id UUID NOT NULL REFERENCES cr_data_records(id) ON DELETE CASCADE,
	total_validators INTEGER NOT NULL DEFAULT 0,
	approval_count INTEGER NOT NULL DEFAULT 0,
	rejection_count INTEGER NOT NULL DEFAULT 0,
	modification_count INTEGER NOT NULL DEFAULT 0,
	consensus_score DECIMAL(5,4) NOT NULL DEFAULT 0.0,
	consensus_status VARCHAR(50) NOT NULL DEFAULT 'pending',
	quality_score DECIMAL(5,4) NOT NULL DEFAULT 0.0,
	final_consensus_data JSONB,
	resolution_notes TEXT,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- STEALTH & PROTECTION MANAGEMENT
-- =====================================================

-- Stealth strategies and configurations
CREATE TABLE cr_stealth_strategies (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	strategy_name VARCHAR(500) NOT NULL,
	strategy_type VARCHAR(100) NOT NULL, -- cloudscraper, playwright, selenium, proxy, etc.
	configuration JSONB NOT NULL DEFAULT '{}',
	capabilities JSONB NOT NULL DEFAULT '{}',
	success_rate DECIMAL(5,4) DEFAULT 0.0,
	performance_metrics JSONB NOT NULL DEFAULT '{}',
	cost_score DECIMAL(5,2) DEFAULT 0.0,
	detection_resistance DECIMAL(5,4) DEFAULT 0.0,
	resource_usage JSONB NOT NULL DEFAULT '{}',
	status VARCHAR(50) NOT NULL DEFAULT 'active',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Protection mechanism detection profiles
CREATE TABLE cr_protection_profiles (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	domain VARCHAR(500) NOT NULL,
	protection_types JSONB NOT NULL DEFAULT '[]', -- cloudflare, captcha, akamai, etc.
	detection_confidence DECIMAL(5,4) NOT NULL DEFAULT 0.0,
	protection_characteristics JSONB NOT NULL DEFAULT '{}',
	recommended_strategies JSONB NOT NULL DEFAULT '[]',
	last_analyzed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	success_history JSONB NOT NULL DEFAULT '{}',
	adaptation_notes TEXT,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Stealth execution results and learning
CREATE TABLE cr_stealth_results (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	strategy_id UUID NOT NULL REFERENCES cr_stealth_strategies(id) ON DELETE CASCADE,
	protection_profile_id UUID REFERENCES cr_protection_profiles(id) ON DELETE SET NULL,
	target_url TEXT NOT NULL,
	execution_success BOOLEAN NOT NULL DEFAULT FALSE,
	response_time_ms INTEGER,
	detection_occurred BOOLEAN NOT NULL DEFAULT FALSE,
	bypass_success BOOLEAN NOT NULL DEFAULT FALSE,
	error_details JSONB,
	performance_data JSONB NOT NULL DEFAULT '{}',
	adaptation_data JSONB NOT NULL DEFAULT '{}',
	learning_feedback JSONB NOT NULL DEFAULT '{}',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- REAL-TIME PROCESSING & ANALYTICS
-- =====================================================

-- Real-time data streams and sessions
CREATE TABLE cr_stream_sessions (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	session_name VARCHAR(500) NOT NULL,
	target_id UUID NOT NULL REFERENCES cr_crawl_targets(id) ON DELETE CASCADE,
	pipeline_id UUID REFERENCES cr_crawl_pipelines(id) ON DELETE SET NULL,
	stream_config JSONB NOT NULL DEFAULT '{}',
	processing_config JSONB NOT NULL DEFAULT '{}',
	session_status VARCHAR(50) NOT NULL DEFAULT 'active',
	start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	end_time TIMESTAMP WITH TIME ZONE,
	records_processed INTEGER NOT NULL DEFAULT 0,
	records_per_second DECIMAL(10,2) NOT NULL DEFAULT 0.0,
	error_count INTEGER NOT NULL DEFAULT 0,
	quality_metrics JSONB NOT NULL DEFAULT '{}',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Real-time analytics and insights
CREATE TABLE cr_analytics_insights (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	insight_type VARCHAR(100) NOT NULL, -- trend, anomaly, pattern, prediction
	data_source VARCHAR(100) NOT NULL,
	insight_title VARCHAR(1000) NOT NULL,
	insight_description TEXT NOT NULL,
	insight_data JSONB NOT NULL DEFAULT '{}',
	confidence_score DECIMAL(5,4) NOT NULL DEFAULT 0.0,
	business_impact DECIMAL(5,4) NOT NULL DEFAULT 0.0,
	actionable_recommendations JSONB NOT NULL DEFAULT '[]',
	related_entities JSONB NOT NULL DEFAULT '[]',
	time_window_start TIMESTAMP WITH TIME ZONE,
	time_window_end TIMESTAMP WITH TIME ZONE,
	status VARCHAR(50) NOT NULL DEFAULT 'active',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Anomaly detection results
CREATE TABLE cr_anomaly_detections (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	dataset_id UUID REFERENCES cr_extracted_datasets(id) ON DELETE CASCADE,
	stream_session_id UUID REFERENCES cr_stream_sessions(id) ON DELETE CASCADE,
	anomaly_type VARCHAR(100) NOT NULL,
	anomaly_description TEXT NOT NULL,
	detection_algorithm VARCHAR(100) NOT NULL,
	anomaly_score DECIMAL(5,4) NOT NULL,
	confidence_level DECIMAL(5,4) NOT NULL,
	affected_records JSONB NOT NULL DEFAULT '[]',
	baseline_comparison JSONB NOT NULL DEFAULT '{}',
	business_impact_assessment JSONB NOT NULL DEFAULT '{}',
	recommended_actions JSONB NOT NULL DEFAULT '[]',
	resolution_status VARCHAR(50) NOT NULL DEFAULT 'open',
	resolved_at TIMESTAMP WITH TIME ZONE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- DISTRIBUTED PROCESSING & PERFORMANCE
-- =====================================================

-- Processing nodes for distributed crawling
CREATE TABLE cr_processing_nodes (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	node_name VARCHAR(500) NOT NULL,
	node_type VARCHAR(100) NOT NULL, -- primary, worker, analytics, stealth
	geographic_region VARCHAR(100),
	node_capabilities JSONB NOT NULL DEFAULT '{}',
	resource_limits JSONB NOT NULL DEFAULT '{}',
	current_load DECIMAL(5,2) NOT NULL DEFAULT 0.0,
	performance_metrics JSONB NOT NULL DEFAULT '{}',
	node_status VARCHAR(50) NOT NULL DEFAULT 'active',
	last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	connection_info JSONB NOT NULL DEFAULT '{}',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Distributed job management
CREATE TABLE cr_distributed_jobs (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	job_name VARCHAR(500) NOT NULL,
	job_type VARCHAR(100) NOT NULL,
	target_id UUID REFERENCES cr_crawl_targets(id) ON DELETE CASCADE,
	pipeline_id UUID REFERENCES cr_crawl_pipelines(id) ON DELETE CASCADE,
	job_configuration JSONB NOT NULL DEFAULT '{}',
	assigned_nodes JSONB NOT NULL DEFAULT '[]',
	job_status VARCHAR(50) NOT NULL DEFAULT 'pending',
	priority_level INTEGER NOT NULL DEFAULT 5,
	resource_requirements JSONB NOT NULL DEFAULT '{}',
	progress_percentage DECIMAL(5,2) NOT NULL DEFAULT 0.0,
	estimated_completion TIMESTAMP WITH TIME ZONE,
	started_at TIMESTAMP WITH TIME ZONE,
	completed_at TIMESTAMP WITH TIME ZONE,
	performance_data JSONB NOT NULL DEFAULT '{}',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics and monitoring
CREATE TABLE cr_performance_metrics (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	metric_type VARCHAR(100) NOT NULL, -- throughput, latency, accuracy, cost
	metric_name VARCHAR(500) NOT NULL,
	metric_value DECIMAL(15,4) NOT NULL,
	metric_unit VARCHAR(50),
	measurement_context JSONB NOT NULL DEFAULT '{}',
	related_entity_type VARCHAR(100), -- target, pipeline, node, job
	related_entity_id UUID,
	time_window_start TIMESTAMP WITH TIME ZONE,
	time_window_end TIMESTAMP WITH TIME ZONE,
	aggregation_level VARCHAR(50) NOT NULL DEFAULT 'raw', -- raw, hourly, daily, weekly
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- GOVERNANCE, COMPLIANCE & AUDIT
-- =====================================================

-- Governance policies and rules
CREATE TABLE cr_governance_policies (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	policy_name VARCHAR(500) NOT NULL,
	policy_type VARCHAR(100) NOT NULL, -- ethical, legal, business, technical
	policy_description TEXT NOT NULL,
	policy_rules JSONB NOT NULL DEFAULT '[]',
	enforcement_level VARCHAR(50) NOT NULL DEFAULT 'warning', -- strict, warning, advisory
	applicable_domains JSONB NOT NULL DEFAULT '[]',
	applicable_targets JSONB NOT NULL DEFAULT '[]',
	violation_actions JSONB NOT NULL DEFAULT '[]',
	monitoring_config JSONB NOT NULL DEFAULT '{}',
	policy_status VARCHAR(50) NOT NULL DEFAULT 'active',
	effective_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	expiry_date TIMESTAMP WITH TIME ZONE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255),
	updated_by VARCHAR(255)
);

-- Compliance audit trails
CREATE TABLE cr_compliance_audits (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	audit_type VARCHAR(100) NOT NULL, -- gdpr, ccpa, robots_txt, rate_limiting
	entity_type VARCHAR(100) NOT NULL,
	entity_id UUID,
	action_performed VARCHAR(500) NOT NULL,
	action_details JSONB NOT NULL DEFAULT '{}',
	compliance_status VARCHAR(50) NOT NULL DEFAULT 'compliant',
	policy_violations JSONB NOT NULL DEFAULT '[]',
	remediation_actions JSONB NOT NULL DEFAULT '[]',
	audit_metadata JSONB NOT NULL DEFAULT '{}',
	performed_by VARCHAR(255),
	audit_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	reviewed_by VARCHAR(255),
	review_timestamp TIMESTAMP WITH TIME ZONE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Robots.txt compliance and respect tracking
CREATE TABLE cr_robots_compliance (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	domain VARCHAR(500) NOT NULL,
	robots_txt_url TEXT,
	robots_txt_content TEXT,
	parsed_rules JSONB NOT NULL DEFAULT '{}',
	compliance_analysis JSONB NOT NULL DEFAULT '{}',
	allowed_paths JSONB NOT NULL DEFAULT '[]',
	disallowed_paths JSONB NOT NULL DEFAULT '[]',
	crawl_delay_seconds INTEGER DEFAULT 0,
	sitemap_urls JSONB NOT NULL DEFAULT '[]',
	last_fetched TIMESTAMP WITH TIME ZONE,
	last_analyzed TIMESTAMP WITH TIME ZONE,
	compliance_status VARCHAR(50) NOT NULL DEFAULT 'pending',
	violation_count INTEGER NOT NULL DEFAULT 0,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =====================================================

-- Primary lookup indexes
CREATE INDEX idx_cr_crawl_targets_tenant_status ON cr_crawl_targets(tenant_id, status);
CREATE INDEX idx_cr_crawl_targets_name ON cr_crawl_targets(tenant_id, name);
CREATE INDEX idx_cr_crawl_targets_created ON cr_crawl_targets(created_at DESC);

CREATE INDEX idx_cr_crawl_pipelines_tenant_status ON cr_crawl_pipelines(tenant_id, deployment_status);
CREATE INDEX idx_cr_crawl_pipelines_target ON cr_crawl_pipelines(target_id);

CREATE INDEX idx_cr_data_sources_tenant_type ON cr_data_sources(tenant_id, source_type);
CREATE INDEX idx_cr_data_sources_health ON cr_data_sources(health_status, last_health_check);

-- Data extraction indexes
CREATE INDEX idx_cr_extracted_datasets_tenant_created ON cr_extracted_datasets(tenant_id, created_at DESC);
CREATE INDEX idx_cr_extracted_datasets_target ON cr_extracted_datasets(crawl_target_id);
CREATE INDEX idx_cr_extracted_datasets_validation ON cr_extracted_datasets(validation_status, consensus_score);

CREATE INDEX idx_cr_data_records_dataset ON cr_data_records(dataset_id);
CREATE INDEX idx_cr_data_records_quality ON cr_data_records(quality_score DESC, confidence_score DESC);
CREATE INDEX idx_cr_data_records_validation ON cr_data_records(validation_status);

-- Full-text search indexes
CREATE INDEX idx_cr_data_records_content_gin ON cr_data_records USING gin(to_tsvector('english', processed_content));
CREATE INDEX idx_cr_data_records_url_trgm ON cr_data_records USING gin(source_url gin_trgm_ops);

-- Business entities indexes
CREATE INDEX idx_cr_business_entities_record ON cr_business_entities(record_id);
CREATE INDEX idx_cr_business_entities_type_name ON cr_business_entities(entity_type, entity_name);
CREATE INDEX idx_cr_business_entities_confidence ON cr_business_entities(confidence_score DESC);

-- Validation indexes
CREATE INDEX idx_cr_validation_sessions_tenant_status ON cr_validation_sessions(tenant_id, session_status);
CREATE INDEX idx_cr_validation_sessions_dataset ON cr_validation_sessions(dataset_id);

CREATE INDEX idx_cr_validators_session ON cr_validators(session_id);
CREATE INDEX idx_cr_validators_user ON cr_validators(tenant_id, user_id);

CREATE INDEX idx_cr_validation_feedback_session ON cr_validation_feedback(session_id);
CREATE INDEX idx_cr_validation_feedback_validator ON cr_validation_feedback(validator_id);
CREATE INDEX idx_cr_validation_feedback_record ON cr_validation_feedback(record_id);

CREATE INDEX idx_cr_validation_consensus_session ON cr_validation_consensus(session_id);
CREATE INDEX idx_cr_validation_consensus_record ON cr_validation_consensus(record_id);
CREATE INDEX idx_cr_validation_consensus_status ON cr_validation_consensus(consensus_status, consensus_score);

-- Stealth and protection indexes
CREATE INDEX idx_cr_stealth_strategies_tenant_type ON cr_stealth_strategies(tenant_id, strategy_type);
CREATE INDEX idx_cr_stealth_strategies_success ON cr_stealth_strategies(success_rate DESC, cost_score ASC);

CREATE INDEX idx_cr_protection_profiles_domain ON cr_protection_profiles(domain);
CREATE INDEX idx_cr_protection_profiles_analyzed ON cr_protection_profiles(last_analyzed DESC);

CREATE INDEX idx_cr_stealth_results_strategy ON cr_stealth_results(strategy_id);
CREATE INDEX idx_cr_stealth_results_success ON cr_stealth_results(execution_success, detection_occurred);
CREATE INDEX idx_cr_stealth_results_url ON cr_stealth_results(target_url);

-- Real-time processing indexes
CREATE INDEX idx_cr_stream_sessions_tenant_status ON cr_stream_sessions(tenant_id, session_status);
CREATE INDEX idx_cr_stream_sessions_target ON cr_stream_sessions(target_id);
CREATE INDEX idx_cr_stream_sessions_performance ON cr_stream_sessions(records_per_second DESC);

CREATE INDEX idx_cr_analytics_insights_tenant_type ON cr_analytics_insights(tenant_id, insight_type);
CREATE INDEX idx_cr_analytics_insights_created ON cr_analytics_insights(created_at DESC);
CREATE INDEX idx_cr_analytics_insights_confidence ON cr_analytics_insights(confidence_score DESC);

CREATE INDEX idx_cr_anomaly_detections_tenant_type ON cr_anomaly_detections(tenant_id, anomaly_type);
CREATE INDEX idx_cr_anomaly_detections_score ON cr_anomaly_detections(anomaly_score DESC);
CREATE INDEX idx_cr_anomaly_detections_status ON cr_anomaly_detections(resolution_status);

-- Distributed processing indexes
CREATE INDEX idx_cr_processing_nodes_tenant_status ON cr_processing_nodes(tenant_id, node_status);
CREATE INDEX idx_cr_processing_nodes_load ON cr_processing_nodes(current_load ASC);
CREATE INDEX idx_cr_processing_nodes_region ON cr_processing_nodes(geographic_region);

CREATE INDEX idx_cr_distributed_jobs_tenant_status ON cr_distributed_jobs(tenant_id, job_status);
CREATE INDEX idx_cr_distributed_jobs_priority ON cr_distributed_jobs(priority_level DESC, created_at ASC);
CREATE INDEX idx_cr_distributed_jobs_target ON cr_distributed_jobs(target_id);

-- Performance metrics indexes
CREATE INDEX idx_cr_performance_metrics_tenant_type ON cr_performance_metrics(tenant_id, metric_type);
CREATE INDEX idx_cr_performance_metrics_created ON cr_performance_metrics(created_at DESC);
CREATE INDEX idx_cr_performance_metrics_entity ON cr_performance_metrics(related_entity_type, related_entity_id);

-- Governance and compliance indexes
CREATE INDEX idx_cr_governance_policies_tenant_type ON cr_governance_policies(tenant_id, policy_type);
CREATE INDEX idx_cr_governance_policies_status ON cr_governance_policies(policy_status, effective_date);

CREATE INDEX idx_cr_compliance_audits_tenant_type ON cr_compliance_audits(tenant_id, audit_type);
CREATE INDEX idx_cr_compliance_audits_entity ON cr_compliance_audits(entity_type, entity_id);
CREATE INDEX idx_cr_compliance_audits_timestamp ON cr_compliance_audits(audit_timestamp DESC);
CREATE INDEX idx_cr_compliance_audits_status ON cr_compliance_audits(compliance_status);

CREATE INDEX idx_cr_robots_compliance_domain ON cr_robots_compliance(domain);
CREATE INDEX idx_cr_robots_compliance_status ON cr_robots_compliance(compliance_status, violation_count);

-- =====================================================
-- TRIGGERS FOR AUDIT AND MAINTENANCE
-- =====================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
	NEW.updated_at = CURRENT_TIMESTAMP;
	RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers to main tables
CREATE TRIGGER update_cr_crawl_targets_updated_at BEFORE UPDATE ON cr_crawl_targets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_crawl_pipelines_updated_at BEFORE UPDATE ON cr_crawl_pipelines FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_data_sources_updated_at BEFORE UPDATE ON cr_data_sources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_extracted_datasets_updated_at BEFORE UPDATE ON cr_extracted_datasets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_data_records_updated_at BEFORE UPDATE ON cr_data_records FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_validation_sessions_updated_at BEFORE UPDATE ON cr_validation_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_stealth_strategies_updated_at BEFORE UPDATE ON cr_stealth_strategies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_protection_profiles_updated_at BEFORE UPDATE ON cr_protection_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_stream_sessions_updated_at BEFORE UPDATE ON cr_stream_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_analytics_insights_updated_at BEFORE UPDATE ON cr_analytics_insights FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_processing_nodes_updated_at BEFORE UPDATE ON cr_processing_nodes FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_distributed_jobs_updated_at BEFORE UPDATE ON cr_distributed_jobs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_governance_policies_updated_at BEFORE UPDATE ON cr_governance_policies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cr_robots_compliance_updated_at BEFORE UPDATE ON cr_robots_compliance FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- FUNCTIONS FOR ADVANCED OPERATIONS
-- =====================================================

-- Function to calculate dataset quality score
CREATE OR REPLACE FUNCTION calculate_dataset_quality_score(dataset_id UUID)
RETURNS DECIMAL(5,4) AS $$
DECLARE
	quality_score DECIMAL(5,4) := 0.0;
	record_count INTEGER := 0;
	avg_confidence DECIMAL(5,4) := 0.0;
	validation_ratio DECIMAL(5,4) := 0.0;
BEGIN
	-- Get basic dataset metrics
	SELECT 
		COUNT(*),
		AVG(confidence_score),
		COUNT(CASE WHEN validation_status = 'approved' THEN 1 END)::DECIMAL / COUNT(*) 
	INTO record_count, avg_confidence, validation_ratio
	FROM cr_data_records 
	WHERE dataset_id = calculate_dataset_quality_score.dataset_id;
	
	-- Calculate composite quality score
	IF record_count > 0 THEN
		quality_score := (avg_confidence * 0.4) + (validation_ratio * 0.6);
	END IF;
	
	RETURN COALESCE(quality_score, 0.0);
END;
$$ LANGUAGE plpgsql;

-- Function to update consensus metrics
CREATE OR REPLACE FUNCTION update_consensus_metrics(session_id UUID)
RETURNS VOID AS $$
BEGIN
	-- Update consensus for all records in the session
	UPDATE cr_validation_consensus 
	SET 
		consensus_score = CASE 
			WHEN total_validators > 0 THEN (approval_count::DECIMAL / total_validators) 
			ELSE 0.0 
		END,
		consensus_status = CASE 
			WHEN total_validators = 0 THEN 'pending'
			WHEN (approval_count::DECIMAL / total_validators) >= 0.8 THEN 'consensus'
			WHEN (rejection_count::DECIMAL / total_validators) >= 0.8 THEN 'rejected'
			ELSE 'conflicted'
		END,
		updated_at = CURRENT_TIMESTAMP
	WHERE session_id = update_consensus_metrics.session_id;
	
	-- Update session completion percentage
	UPDATE cr_validation_sessions 
	SET 
		completion_percentage = (
			SELECT COUNT(CASE WHEN consensus_status IN ('consensus', 'rejected') THEN 1 END)::DECIMAL * 100.0 / COUNT(*)
			FROM cr_validation_consensus 
			WHERE session_id = update_consensus_metrics.session_id
		),
		updated_at = CURRENT_TIMESTAMP
	WHERE id = update_consensus_metrics.session_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update stealth strategy success rates
CREATE OR REPLACE FUNCTION update_stealth_success_rates()
RETURNS VOID AS $$
BEGIN
	UPDATE cr_stealth_strategies 
	SET 
		success_rate = COALESCE((
			SELECT COUNT(CASE WHEN execution_success THEN 1 END)::DECIMAL / COUNT(*)
			FROM cr_stealth_results 
			WHERE strategy_id = cr_stealth_strategies.id
			AND created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
		), 0.0),
		updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- SAMPLE DATA FOR TESTING (OPTIONAL)
-- =====================================================

-- Insert sample governance policies
INSERT INTO cr_governance_policies (tenant_id, policy_name, policy_type, policy_description, policy_rules, enforcement_level) VALUES
('default_tenant', 'Respectful Crawling Policy', 'ethical', 'Ensures respectful and ethical web crawling practices', 
 '["respect_robots_txt", "implement_delays", "avoid_overloading", "identify_crawler"]', 'strict'),
('default_tenant', 'Data Privacy Compliance', 'legal', 'GDPR and CCPA compliance for crawled data',
 '["anonymize_pii", "secure_storage", "data_retention_limits", "consent_tracking"]', 'strict'),
('default_tenant', 'Quality Assurance Standards', 'business', 'Minimum quality standards for extracted data',
 '["minimum_confidence_70", "validation_required", "duplicate_detection", "accuracy_monitoring"]', 'warning');

-- Insert sample stealth strategies
INSERT INTO cr_stealth_strategies (tenant_id, strategy_name, strategy_type, configuration, capabilities, success_rate, cost_score) VALUES
('default_tenant', 'CloudScraper Priority', 'cloudscraper', 
 '{"browser": "chrome", "delay_range": [2, 5], "user_agents": "random"}', 
 '["javascript_execution", "captcha_bypass", "cloudflare_bypass"]', 0.85, 1.2),
('default_tenant', 'Playwright Stealth', 'playwright', 
 '{"browser": "chromium", "stealth_mode": true, "viewport": "random"}', 
 '["full_browser", "javascript_heavy", "complex_interactions"]', 0.78, 2.8),
('default_tenant', 'Basic HTTP with Rotation', 'http_stealth',
 '{"rotate_agents": true, "proxy_rotation": true, "delay_range": [1, 3]}',
 '["fast_requests", "basic_bypass", "low_detection"]', 0.65, 0.5);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for crawler dashboard metrics
CREATE VIEW cr_dashboard_metrics AS
SELECT 
	cr_crawl_targets.tenant_id,
	COUNT(DISTINCT cr_crawl_targets.id) as total_targets,
	COUNT(DISTINCT cr_crawl_pipelines.id) as total_pipelines,
	COUNT(DISTINCT cr_extracted_datasets.id) as total_datasets,
	SUM(cr_extracted_datasets.record_count) as total_records,
	AVG(CASE WHEN cr_extracted_datasets.consensus_score > 0 THEN cr_extracted_datasets.consensus_score END) as avg_quality_score,
	COUNT(DISTINCT cr_validation_sessions.id) as active_validation_sessions
FROM cr_crawl_targets
LEFT JOIN cr_crawl_pipelines ON cr_crawl_targets.id = cr_crawl_pipelines.target_id
LEFT JOIN cr_extracted_datasets ON cr_crawl_targets.id = cr_extracted_datasets.crawl_target_id
LEFT JOIN cr_validation_sessions ON cr_extracted_datasets.id = cr_validation_sessions.dataset_id
WHERE cr_crawl_targets.status = 'active'
GROUP BY cr_crawl_targets.tenant_id;

-- View for stealth performance overview
CREATE VIEW cr_stealth_performance AS
SELECT 
	cs.tenant_id,
	cs.strategy_name,
	cs.strategy_type,
	cs.success_rate,
	cs.cost_score,
	COUNT(csr.id) as total_attempts,
	COUNT(CASE WHEN csr.execution_success THEN 1 END) as successful_attempts,
	AVG(csr.response_time_ms) as avg_response_time,
	COUNT(CASE WHEN csr.detection_occurred THEN 1 END) as detection_count
FROM cr_stealth_strategies cs
LEFT JOIN cr_stealth_results csr ON cs.id = csr.strategy_id
WHERE cs.status = 'active'
GROUP BY cs.tenant_id, cs.id, cs.strategy_name, cs.strategy_type, cs.success_rate, cs.cost_score;

-- View for validation progress tracking
CREATE VIEW cr_validation_progress AS
SELECT 
	vs.tenant_id,
	vs.session_name,
	vs.session_status,
	vs.completion_percentage,
	COUNT(DISTINCT v.id) as validator_count,
	COUNT(DISTINCT vf.id) as feedback_count,
	COUNT(CASE WHEN vc.consensus_status = 'consensus' THEN 1 END) as consensus_reached,
	COUNT(CASE WHEN vc.consensus_status = 'conflicted' THEN 1 END) as conflicts_pending,
	AVG(vf.quality_rating) as avg_quality_rating
FROM cr_validation_sessions vs
LEFT JOIN cr_validators v ON vs.id = v.session_id
LEFT JOIN cr_validation_feedback vf ON vs.id = vf.session_id
LEFT JOIN cr_validation_consensus vc ON vs.id = vc.session_id
GROUP BY vs.tenant_id, vs.id, vs.session_name, vs.session_status, vs.completion_percentage;

-- =====================================================
-- CLEANUP AND MAINTENANCE PROCEDURES
-- =====================================================

-- Procedure to archive old performance metrics
CREATE OR REPLACE FUNCTION archive_old_performance_metrics(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
	deleted_count INTEGER;
BEGIN
	DELETE FROM cr_performance_metrics 
	WHERE created_at < CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL
	AND aggregation_level = 'raw';
	
	GET DIAGNOSTICS deleted_count = ROW_COUNT;
	RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Procedure to clean up old stealth results
CREATE OR REPLACE FUNCTION cleanup_old_stealth_results(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
	deleted_count INTEGER;
BEGIN
	DELETE FROM cr_stealth_results 
	WHERE created_at < CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL;
	
	GET DIAGNOSTICS deleted_count = ROW_COUNT;
	RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- RAG AND GRAPHRAG INTEGRATION TABLES
-- =====================================================

-- RAG chunks with vector embeddings
CREATE TABLE cr_rag_chunks (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	record_id UUID NOT NULL REFERENCES cr_data_records(id) ON DELETE CASCADE,
	chunk_index INTEGER NOT NULL,
	chunk_text TEXT NOT NULL,
	chunk_markdown TEXT NOT NULL,
	chunk_fingerprint VARCHAR(64) DEFAULT '',
	vector_embeddings vector(1536), -- Configurable dimensions
	embedding_model VARCHAR(100) DEFAULT 'text-embedding-ada-002',
	vector_dimensions INTEGER DEFAULT 1536,
	semantic_similarity_threshold DECIMAL(5,4) DEFAULT 0.8,
	chunk_overlap_start INTEGER DEFAULT 0,
	chunk_overlap_end INTEGER DEFAULT 0,
	entities_extracted TEXT[] DEFAULT '{}',
	related_chunks TEXT[] DEFAULT '{}',
	contextual_metadata JSONB NOT NULL DEFAULT '{}',
	indexing_status VARCHAR(50) NOT NULL DEFAULT 'pending',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT cr_rag_chunks_chunk_index_check CHECK (chunk_index >= 0),
	CONSTRAINT cr_rag_chunks_dimensions_check CHECK (vector_dimensions > 0),
	UNIQUE(tenant_id, record_id, chunk_index)
);

-- GraphRAG knowledge graph nodes
CREATE TABLE cr_graphrag_nodes (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	record_id UUID NOT NULL REFERENCES cr_data_records(id) ON DELETE CASCADE,
	node_type VARCHAR(100) NOT NULL,
	node_name VARCHAR(1000) NOT NULL,
	node_description TEXT,
	node_properties JSONB NOT NULL DEFAULT '{}',
	entity_type VARCHAR(100),
	confidence_score DECIMAL(5,4) DEFAULT 0.0,
	salience_score DECIMAL(5,4) DEFAULT 0.0,
	vector_embeddings vector(1536),
	related_chunks TEXT[] DEFAULT '{}',
	knowledge_graph_id VARCHAR(255),
	node_status VARCHAR(50) NOT NULL DEFAULT 'active',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT cr_graphrag_nodes_confidence_check CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
	CONSTRAINT cr_graphrag_nodes_salience_check CHECK (salience_score >= 0.0 AND salience_score <= 1.0)
);

-- GraphRAG knowledge graph relations
CREATE TABLE cr_graphrag_relations (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	source_node_id UUID NOT NULL REFERENCES cr_graphrag_nodes(id) ON DELETE CASCADE,
	target_node_id UUID NOT NULL REFERENCES cr_graphrag_nodes(id) ON DELETE CASCADE,
	relation_type VARCHAR(100) NOT NULL,
	relation_label VARCHAR(500) NOT NULL,
	relation_properties JSONB NOT NULL DEFAULT '{}',
	confidence_score DECIMAL(5,4) DEFAULT 0.0,
	strength_score DECIMAL(5,4) DEFAULT 0.0,
	evidence_chunks TEXT[] DEFAULT '{}',
	context_window TEXT,
	knowledge_graph_id VARCHAR(255),
	relation_status VARCHAR(50) NOT NULL DEFAULT 'active',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT cr_graphrag_relations_confidence_check CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
	CONSTRAINT cr_graphrag_relations_strength_check CHECK (strength_score >= 0.0 AND strength_score <= 1.0),
	CONSTRAINT cr_graphrag_relations_self_ref_check CHECK (source_node_id != target_node_id)
);

-- Knowledge graphs for GraphRAG
CREATE TABLE cr_knowledge_graphs (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	graph_name VARCHAR(500) NOT NULL,
	description TEXT,
	domain VARCHAR(100) NOT NULL,
	node_count INTEGER DEFAULT 0,
	relation_count INTEGER DEFAULT 0,
	entity_types TEXT[] DEFAULT '{}',
	relation_types TEXT[] DEFAULT '{}',
	graph_statistics JSONB NOT NULL DEFAULT '{}',
	graph_schema JSONB NOT NULL DEFAULT '{}',
	indexing_config JSONB NOT NULL DEFAULT '{}',
	graph_status VARCHAR(50) NOT NULL DEFAULT 'building',
	last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(255),
	CONSTRAINT cr_knowledge_graphs_counts_check CHECK (node_count >= 0 AND relation_count >= 0),
	UNIQUE(tenant_id, graph_name)
);

-- Content fingerprints for duplicate detection
CREATE TABLE cr_content_fingerprints (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id VARCHAR(255) NOT NULL,
	fingerprint_hash VARCHAR(64) NOT NULL,
	content_type VARCHAR(100) NOT NULL,
	content_length INTEGER NOT NULL,
	source_url TEXT NOT NULL,
	related_records TEXT[] DEFAULT '{}',
	first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	occurrence_count INTEGER DEFAULT 1,
	duplicate_cluster_id VARCHAR(255),
	content_similarity_scores JSONB NOT NULL DEFAULT '{}',
	fingerprint_metadata JSONB NOT NULL DEFAULT '{}',
	status VARCHAR(50) NOT NULL DEFAULT 'unique',
	CONSTRAINT cr_content_fingerprints_length_check CHECK (content_length >= 0),
	CONSTRAINT cr_content_fingerprints_count_check CHECK (occurrence_count >= 1),
	UNIQUE(tenant_id, fingerprint_hash)
);

-- =====================================================
-- RAG/GRAPHRAG INDEXES
-- =====================================================

-- RAG chunk indexes
CREATE INDEX idx_cr_rag_chunks_tenant_record ON cr_rag_chunks(tenant_id, record_id);
CREATE INDEX idx_cr_rag_chunks_indexing_status ON cr_rag_chunks(indexing_status) WHERE indexing_status != 'indexed';
CREATE INDEX idx_cr_rag_chunks_fingerprint ON cr_rag_chunks(chunk_fingerprint) WHERE chunk_fingerprint != '';
CREATE INDEX idx_cr_rag_chunks_embedding_model ON cr_rag_chunks(embedding_model);
CREATE INDEX idx_cr_rag_chunks_created_at ON cr_rag_chunks(created_at);
-- Vector similarity search index (requires pgvector)
CREATE INDEX idx_cr_rag_chunks_vector_cosine ON cr_rag_chunks USING ivfflat (vector_embeddings vector_cosine_ops) WITH (lists = 100);

-- GraphRAG node indexes
CREATE INDEX idx_cr_graphrag_nodes_tenant_type ON cr_graphrag_nodes(tenant_id, node_type);
CREATE INDEX idx_cr_graphrag_nodes_name_gin ON cr_graphrag_nodes USING gin(to_tsvector('english', node_name));
CREATE INDEX idx_cr_graphrag_nodes_entity_type ON cr_graphrag_nodes(entity_type) WHERE entity_type IS NOT NULL;
CREATE INDEX idx_cr_graphrag_nodes_confidence ON cr_graphrag_nodes(confidence_score DESC);
CREATE INDEX idx_cr_graphrag_nodes_knowledge_graph ON cr_graphrag_nodes(knowledge_graph_id) WHERE knowledge_graph_id IS NOT NULL;
CREATE INDEX idx_cr_graphrag_nodes_status ON cr_graphrag_nodes(node_status) WHERE node_status != 'active';

-- GraphRAG relation indexes
CREATE INDEX idx_cr_graphrag_relations_source_target ON cr_graphrag_relations(source_node_id, target_node_id);
CREATE INDEX idx_cr_graphrag_relations_type ON cr_graphrag_relations(relation_type);
CREATE INDEX idx_cr_graphrag_relations_confidence ON cr_graphrag_relations(confidence_score DESC);
CREATE INDEX idx_cr_graphrag_relations_knowledge_graph ON cr_graphrag_relations(knowledge_graph_id) WHERE knowledge_graph_id IS NOT NULL;
CREATE INDEX idx_cr_graphrag_relations_status ON cr_graphrag_relations(relation_status) WHERE relation_status != 'active';

-- Knowledge graph indexes
CREATE INDEX idx_cr_knowledge_graphs_tenant_domain ON cr_knowledge_graphs(tenant_id, domain);
CREATE INDEX idx_cr_knowledge_graphs_status ON cr_knowledge_graphs(graph_status) WHERE graph_status != 'active';
CREATE INDEX idx_cr_knowledge_graphs_updated ON cr_knowledge_graphs(last_updated DESC);

-- Content fingerprint indexes
CREATE INDEX idx_cr_content_fingerprints_hash ON cr_content_fingerprints(fingerprint_hash);
CREATE INDEX idx_cr_content_fingerprints_tenant_type ON cr_content_fingerprints(tenant_id, content_type);
CREATE INDEX idx_cr_content_fingerprints_occurrence ON cr_content_fingerprints(occurrence_count DESC) WHERE occurrence_count > 1;
CREATE INDEX idx_cr_content_fingerprints_cluster ON cr_content_fingerprints(duplicate_cluster_id) WHERE duplicate_cluster_id IS NOT NULL;
CREATE INDEX idx_cr_content_fingerprints_status ON cr_content_fingerprints(status) WHERE status != 'unique';

-- Additional indexes for data records (RAG/GraphRAG fields)
CREATE INDEX idx_cr_data_records_fingerprint ON cr_data_records(content_fingerprint) WHERE content_fingerprint != '';
CREATE INDEX idx_cr_data_records_processing_stage ON cr_data_records(content_processing_stage) WHERE content_processing_stage != 'completed';
CREATE INDEX idx_cr_data_records_graphrag_node ON cr_data_records(graphrag_node_id) WHERE graphrag_node_id IS NOT NULL;
CREATE INDEX idx_cr_data_records_vector_cosine ON cr_data_records USING ivfflat (vector_embeddings vector_cosine_ops) WITH (lists = 100);

-- Grant permissions (adjust as needed for APG security model)
-- These should be customized based on APG's specific RBAC implementation
GRANT USAGE ON SCHEMA crawler TO apg_crawler_service;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA crawler TO apg_crawler_service;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA crawler TO apg_crawler_service;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA crawler TO apg_crawler_service;

-- Grant read-only access for analytics
GRANT USAGE ON SCHEMA crawler TO apg_analytics_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA crawler TO apg_analytics_readonly;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA crawler TO apg_analytics_readonly;

COMMENT ON SCHEMA crawler IS 'APG Crawler Capability - Enterprise Web Intelligence Platform';
COMMENT ON TABLE cr_crawl_targets IS 'Defines crawling targets with business context and requirements';
COMMENT ON TABLE cr_crawl_pipelines IS 'Visual pipeline configurations for automated processing';
COMMENT ON TABLE cr_extracted_datasets IS 'Datasets extracted from crawling operations with quality metrics';
COMMENT ON TABLE cr_validation_sessions IS 'Collaborative validation sessions for quality assurance';
COMMENT ON TABLE cr_stealth_strategies IS 'Stealth strategies for bypassing anti-bot protection';
COMMENT ON TABLE cr_analytics_insights IS 'Real-time analytics insights and business intelligence';
COMMENT ON TABLE cr_governance_policies IS 'Governance policies for ethical and compliant crawling';
COMMENT ON TABLE cr_rag_chunks IS 'RAG chunks with vector embeddings for semantic search';
COMMENT ON TABLE cr_graphrag_nodes IS 'GraphRAG knowledge graph nodes for entity representation';
COMMENT ON TABLE cr_graphrag_relations IS 'GraphRAG knowledge graph relations between entities';
COMMENT ON TABLE cr_knowledge_graphs IS 'Knowledge graphs container for GraphRAG processing';
COMMENT ON TABLE cr_content_fingerprints IS 'Content fingerprints for duplicate detection and versioning';

-- Schema validation
DO $$
BEGIN
	ASSERT (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'crawler') >= 25, 
		'Insufficient tables created in crawler schema';
	ASSERT (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'crawler') >= 60,
		'Insufficient indexes created for crawler schema';
	RAISE NOTICE 'APG Crawler database schema validation completed successfully';
END $$;