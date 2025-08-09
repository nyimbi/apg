-- APG Natural Language Processing Database Schema
-- PostgreSQL schema with vector extensions, multi-tenancy, and performance optimization
-- Compatible with APG's existing models and multi-tenant architecture

-- Enable required extensions for NLP processing
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- trigram matching for text search
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- GIN indexes for performance

-- Create schema for NLP capability
CREATE SCHEMA IF NOT EXISTS nlp;

-- Set search path to include NLP schema
SET search_path TO nlp, public;

-- Document storage with multi-tenant isolation
CREATE TABLE nlp.documents (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	
	-- Document content and metadata
	content TEXT NOT NULL CHECK (length(content) > 0),
	title VARCHAR(500),
	language VARCHAR(10),
	detected_language VARCHAR(10),
	content_type VARCHAR(50) DEFAULT 'plain_text',
	
	-- Document metadata (JSONB for flexible storage)
	metadata JSONB DEFAULT '{}',
	source_url TEXT,
	author VARCHAR(200),
	created_date TIMESTAMP WITH TIME ZONE,
	
	-- Processing information
	processing_history UUID[] DEFAULT '{}',
	quality_score DECIMAL(3,2) DEFAULT 0.0 CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
	word_count INTEGER DEFAULT 0 CHECK (word_count >= 0),
	character_count INTEGER DEFAULT 0 CHECK (character_count >= 0),
	
	-- Vector embeddings for semantic search (1536-dimensional for OpenAI embeddings)
	content_embedding vector(1536),
	title_embedding vector(384),  -- Smaller embedding for titles
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by UUID,
	updated_by UUID,
	is_deleted BOOLEAN DEFAULT FALSE,
	deleted_at TIMESTAMP WITH TIME ZONE
);

-- NLP model registry with performance tracking
CREATE TABLE nlp.models (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	
	-- Model identity and configuration
	name VARCHAR(200) NOT NULL,
	model_key VARCHAR(200) NOT NULL,
	version VARCHAR(50) DEFAULT '1.0.0',
	provider VARCHAR(50) NOT NULL,
	provider_model_name VARCHAR(200) NOT NULL,
	model_path TEXT,
	config_params JSONB DEFAULT '{}',
	
	-- Model capabilities
	supported_tasks TEXT[] NOT NULL,
	supported_languages TEXT[] NOT NULL,
	max_input_length INTEGER,
	context_window INTEGER,
	
	-- Performance characteristics
	average_latency_ms DECIMAL(10,3) DEFAULT 0.0 CHECK (average_latency_ms >= 0.0),
	throughput_docs_per_minute INTEGER DEFAULT 0 CHECK (throughput_docs_per_minute >= 0),
	memory_usage_mb DECIMAL(10,2) DEFAULT 0.0 CHECK (memory_usage_mb >= 0.0),
	accuracy_score DECIMAL(3,2) DEFAULT 0.0 CHECK (accuracy_score >= 0.0 AND accuracy_score <= 1.0),
	
	-- Model status and health
	is_active BOOLEAN DEFAULT TRUE,
	is_loaded BOOLEAN DEFAULT FALSE,
	health_status VARCHAR(20) DEFAULT 'unknown' CHECK (health_status IN ('healthy', 'degraded', 'unhealthy', 'unknown')),
	last_health_check TIMESTAMP WITH TIME ZONE,
	
	-- Usage statistics
	total_requests BIGINT DEFAULT 0 CHECK (total_requests >= 0),
	successful_requests BIGINT DEFAULT 0 CHECK (successful_requests >= 0),
	failed_requests BIGINT DEFAULT 0 CHECK (failed_requests >= 0),
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by UUID,
	updated_by UUID,
	is_deleted BOOLEAN DEFAULT FALSE,
	deleted_at TIMESTAMP WITH TIME ZONE
);

-- Processing requests with comprehensive tracking
CREATE TABLE nlp.processing_requests (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	user_id UUID,
	session_id UUID,
	
	-- Processing configuration
	task_type VARCHAR(50) NOT NULL,
	document_id UUID REFERENCES nlp.documents(id),
	text_content TEXT,
	language VARCHAR(10),
	quality_level VARCHAR(20) DEFAULT 'balanced' CHECK (quality_level IN ('fast', 'balanced', 'accurate', 'best')),
	
	-- Model selection preferences
	preferred_model UUID REFERENCES nlp.models(id),
	preferred_provider VARCHAR(50),
	fallback_enabled BOOLEAN DEFAULT TRUE,
	
	-- Processing options (JSONB for flexibility)
	parameters JSONB DEFAULT '{}',
	timeout_seconds INTEGER DEFAULT 300 CHECK (timeout_seconds >= 1 AND timeout_seconds <= 3600),
	priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
	
	-- Output options
	include_confidence BOOLEAN DEFAULT TRUE,
	include_explanations BOOLEAN DEFAULT FALSE,
	output_format VARCHAR(20) DEFAULT 'json' CHECK (output_format IN ('json', 'xml', 'text')),
	
	-- Request status
	status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by UUID
);

-- Processing results with comprehensive metadata
CREATE TABLE nlp.processing_results (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	request_id UUID NOT NULL REFERENCES nlp.processing_requests(id),
	tenant_id UUID NOT NULL,
	
	-- Processing metadata
	task_type VARCHAR(50) NOT NULL,
	model_used UUID REFERENCES nlp.models(id),
	provider_used VARCHAR(50) NOT NULL,
	language_detected VARCHAR(10),
	
	-- Processing performance metrics
	processing_time_ms DECIMAL(10,3) NOT NULL CHECK (processing_time_ms >= 0.0),
	queue_time_ms DECIMAL(10,3) DEFAULT 0.0 CHECK (queue_time_ms >= 0.0),
	total_time_ms DECIMAL(10,3) NOT NULL CHECK (total_time_ms >= 0.0),
	confidence_score DECIMAL(3,2) DEFAULT 0.0 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
	
	-- Processing results (JSONB for flexible storage)
	results JSONB NOT NULL,
	metadata JSONB DEFAULT '{}',
	explanations TEXT[],
	warnings TEXT[] DEFAULT '{}',
	
	-- Quality metrics
	quality_score DECIMAL(3,2) DEFAULT 0.0 CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
	completeness_score DECIMAL(3,2) DEFAULT 0.0 CHECK (completeness_score >= 0.0 AND completeness_score <= 1.0),
	
	-- Status and error handling
	status VARCHAR(20) DEFAULT 'completed' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
	error_message TEXT,
	error_code VARCHAR(50),
	
	-- Result embeddings for similarity search
	result_embedding vector(1536),
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	processed_by UUID
);

-- Streaming sessions for real-time processing
CREATE TABLE nlp.streaming_sessions (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	user_id UUID NOT NULL,
	
	-- Session configuration
	task_type VARCHAR(50) NOT NULL,
	model_id UUID REFERENCES nlp.models(id),
	language VARCHAR(10),
	
	-- Streaming parameters
	chunk_size INTEGER DEFAULT 1000 CHECK (chunk_size >= 100 AND chunk_size <= 10000),
	overlap_size INTEGER DEFAULT 100 CHECK (overlap_size >= 0 AND overlap_size <= 1000),
	aggregation_window_ms INTEGER DEFAULT 5000 CHECK (aggregation_window_ms >= 1000 AND aggregation_window_ms <= 60000),
	
	-- Session status
	status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'stopped', 'error')),
	is_connected BOOLEAN DEFAULT TRUE,
	connection_id VARCHAR(200),
	
	-- Processing metrics
	chunks_processed INTEGER DEFAULT 0 CHECK (chunks_processed >= 0),
	total_characters BIGINT DEFAULT 0 CHECK (total_characters >= 0),
	average_latency_ms DECIMAL(10,3) DEFAULT 0.0 CHECK (average_latency_ms >= 0.0),
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Streaming chunks for real-time processing
CREATE TABLE nlp.streaming_chunks (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	session_id UUID NOT NULL REFERENCES nlp.streaming_sessions(id),
	sequence_number INTEGER NOT NULL CHECK (sequence_number >= 0),
	
	-- Chunk content
	text_content TEXT NOT NULL CHECK (length(text_content) > 0),
	start_position INTEGER NOT NULL CHECK (start_position >= 0),
	end_position INTEGER NOT NULL CHECK (end_position >= 0),
	
	-- Processing metadata
	processing_time_ms DECIMAL(10,3) CHECK (processing_time_ms >= 0.0),
	confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
	
	-- Chunk results
	results JSONB,
	status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
	
	-- Chunk embedding for semantic continuity
	chunk_embedding vector(384),
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	processed_at TIMESTAMP WITH TIME ZONE
);

-- Annotation projects for collaborative annotation
CREATE TABLE nlp.annotation_projects (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	
	-- Project identity
	name VARCHAR(200) NOT NULL,
	description TEXT,
	
	-- Project configuration
	annotation_type VARCHAR(50) NOT NULL,
	annotation_schema JSONB NOT NULL,
	guidelines TEXT,
	
	-- Team and collaboration
	team_members UUID[] NOT NULL,
	project_manager UUID NOT NULL,
	consensus_threshold DECIMAL(3,2) DEFAULT 0.8 CHECK (consensus_threshold >= 0.5 AND consensus_threshold <= 1.0),
	
	-- Project status
	status VARCHAR(20) DEFAULT 'planning' CHECK (status IN ('planning', 'active', 'review', 'completed', 'archived')),
	is_training_enabled BOOLEAN DEFAULT FALSE,
	
	-- Document management
	document_count INTEGER DEFAULT 0 CHECK (document_count >= 0),
	completed_annotations INTEGER DEFAULT 0 CHECK (completed_annotations >= 0),
	
	-- Quality metrics
	inter_annotator_agreement DECIMAL(3,2) CHECK (inter_annotator_agreement >= 0.0 AND inter_annotator_agreement <= 1.0),
	average_annotation_time DECIMAL(10,3) CHECK (average_annotation_time >= 0.0),
	quality_score DECIMAL(3,2) DEFAULT 0.0 CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by UUID NOT NULL,
	is_deleted BOOLEAN DEFAULT FALSE,
	deleted_at TIMESTAMP WITH TIME ZONE
);

-- Text annotations with consensus tracking
CREATE TABLE nlp.text_annotations (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	project_id UUID NOT NULL REFERENCES nlp.annotation_projects(id),
	document_id UUID NOT NULL REFERENCES nlp.documents(id),
	annotator_id UUID NOT NULL,
	
	-- Annotation content
	start_position INTEGER NOT NULL CHECK (start_position >= 0),
	end_position INTEGER NOT NULL CHECK (end_position >= 0),
	annotated_text TEXT NOT NULL,
	annotation_value JSONB NOT NULL,
	
	-- Annotation metadata
	confidence DECIMAL(3,2) DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
	notes TEXT,
	time_spent_seconds DECIMAL(10,3) CHECK (time_spent_seconds >= 0.0),
	
	-- Consensus and quality
	consensus_score DECIMAL(3,2) CHECK (consensus_score >= 0.0 AND consensus_score <= 1.0),
	quality_score DECIMAL(3,2) DEFAULT 0.0 CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
	is_gold_standard BOOLEAN DEFAULT FALSE,
	
	-- Validation and review
	is_validated BOOLEAN DEFAULT FALSE,
	validation_feedback TEXT,
	validated_by UUID,
	validated_at TIMESTAMP WITH TIME ZONE,
	
	-- Annotation embedding for similarity analysis
	annotation_embedding vector(384),
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	is_deleted BOOLEAN DEFAULT FALSE,
	deleted_at TIMESTAMP WITH TIME ZONE
);

-- Text analytics and business intelligence
CREATE TABLE nlp.text_analytics (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	name VARCHAR(200) NOT NULL,
	
	-- Analytics configuration
	analysis_type VARCHAR(50) NOT NULL CHECK (analysis_type IN ('sentiment_trends', 'entity_analysis', 'topic_modeling', 'custom')),
	time_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
	time_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
	
	-- Data sources
	document_ids UUID[] NOT NULL,
	filter_criteria JSONB DEFAULT '{}',
	
	-- Analysis results (JSONB for flexible storage)
	insights JSONB DEFAULT '[]',
	trends JSONB DEFAULT '[]',
	predictions JSONB DEFAULT '[]',
	
	-- Quality and confidence
	confidence_score DECIMAL(3,2) DEFAULT 0.0 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
	data_quality_score DECIMAL(3,2) DEFAULT 0.0 CHECK (data_quality_score >= 0.0 AND data_quality_score <= 1.0),
	statistical_significance DECIMAL(3,2) CHECK (statistical_significance >= 0.0 AND statistical_significance <= 1.0),
	
	-- Processing metadata
	processing_time_seconds DECIMAL(10,3) NOT NULL CHECK (processing_time_seconds >= 0.0),
	model_versions JSONB DEFAULT '{}',
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by UUID NOT NULL
);

-- Model training configurations
CREATE TABLE nlp.model_training_configs (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	name VARCHAR(200) NOT NULL,
	
	-- Base model configuration
	base_model_id UUID NOT NULL REFERENCES nlp.models(id),
	target_task VARCHAR(50) NOT NULL,
	domain VARCHAR(100),
	
	-- Training data
	training_data_source VARCHAR(50) NOT NULL CHECK (training_data_source IN ('annotations', 'documents', 'external')),
	annotation_project_id UUID REFERENCES nlp.annotation_projects(id),
	training_document_ids UUID[] DEFAULT '{}',
	validation_split DECIMAL(3,2) DEFAULT 0.2 CHECK (validation_split >= 0.1 AND validation_split <= 0.5),
	
	-- Training parameters
	learning_rate DECIMAL(10,8) DEFAULT 0.001 CHECK (learning_rate > 0.0 AND learning_rate <= 1.0),
	batch_size INTEGER DEFAULT 32 CHECK (batch_size >= 1 AND batch_size <= 512),
	max_epochs INTEGER DEFAULT 10 CHECK (max_epochs >= 1 AND max_epochs <= 1000),
	early_stopping_patience INTEGER DEFAULT 5 CHECK (early_stopping_patience >= 1 AND early_stopping_patience <= 100),
	
	-- Resource configuration
	use_gpu BOOLEAN DEFAULT TRUE,
	max_memory_gb DECIMAL(10,2) CHECK (max_memory_gb > 0.0),
	parallel_workers INTEGER DEFAULT 1 CHECK (parallel_workers >= 1 AND parallel_workers <= 16),
	
	-- Training status
	status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'preparing', 'training', 'evaluating', 'completed', 'failed')),
	progress_percentage DECIMAL(5,2) DEFAULT 0.0 CHECK (progress_percentage >= 0.0 AND progress_percentage <= 100.0),
	
	-- Results and metrics
	final_accuracy DECIMAL(3,2) CHECK (final_accuracy >= 0.0 AND final_accuracy <= 1.0),
	training_loss DECIMAL(10,6) CHECK (training_loss >= 0.0),
	validation_loss DECIMAL(10,6) CHECK (validation_loss >= 0.0),
	training_time_seconds DECIMAL(10,3) CHECK (training_time_seconds >= 0.0),
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	started_at TIMESTAMP WITH TIME ZONE,
	completed_at TIMESTAMP WITH TIME ZONE,
	created_by UUID NOT NULL
);

-- System health monitoring
CREATE TABLE nlp.system_health (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID,  -- NULL for system-wide health checks
	
	-- System status
	overall_status VARCHAR(20) NOT NULL CHECK (overall_status IN ('healthy', 'degraded', 'unhealthy', 'maintenance')),
	component_status JSONB NOT NULL,
	
	-- Performance metrics
	average_response_time_ms DECIMAL(10,3) NOT NULL CHECK (average_response_time_ms >= 0.0),
	requests_per_minute INTEGER NOT NULL CHECK (requests_per_minute >= 0),
	active_sessions INTEGER NOT NULL CHECK (active_sessions >= 0),
	queue_depth INTEGER NOT NULL CHECK (queue_depth >= 0),
	
	-- Resource utilization
	cpu_usage_percent DECIMAL(5,2) NOT NULL CHECK (cpu_usage_percent >= 0.0 AND cpu_usage_percent <= 100.0),
	memory_usage_percent DECIMAL(5,2) NOT NULL CHECK (memory_usage_percent >= 0.0 AND memory_usage_percent <= 100.0),
	gpu_usage_percent DECIMAL(5,2) CHECK (gpu_usage_percent >= 0.0 AND gpu_usage_percent <= 100.0),
	disk_usage_percent DECIMAL(5,2) NOT NULL CHECK (disk_usage_percent >= 0.0 AND disk_usage_percent <= 100.0),
	
	-- Model status summary
	total_models INTEGER NOT NULL CHECK (total_models >= 0),
	active_models INTEGER NOT NULL CHECK (active_models >= 0),
	loaded_models INTEGER NOT NULL CHECK (loaded_models >= 0),
	failed_models INTEGER NOT NULL CHECK (failed_models >= 0),
	
	-- Error and alert information
	recent_errors JSONB DEFAULT '[]',
	active_alerts TEXT[] DEFAULT '{}',
	
	-- APG audit field
	timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance optimization

-- Documents indexes
CREATE INDEX idx_documents_tenant_id ON nlp.documents(tenant_id);
CREATE INDEX idx_documents_language ON nlp.documents(language);
CREATE INDEX idx_documents_content_type ON nlp.documents(content_type);
CREATE INDEX idx_documents_created_at ON nlp.documents(created_at DESC);
CREATE INDEX idx_documents_quality_score ON nlp.documents(quality_score DESC);
CREATE INDEX idx_documents_word_count ON nlp.documents(word_count DESC);
CREATE INDEX idx_documents_is_deleted ON nlp.documents(is_deleted) WHERE is_deleted = FALSE;

-- Full-text search index on document content
CREATE INDEX idx_documents_content_fts ON nlp.documents USING gin(to_tsvector('english', content));
CREATE INDEX idx_documents_title_fts ON nlp.documents USING gin(to_tsvector('english', title));

-- Vector similarity indexes for semantic search
CREATE INDEX idx_documents_content_embedding ON nlp.documents USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_documents_title_embedding ON nlp.documents USING ivfflat (title_embedding vector_cosine_ops) WITH (lists = 100);

-- Metadata JSONB indexes
CREATE INDEX idx_documents_metadata_gin ON nlp.documents USING gin(metadata);

-- Models indexes
CREATE INDEX idx_models_tenant_id ON nlp.models(tenant_id);
CREATE INDEX idx_models_provider ON nlp.models(provider);
CREATE INDEX idx_models_is_active ON nlp.models(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_models_health_status ON nlp.models(health_status);
CREATE INDEX idx_models_average_latency ON nlp.models(average_latency_ms ASC);
CREATE INDEX idx_models_accuracy_score ON nlp.models(accuracy_score DESC);
CREATE INDEX idx_models_supported_tasks ON nlp.models USING gin(supported_tasks);
CREATE INDEX idx_models_supported_languages ON nlp.models USING gin(supported_languages);

-- Processing requests indexes
CREATE INDEX idx_processing_requests_tenant_id ON nlp.processing_requests(tenant_id);
CREATE INDEX idx_processing_requests_user_id ON nlp.processing_requests(user_id);
CREATE INDEX idx_processing_requests_task_type ON nlp.processing_requests(task_type);
CREATE INDEX idx_processing_requests_status ON nlp.processing_requests(status);
CREATE INDEX idx_processing_requests_priority ON nlp.processing_requests(priority);
CREATE INDEX idx_processing_requests_created_at ON nlp.processing_requests(created_at DESC);
CREATE INDEX idx_processing_requests_document_id ON nlp.processing_requests(document_id);

-- Processing results indexes
CREATE INDEX idx_processing_results_request_id ON nlp.processing_results(request_id);
CREATE INDEX idx_processing_results_tenant_id ON nlp.processing_results(tenant_id);
CREATE INDEX idx_processing_results_task_type ON nlp.processing_results(task_type);
CREATE INDEX idx_processing_results_model_used ON nlp.processing_results(model_used);
CREATE INDEX idx_processing_results_processing_time ON nlp.processing_results(processing_time_ms ASC);
CREATE INDEX idx_processing_results_confidence_score ON nlp.processing_results(confidence_score DESC);
CREATE INDEX idx_processing_results_created_at ON nlp.processing_results(created_at DESC);

-- Vector index for result similarity
CREATE INDEX idx_processing_results_embedding ON nlp.processing_results USING ivfflat (result_embedding vector_cosine_ops) WITH (lists = 100);

-- Results JSONB index for flexible querying
CREATE INDEX idx_processing_results_results_gin ON nlp.processing_results USING gin(results);

-- Streaming sessions indexes
CREATE INDEX idx_streaming_sessions_tenant_id ON nlp.streaming_sessions(tenant_id);
CREATE INDEX idx_streaming_sessions_user_id ON nlp.streaming_sessions(user_id);
CREATE INDEX idx_streaming_sessions_status ON nlp.streaming_sessions(status);
CREATE INDEX idx_streaming_sessions_created_at ON nlp.streaming_sessions(created_at DESC);
CREATE INDEX idx_streaming_sessions_last_activity ON nlp.streaming_sessions(last_activity DESC);

-- Streaming chunks indexes
CREATE INDEX idx_streaming_chunks_session_id ON nlp.streaming_chunks(session_id);
CREATE INDEX idx_streaming_chunks_sequence ON nlp.streaming_chunks(session_id, sequence_number);
CREATE INDEX idx_streaming_chunks_status ON nlp.streaming_chunks(status);
CREATE INDEX idx_streaming_chunks_created_at ON nlp.streaming_chunks(created_at DESC);

-- Vector index for chunk embeddings
CREATE INDEX idx_streaming_chunks_embedding ON nlp.streaming_chunks USING ivfflat (chunk_embedding vector_cosine_ops) WITH (lists = 100);

-- Annotation projects indexes
CREATE INDEX idx_annotation_projects_tenant_id ON nlp.annotation_projects(tenant_id);
CREATE INDEX idx_annotation_projects_status ON nlp.annotation_projects(status);
CREATE INDEX idx_annotation_projects_created_by ON nlp.annotation_projects(created_by);
CREATE INDEX idx_annotation_projects_is_deleted ON nlp.annotation_projects(is_deleted) WHERE is_deleted = FALSE;

-- Team members GIN index for array searches
CREATE INDEX idx_annotation_projects_team_members ON nlp.annotation_projects USING gin(team_members);

-- Text annotations indexes
CREATE INDEX idx_text_annotations_project_id ON nlp.text_annotations(project_id);
CREATE INDEX idx_text_annotations_document_id ON nlp.text_annotations(document_id);
CREATE INDEX idx_text_annotations_annotator_id ON nlp.text_annotations(annotator_id);
CREATE INDEX idx_text_annotations_position ON nlp.text_annotations(document_id, start_position, end_position);
CREATE INDEX idx_text_annotations_quality_score ON nlp.text_annotations(quality_score DESC);
CREATE INDEX idx_text_annotations_is_validated ON nlp.text_annotations(is_validated);
CREATE INDEX idx_text_annotations_is_gold_standard ON nlp.text_annotations(is_gold_standard) WHERE is_gold_standard = TRUE;

-- Vector index for annotation embeddings
CREATE INDEX idx_text_annotations_embedding ON nlp.text_annotations USING ivfflat (annotation_embedding vector_cosine_ops) WITH (lists = 100);

-- Analytics indexes
CREATE INDEX idx_text_analytics_tenant_id ON nlp.text_analytics(tenant_id);
CREATE INDEX idx_text_analytics_analysis_type ON nlp.text_analytics(analysis_type);
CREATE INDEX idx_text_analytics_time_period ON nlp.text_analytics(time_period_start, time_period_end);
CREATE INDEX idx_text_analytics_created_by ON nlp.text_analytics(created_by);
CREATE INDEX idx_text_analytics_created_at ON nlp.text_analytics(created_at DESC);

-- Document IDs GIN index for array searches
CREATE INDEX idx_text_analytics_document_ids ON nlp.text_analytics USING gin(document_ids);

-- Model training configs indexes
CREATE INDEX idx_model_training_configs_tenant_id ON nlp.model_training_configs(tenant_id);
CREATE INDEX idx_model_training_configs_base_model_id ON nlp.model_training_configs(base_model_id);
CREATE INDEX idx_model_training_configs_status ON nlp.model_training_configs(status);
CREATE INDEX idx_model_training_configs_created_by ON nlp.model_training_configs(created_by);
CREATE INDEX idx_model_training_configs_annotation_project_id ON nlp.model_training_configs(annotation_project_id);

-- System health indexes
CREATE INDEX idx_system_health_tenant_id ON nlp.system_health(tenant_id);
CREATE INDEX idx_system_health_overall_status ON nlp.system_health(overall_status);
CREATE INDEX idx_system_health_timestamp ON nlp.system_health(timestamp DESC);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION nlp.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
	NEW.updated_at = CURRENT_TIMESTAMP;
	RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON nlp.documents
	FOR EACH ROW EXECUTE FUNCTION nlp.update_updated_at_column();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON nlp.models
	FOR EACH ROW EXECUTE FUNCTION nlp.update_updated_at_column();

CREATE TRIGGER update_annotation_projects_updated_at BEFORE UPDATE ON nlp.annotation_projects
	FOR EACH ROW EXECUTE FUNCTION nlp.update_updated_at_column();

CREATE TRIGGER update_text_annotations_updated_at BEFORE UPDATE ON nlp.text_annotations
	FOR EACH ROW EXECUTE FUNCTION nlp.update_updated_at_column();

-- Create function for soft delete support
CREATE OR REPLACE FUNCTION nlp.soft_delete_record()
RETURNS TRIGGER AS $$
BEGIN
	NEW.is_deleted = TRUE;
	NEW.deleted_at = CURRENT_TIMESTAMP;
	RETURN NEW;
END;
$$ language 'plpgsql';

-- Row-level security for multi-tenant isolation
ALTER TABLE nlp.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE nlp.models ENABLE ROW LEVEL SECURITY;
ALTER TABLE nlp.processing_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE nlp.processing_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE nlp.streaming_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE nlp.streaming_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE nlp.annotation_projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE nlp.text_annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE nlp.text_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE nlp.model_training_configs ENABLE ROW LEVEL SECURITY;

-- Example RLS policies (would be customized based on APG's auth system)
-- These policies ensure tenant isolation
CREATE POLICY nlp_documents_tenant_isolation ON nlp.documents
	FOR ALL USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY nlp_models_tenant_isolation ON nlp.models
	FOR ALL USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY nlp_processing_requests_tenant_isolation ON nlp.processing_requests
	FOR ALL USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY nlp_processing_results_tenant_isolation ON nlp.processing_results
	FOR ALL USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY nlp_streaming_sessions_tenant_isolation ON nlp.streaming_sessions
	FOR ALL USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY nlp_annotation_projects_tenant_isolation ON nlp.annotation_projects
	FOR ALL USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY nlp_text_analytics_tenant_isolation ON nlp.text_analytics
	FOR ALL USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY nlp_model_training_configs_tenant_isolation ON nlp.model_training_configs
	FOR ALL USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

-- Create materialized views for performance analytics
CREATE MATERIALIZED VIEW nlp.model_performance_summary AS
SELECT 
	m.id,
	m.name,
	m.provider,
	m.average_latency_ms,
	m.accuracy_score,
	m.total_requests,
	m.successful_requests,
	m.failed_requests,
	CASE 
		WHEN m.total_requests > 0 THEN (m.successful_requests::decimal / m.total_requests * 100)
		ELSE 0
	END as success_rate_percent,
	COUNT(pr.id) as recent_requests,
	AVG(pr.processing_time_ms) as recent_avg_latency,
	AVG(pr.confidence_score) as recent_avg_confidence
FROM nlp.models m
LEFT JOIN nlp.processing_results pr ON m.id = pr.model_used 
	AND pr.created_at >= NOW() - INTERVAL '24 hours'
WHERE m.is_deleted = FALSE
GROUP BY m.id, m.name, m.provider, m.average_latency_ms, m.accuracy_score, 
		 m.total_requests, m.successful_requests, m.failed_requests;

-- Create indexes on materialized view
CREATE INDEX idx_model_performance_summary_success_rate ON nlp.model_performance_summary(success_rate_percent DESC);
CREATE INDEX idx_model_performance_summary_recent_latency ON nlp.model_performance_summary(recent_avg_latency ASC);

-- Create refresh function for materialized views
CREATE OR REPLACE FUNCTION nlp.refresh_performance_views()
RETURNS void AS $$
BEGIN
	REFRESH MATERIALIZED VIEW nlp.model_performance_summary;
END;
$$ LANGUAGE plpgsql;

-- Create scheduled job to refresh performance views (requires pg_cron extension)
-- SELECT cron.schedule('refresh-nlp-views', '*/15 * * * *', 'SELECT nlp.refresh_performance_views()');

-- Grant permissions for APG roles (would be customized based on APG's permission system)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA nlp TO apg_nlp_users;
-- GRANT USAGE ON SCHEMA nlp TO apg_nlp_users;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA nlp TO apg_nlp_users;

-- Performance optimization settings
SET shared_preload_libraries = 'pg_stat_statements,vector';
SET max_parallel_workers_per_gather = 4;
SET effective_cache_size = '8GB';
SET random_page_cost = 1.1;
SET checkpoint_completion_target = 0.9;
SET wal_buffers = '16MB';
SET default_statistics_target = 100;

-- Vector search optimization
SET ivfflat.probes = 10;

COMMENT ON SCHEMA nlp IS 'APG Natural Language Processing capability database schema with multi-tenancy, vector search, and performance optimization';