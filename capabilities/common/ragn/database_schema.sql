-- APG RAG Capability Database Schema
-- PostgreSQL + pgvector + pgai optimized for enterprise RAG operations
-- Supports multi-tenant isolation, vector similarity search, and AI operations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pgai"; 
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ============================================================================
-- TENANT ISOLATION SCHEMA
-- ============================================================================

-- RAG Knowledge Bases - Multi-tenant knowledge base management with capability prefix support
CREATE TABLE apg_rag_knowledge_bases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    capability_id VARCHAR(100) NOT NULL, -- APG capability using this knowledge base
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Configuration
    embedding_model VARCHAR(100) DEFAULT 'bge-m3',
    generation_model VARCHAR(100) DEFAULT 'qwen3',
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 100,
    
    -- Vector configuration
    vector_dimensions INTEGER DEFAULT 1024,
    similarity_threshold REAL DEFAULT 0.7,
    
    -- Status and metadata
    status VARCHAR(50) DEFAULT 'active',
    document_count INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    last_indexed_at TIMESTAMP WITH TIME ZONE,
    
    -- APG integration
    apg_context JSONB DEFAULT '{}', -- Context data from other APG capabilities
    sharing_permissions JSONB DEFAULT '{}', -- Cross-capability sharing rules
    
    -- APG standard fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255),
    
    -- Multi-tenant and capability constraints
    CONSTRAINT uk_apg_rag_kb_tenant_cap_name UNIQUE (tenant_id, capability_id, name),
    CONSTRAINT ck_apg_rag_kb_status CHECK (status IN ('active', 'inactive', 'indexing', 'error')),
    CONSTRAINT ck_apg_rag_kb_vector_dims CHECK (vector_dimensions > 0),
    CONSTRAINT ck_apg_rag_kb_chunk_size CHECK (chunk_size > 0 AND chunk_size <= 8192)
);

-- RAG Documents - Document storage with metadata and versioning
CREATE TABLE apg_rag_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    knowledge_base_id UUID NOT NULL REFERENCES apg_rag_knowledge_bases(id) ON DELETE CASCADE,
    
    -- Document identification
    source_path TEXT NOT NULL,
    filename VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) NOT NULL, -- SHA-256 hash for deduplication
    content_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    
    -- Content and metadata
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    language VARCHAR(10) DEFAULT 'en',
    
    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending',
    processing_error TEXT,
    chunk_count INTEGER DEFAULT 0,
    
    -- Version control
    version INTEGER DEFAULT 1,
    parent_document_id UUID REFERENCES apg_rag_documents(id),
    
    -- APG standard fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255),
    
    -- Multi-tenant constraints
    CONSTRAINT uk_rg_doc_tenant_kb_hash UNIQUE (tenant_id, knowledge_base_id, file_hash),
    CONSTRAINT ck_rg_doc_status CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    CONSTRAINT ck_rg_doc_size CHECK (file_size > 0),
    CONSTRAINT ck_rg_doc_version CHECK (version > 0)
);

-- RAG Document Chunks - Chunked content with embeddings
CREATE TABLE apg_rag_document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    document_id UUID NOT NULL REFERENCES apg_rag_documents(id) ON DELETE CASCADE,
    knowledge_base_id UUID NOT NULL REFERENCES apg_rag_knowledge_bases(id) ON DELETE CASCADE,
    
    -- Chunk identification and content
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL, -- For deduplication
    
    -- Vector embeddings (bge-m3: 1024 dimensions)
    embedding vector(1024) NOT NULL,
    
    -- Chunk metadata and positioning
    start_position INTEGER,
    end_position INTEGER,
    token_count INTEGER,
    character_count INTEGER DEFAULT LENGTH(content),
    
    -- Hierarchical relationships
    parent_chunk_id UUID REFERENCES apg_rag_document_chunks(id),
    section_title TEXT,
    section_level INTEGER DEFAULT 0,
    
    -- Quality and confidence metrics
    embedding_confidence REAL DEFAULT 1.0,
    content_quality_score REAL DEFAULT 0.0,
    
    -- Processing metadata
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding_model VARCHAR(100) DEFAULT 'bge-m3',
    
    -- APG standard fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Multi-tenant constraints
    CONSTRAINT uk_rg_chunk_doc_index UNIQUE (document_id, chunk_index),
    CONSTRAINT ck_rg_chunk_positions CHECK (start_position <= end_position),
    CONSTRAINT ck_rg_chunk_tokens CHECK (token_count > 0),
    CONSTRAINT ck_rg_chunk_confidence CHECK (embedding_confidence >= 0.0 AND embedding_confidence <= 1.0),
    CONSTRAINT ck_rg_chunk_quality CHECK (content_quality_score >= 0.0 AND content_quality_score <= 1.0)
);

-- RAG Conversations - Multi-turn conversation management
CREATE TABLE apg_rag_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    knowledge_base_id UUID REFERENCES apg_rag_knowledge_bases(id) ON DELETE SET NULL,
    
    -- Conversation metadata
    title VARCHAR(500),
    description TEXT,
    context_summary TEXT,
    
    -- Configuration
    generation_model VARCHAR(100) DEFAULT 'qwen3',
    max_context_tokens INTEGER DEFAULT 4096,
    temperature REAL DEFAULT 0.7,
    
    -- State management
    status VARCHAR(50) DEFAULT 'active',
    turn_count INTEGER DEFAULT 0,
    total_tokens_used INTEGER DEFAULT 0,
    
    -- User and session info
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    
    -- APG standard fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255),
    
    -- Constraints
    CONSTRAINT ck_rg_conv_status CHECK (status IN ('active', 'completed', 'archived')),
    CONSTRAINT ck_rg_conv_temp CHECK (temperature >= 0.0 AND temperature <= 2.0),
    CONSTRAINT ck_rg_conv_tokens CHECK (max_context_tokens > 0)
);

-- RAG Conversation Turns - Individual conversation exchanges
CREATE TABLE apg_rag_conversation_turns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    conversation_id UUID NOT NULL REFERENCES apg_rag_conversations(id) ON DELETE CASCADE,
    
    -- Turn identification
    turn_number INTEGER NOT NULL,
    turn_type VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
    
    -- Content
    content TEXT NOT NULL,
    content_tokens INTEGER,
    
    -- RAG-specific data
    query_embedding vector(1024),
    retrieved_chunks UUID[] DEFAULT '{}', -- Array of chunk IDs
    retrieval_scores REAL[] DEFAULT '{}',
    
    -- Generation metadata
    model_used VARCHAR(100),
    generation_time_ms INTEGER,
    generation_tokens INTEGER,
    confidence_score REAL DEFAULT 0.0,
    
    -- Context and memory
    context_used TEXT,
    memory_summary TEXT,
    
    -- APG standard fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_rg_turn_conv_number UNIQUE (conversation_id, turn_number),
    CONSTRAINT ck_rg_turn_type CHECK (turn_type IN ('user', 'assistant', 'system')),
    CONSTRAINT ck_rg_turn_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT ck_rg_turn_tokens CHECK (content_tokens > 0)
);

-- RAG Retrieval Results - Store retrieval results for analysis
CREATE TABLE apg_rag_retrieval_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    conversation_turn_id UUID REFERENCES apg_rag_conversation_turns(id) ON DELETE CASCADE,
    
    -- Query information
    query_text TEXT NOT NULL,
    query_embedding vector(1024) NOT NULL,
    query_hash VARCHAR(64) NOT NULL,
    
    -- Retrieval configuration
    knowledge_base_id UUID NOT NULL REFERENCES apg_rag_knowledge_bases(id) ON DELETE CASCADE,
    k_retrievals INTEGER DEFAULT 10,
    similarity_threshold REAL DEFAULT 0.7,
    
    -- Results
    retrieved_chunk_ids UUID[] NOT NULL,
    similarity_scores REAL[] NOT NULL,
    retrieval_method VARCHAR(100) DEFAULT 'vector_similarity',
    
    -- Performance metrics
    retrieval_time_ms INTEGER NOT NULL,
    total_candidates INTEGER DEFAULT 0,
    
    -- Quality metrics
    result_quality_score REAL DEFAULT 0.0,
    diversity_score REAL DEFAULT 0.0,
    
    -- APG standard fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT ck_rg_retr_k CHECK (k_retrievals > 0),
    CONSTRAINT ck_rg_retr_threshold CHECK (similarity_threshold >= 0.0 AND similarity_threshold <= 1.0),
    CONSTRAINT ck_rg_retr_time CHECK (retrieval_time_ms >= 0),
    CONSTRAINT ck_rg_retr_quality CHECK (result_quality_score >= 0.0 AND result_quality_score <= 1.0)
);

-- RAG Generated Responses - Store generated responses with provenance
CREATE TABLE apg_rag_generated_responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    conversation_turn_id UUID NOT NULL REFERENCES apg_rag_conversation_turns(id) ON DELETE CASCADE,
    retrieval_result_id UUID REFERENCES apg_rag_retrieval_results(id) ON DELETE SET NULL,
    
    -- Generation input
    prompt TEXT NOT NULL,
    context_used TEXT,
    source_chunks UUID[] DEFAULT '{}',
    
    -- Generated content
    response_text TEXT NOT NULL,
    response_tokens INTEGER,
    
    -- Model and generation metadata
    generation_model VARCHAR(100) NOT NULL,
    model_parameters JSONB DEFAULT '{}',
    generation_time_ms INTEGER NOT NULL,
    
    -- Quality and attribution
    confidence_score REAL DEFAULT 0.0,
    factual_accuracy_score REAL DEFAULT 0.0,
    source_attribution JSONB DEFAULT '{}', -- Maps sources to text spans
    
    -- Validation status
    validation_status VARCHAR(50) DEFAULT 'pending',
    validation_feedback TEXT,
    
    -- APG standard fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT ck_rg_gen_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT ck_rg_gen_accuracy CHECK (factual_accuracy_score >= 0.0 AND factual_accuracy_score <= 1.0),
    CONSTRAINT ck_rg_gen_validation CHECK (validation_status IN ('pending', 'approved', 'rejected', 'needs_review')),
    CONSTRAINT ck_rg_gen_time CHECK (generation_time_ms >= 0),
    CONSTRAINT ck_rg_gen_tokens CHECK (response_tokens > 0)
);

-- ============================================================================
-- KNOWLEDGE GRAPH TABLES
-- ============================================================================

-- RAG Entities - Knowledge graph entities extracted from documents
CREATE TABLE apg_rag_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    knowledge_base_id UUID NOT NULL REFERENCES apg_rag_knowledge_bases(id) ON DELETE CASCADE,
    
    -- Entity identification
    name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL, -- PERSON, ORG, GPE, etc.
    normalized_name VARCHAR(500), -- Normalized for deduplication
    
    -- Entity embedding and similarity
    embedding vector(1024),
    
    -- Entity metadata
    description TEXT,
    properties JSONB DEFAULT '{}',
    confidence_score REAL DEFAULT 0.0,
    
    -- Frequency and importance
    mention_count INTEGER DEFAULT 1,
    importance_score REAL DEFAULT 0.0,
    
    -- Source tracking
    source_documents UUID[] DEFAULT '{}',
    first_mentioned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- APG standard fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_rg_entity_kb_norm_name UNIQUE (knowledge_base_id, normalized_name, entity_type),
    CONSTRAINT ck_rg_entity_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT ck_rg_entity_importance CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    CONSTRAINT ck_rg_entity_mentions CHECK (mention_count > 0)
);

-- RAG Relationships - Knowledge graph relationships between entities
CREATE TABLE apg_rag_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    knowledge_base_id UUID NOT NULL REFERENCES apg_rag_knowledge_bases(id) ON DELETE CASCADE,
    
    -- Relationship definition
    source_entity_id UUID NOT NULL REFERENCES apg_rag_entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES apg_rag_entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL, -- 'works_for', 'located_in', etc.
    
    -- Relationship metadata
    description TEXT,
    properties JSONB DEFAULT '{}',
    confidence_score REAL DEFAULT 0.0,
    
    -- Frequency and strength
    mention_count INTEGER DEFAULT 1,
    strength_score REAL DEFAULT 0.0,
    
    -- Source tracking
    source_chunks UUID[] DEFAULT '{}',
    evidence_text TEXT,
    
    -- APG standard fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_rg_rel_source_target_type UNIQUE (source_entity_id, target_entity_id, relationship_type),
    CONSTRAINT ck_rg_rel_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT ck_rg_rel_strength CHECK (strength_score >= 0.0 AND strength_score <= 1.0),
    CONSTRAINT ck_rg_rel_mentions CHECK (mention_count > 0),
    CONSTRAINT ck_rg_rel_not_self CHECK (source_entity_id != target_entity_id)
);

-- ============================================================================
-- PERFORMANCE AND MONITORING TABLES
-- ============================================================================

-- RAG Query Analytics - Track query performance and patterns
CREATE TABLE apg_rag_query_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Query information
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64) NOT NULL,
    query_type VARCHAR(50) NOT NULL, -- 'search', 'rag', 'chat'
    
    -- Performance metrics
    total_time_ms INTEGER NOT NULL,
    retrieval_time_ms INTEGER DEFAULT 0,
    generation_time_ms INTEGER DEFAULT 0,
    
    -- Results quality
    results_count INTEGER DEFAULT 0,
    user_satisfaction REAL,
    click_through_rate REAL,
    
    -- Context
    knowledge_base_id UUID REFERENCES apg_rag_knowledge_bases(id) ON DELETE SET NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    
    -- APG standard fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT ck_rg_analytics_time CHECK (total_time_ms >= 0),
    CONSTRAINT ck_rg_analytics_satisfaction CHECK (user_satisfaction IS NULL OR (user_satisfaction >= 0.0 AND user_satisfaction <= 1.0)),
    CONSTRAINT ck_rg_analytics_ctr CHECK (click_through_rate IS NULL OR (click_through_rate >= 0.0 AND click_through_rate <= 1.0))
);

-- RAG System Metrics - System performance tracking
CREATE TABLE apg_rag_system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Metric identification
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'counter', 'gauge', 'histogram'
    
    -- Metric values
    value_numeric REAL,
    value_text TEXT,
    
    -- Context and labels
    labels JSONB DEFAULT '{}',
    component VARCHAR(100), -- 'retrieval', 'generation', 'indexing'
    
    -- Timestamp
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT ck_rg_metrics_type CHECK (metric_type IN ('counter', 'gauge', 'histogram', 'summary'))
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Tenant isolation indexes
CREATE INDEX idx_rg_kb_tenant ON apg_rag_knowledge_bases(tenant_id);
CREATE INDEX idx_rg_docs_tenant ON apg_rag_documents(tenant_id);
CREATE INDEX idx_rg_chunks_tenant ON apg_rag_document_chunks(tenant_id);
CREATE INDEX idx_rg_conv_tenant ON apg_rag_conversations(tenant_id);

-- Vector similarity indexes (IVFFLAT for fast approximate search)
CREATE INDEX idx_rg_chunks_embedding ON apg_rag_document_chunks 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 1000);

CREATE INDEX idx_rg_turns_embedding ON apg_rag_conversation_turns 
    USING ivfflat (query_embedding vector_cosine_ops) 
    WITH (lists = 100);

CREATE INDEX idx_rg_retr_embedding ON apg_rag_retrieval_results 
    USING ivfflat (query_embedding vector_cosine_ops) 
    WITH (lists = 100);

CREATE INDEX idx_apg_rag_entities_embedding ON apg_rag_entities 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);

-- Full-text search indexes
CREATE INDEX idx_rg_docs_content_fts ON apg_rag_documents USING gin(to_tsvector('english', content));
CREATE INDEX idx_rg_chunks_content_fts ON apg_rag_document_chunks USING gin(to_tsvector('english', content));

-- Relationship and foreign key indexes
CREATE INDEX idx_rg_docs_kb ON apg_rag_documents(knowledge_base_id);
CREATE INDEX idx_rg_chunks_doc ON apg_rag_document_chunks(document_id);
CREATE INDEX idx_rg_chunks_kb ON apg_rag_document_chunks(knowledge_base_id);
CREATE INDEX idx_rg_turns_conv ON apg_rag_conversation_turns(conversation_id);
CREATE INDEX idx_rg_retr_kb ON apg_rag_retrieval_results(knowledge_base_id);

-- Performance indexes
CREATE INDEX idx_rg_docs_status ON apg_rag_documents(processing_status);
CREATE INDEX idx_rg_chunks_processed ON apg_rag_document_chunks(processed_at);
CREATE INDEX idx_rg_analytics_created ON apg_rag_query_analytics(created_at);
CREATE INDEX idx_rg_metrics_recorded ON apg_rag_system_metrics(recorded_at);

-- Hash and deduplication indexes
CREATE INDEX idx_rg_docs_hash ON apg_rag_documents(file_hash);
CREATE INDEX idx_rg_chunks_hash ON apg_rag_document_chunks(content_hash);
CREATE INDEX idx_rg_retr_query_hash ON apg_rag_retrieval_results(query_hash);

-- Composite indexes for common queries
CREATE INDEX idx_rg_chunks_kb_embedding ON apg_rag_document_chunks(knowledge_base_id, embedding);
CREATE INDEX idx_rg_conv_user_created ON apg_rag_conversations(user_id, created_at);
CREATE INDEX idx_rg_analytics_tenant_created ON apg_rag_query_analytics(tenant_id, created_at);

-- ============================================================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- ============================================================================

-- Knowledge base statistics
CREATE MATERIALIZED VIEW mv_rg_kb_stats AS
SELECT 
    kb.id as knowledge_base_id,
    kb.tenant_id,
    kb.name,
    COUNT(DISTINCT d.id) as document_count,
    COUNT(DISTINCT c.id) as chunk_count,
    AVG(c.embedding_confidence) as avg_embedding_confidence,
    MAX(d.updated_at) as last_document_update,
    SUM(d.file_size) as total_file_size
FROM apg_rag_knowledge_bases kb
LEFT JOIN apg_rag_documents d ON kb.id = d.knowledge_base_id AND d.processing_status = 'completed'
LEFT JOIN apg_rag_document_chunks c ON d.id = c.document_id
GROUP BY kb.id, kb.tenant_id, kb.name;

CREATE UNIQUE INDEX idx_mv_rg_kb_stats ON mv_rg_kb_stats(knowledge_base_id);
CREATE INDEX idx_mv_rg_kb_stats_tenant ON mv_rg_kb_stats(tenant_id);

-- Document processing summary
CREATE MATERIALIZED VIEW mv_rg_doc_processing AS
SELECT 
    tenant_id,
    knowledge_base_id,
    processing_status,
    COUNT(*) as document_count,
    AVG(chunk_count) as avg_chunks_per_doc,
    SUM(file_size) as total_size
FROM apg_rag_documents
GROUP BY tenant_id, knowledge_base_id, processing_status;

CREATE INDEX idx_mv_rg_doc_proc_tenant_kb ON mv_rg_doc_processing(tenant_id, knowledge_base_id);

-- ============================================================================
-- TRIGGERS FOR AUTOMATED MAINTENANCE
-- ============================================================================

-- Update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER trigger_rg_kb_updated_at BEFORE UPDATE ON apg_rag_knowledge_bases 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_rg_docs_updated_at BEFORE UPDATE ON apg_rag_documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_rg_chunks_updated_at BEFORE UPDATE ON apg_rag_document_chunks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update knowledge base statistics
CREATE OR REPLACE FUNCTION update_kb_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update document count and last indexed timestamp
    UPDATE apg_rag_knowledge_bases 
    SET 
        document_count = (
            SELECT COUNT(*) 
            FROM apg_rag_documents 
            WHERE knowledge_base_id = COALESCE(NEW.knowledge_base_id, OLD.knowledge_base_id)
            AND processing_status = 'completed'
        ),
        total_chunks = (
            SELECT COUNT(*) 
            FROM apg_rag_document_chunks 
            WHERE knowledge_base_id = COALESCE(NEW.knowledge_base_id, OLD.knowledge_base_id)
        ),
        last_indexed_at = NOW(),
        updated_at = NOW()
    WHERE id = COALESCE(NEW.knowledge_base_id, OLD.knowledge_base_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ language 'plpgsql';

CREATE TRIGGER trigger_rg_docs_kb_stats AFTER INSERT OR UPDATE OR DELETE ON apg_rag_documents 
    FOR EACH ROW EXECUTE FUNCTION update_kb_stats();

CREATE TRIGGER trigger_rg_chunks_kb_stats AFTER INSERT OR UPDATE OR DELETE ON apg_rag_document_chunks 
    FOR EACH ROW EXECUTE FUNCTION update_kb_stats();

-- ============================================================================
-- FUNCTIONS FOR RAG OPERATIONS
-- ============================================================================

-- Vector similarity search function
CREATE OR REPLACE FUNCTION rg_vector_search(
    p_tenant_id VARCHAR(255),
    p_knowledge_base_id UUID,
    p_query_embedding vector(1024),
    p_limit INTEGER DEFAULT 10,
    p_threshold REAL DEFAULT 0.7
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity_score REAL,
    document_title TEXT,
    chunk_index INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.document_id,
        c.content,
        1 - (c.embedding <=> p_query_embedding) as similarity,
        d.title,
        c.chunk_index
    FROM apg_rag_document_chunks c
    JOIN apg_rag_documents d ON c.document_id = d.id
    WHERE c.tenant_id = p_tenant_id
        AND c.knowledge_base_id = p_knowledge_base_id
        AND d.processing_status = 'completed'
        AND 1 - (c.embedding <=> p_query_embedding) >= p_threshold
    ORDER BY c.embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Hybrid search combining vector and text search
CREATE OR REPLACE FUNCTION rg_hybrid_search(
    p_tenant_id VARCHAR(255),
    p_knowledge_base_id UUID,
    p_query_text TEXT,
    p_query_embedding vector(1024),
    p_limit INTEGER DEFAULT 10,
    p_vector_weight REAL DEFAULT 0.7,
    p_text_weight REAL DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score REAL,
    vector_score REAL,
    text_score REAL
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            c.id,
            c.document_id,
            c.content,
            1 - (c.embedding <=> p_query_embedding) as v_score
        FROM apg_rag_document_chunks c
        JOIN apg_rag_documents d ON c.document_id = d.id
        WHERE c.tenant_id = p_tenant_id
            AND c.knowledge_base_id = p_knowledge_base_id
            AND d.processing_status = 'completed'
    ),
    text_results AS (
        SELECT 
            c.id,
            c.document_id,
            c.content,
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', p_query_text)) as t_score
        FROM apg_rag_document_chunks c
        JOIN apg_rag_documents d ON c.document_id = d.id
        WHERE c.tenant_id = p_tenant_id
            AND c.knowledge_base_id = p_knowledge_base_id
            AND d.processing_status = 'completed'
            AND to_tsvector('english', c.content) @@ plainto_tsquery('english', p_query_text)
    )
    SELECT 
        COALESCE(vr.id, tr.id),
        COALESCE(vr.document_id, tr.document_id),
        COALESCE(vr.content, tr.content),
        (COALESCE(vr.v_score, 0) * p_vector_weight + COALESCE(tr.t_score, 0) * p_text_weight) as combined,
        COALESCE(vr.v_score, 0) as vector,
        COALESCE(tr.t_score, 0) as text
    FROM vector_results vr
    FULL OUTER JOIN text_results tr ON vr.id = tr.id
    ORDER BY combined DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Knowledge graph entity search
CREATE OR REPLACE FUNCTION rg_entity_search(
    p_tenant_id VARCHAR(255),
    p_knowledge_base_id UUID,
    p_entity_name VARCHAR(500),
    p_entity_type VARCHAR(100) DEFAULT NULL
)
RETURNS TABLE (
    entity_id UUID,
    entity_name VARCHAR(500),
    entity_type VARCHAR(100),
    confidence_score REAL,
    mention_count INTEGER,
    related_documents UUID[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id,
        e.name,
        e.entity_type,
        e.confidence_score,
        e.mention_count,
        e.source_documents
    FROM apg_rag_entities e
    WHERE e.tenant_id = p_tenant_id
        AND e.knowledge_base_id = p_knowledge_base_id
        AND (p_entity_type IS NULL OR e.entity_type = p_entity_type)
        AND (e.name ILIKE '%' || p_entity_name || '%' 
             OR e.normalized_name ILIKE '%' || p_entity_name || '%')
    ORDER BY e.importance_score DESC, e.mention_count DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) FOR MULTI-TENANT ISOLATION
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE apg_rag_knowledge_bases ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_rag_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_rag_document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_rag_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_rag_conversation_turns ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_rag_retrieval_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_rag_generated_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_rag_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_rag_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_rag_query_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE apg_rag_system_metrics ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (will be activated when APG auth is integrated)
-- For now, these are commented out as they require APG auth context

/*
-- Example RLS policies for tenant isolation
CREATE POLICY rls_rg_kb_tenant ON apg_rag_knowledge_bases
    USING (tenant_id = current_setting('apg.current_tenant_id'));

CREATE POLICY rls_rg_docs_tenant ON apg_rag_documents
    USING (tenant_id = current_setting('apg.current_tenant_id'));
    
-- Additional policies would be created for all tables...
*/

-- ============================================================================
-- INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- Insert default configuration values that can be referenced by the application
INSERT INTO apg_rag_system_metrics (tenant_id, metric_name, metric_type, value_text, component, recorded_at)
VALUES 
    ('system', 'schema_version', 'gauge', '1.0.0', 'database', NOW()),
    ('system', 'pgvector_version', 'gauge', 'installed', 'database', NOW()),
    ('system', 'pgai_version', 'gauge', 'installed', 'database', NOW());

-- Grant necessary permissions (adjust based on APG user setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO apg_rag_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO apg_rag_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO apg_rag_user;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE apg_rag_knowledge_bases IS 'Multi-tenant knowledge base management with vector configuration';
COMMENT ON TABLE apg_rag_documents IS 'Document storage with metadata, versioning, and processing status';
COMMENT ON TABLE apg_rag_document_chunks IS 'Chunked content with bge-m3 embeddings for vector similarity search';
COMMENT ON TABLE apg_rag_conversations IS 'Multi-turn conversation management with context persistence';
COMMENT ON TABLE apg_rag_conversation_turns IS 'Individual conversation exchanges with RAG context';
COMMENT ON TABLE apg_rag_retrieval_results IS 'Retrieval results storage for analysis and optimization';
COMMENT ON TABLE apg_rag_generated_responses IS 'Generated responses with provenance and quality tracking';
COMMENT ON TABLE apg_rag_entities IS 'Knowledge graph entities extracted from documents';
COMMENT ON TABLE apg_rag_relationships IS 'Knowledge graph relationships with confidence scoring';
COMMENT ON TABLE apg_rag_query_analytics IS 'Query performance tracking and pattern analysis';
COMMENT ON TABLE apg_rag_system_metrics IS 'System performance and health metrics';

COMMENT ON FUNCTION rg_vector_search IS 'Vector similarity search using pgvector cosine distance';
COMMENT ON FUNCTION rg_hybrid_search IS 'Hybrid search combining vector similarity and text search';
COMMENT ON FUNCTION rg_entity_search IS 'Knowledge graph entity search with fuzzy matching';

-- Schema creation completed successfully
SELECT 'APG RAG Database Schema v1.0.0 created successfully' AS result;