-- APG GraphRAG Database Schema with Apache AGE Support
-- Copyright © 2025 Datacraft (nyimbi@gmail.com)
-- PostgreSQL + Apache AGE Graph Database Schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Install Apache AGE extension for graph database capabilities
CREATE EXTENSION IF NOT EXISTS age;

-- Load AGE into the search path
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Create the GraphRAG graph for Apache AGE operations
SELECT create_graph('graphrag_knowledge');

-- ============================================================================
-- CORE GRAPHRAG TABLES
-- ============================================================================

-- Knowledge Graphs - Top-level knowledge graph containers
CREATE TABLE gr_knowledge_graphs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    schema_version VARCHAR(20) DEFAULT '1.0.0',
    graph_type VARCHAR(100) DEFAULT 'knowledge_graph',
    metadata JSONB DEFAULT '{}',
    quality_metrics JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT gr_kg_tenant_name_unique UNIQUE (tenant_id, name),
    CONSTRAINT gr_kg_status_check CHECK (status IN ('active', 'inactive', 'building', 'error'))
);

-- Graph Entities - Nodes in the knowledge graph
CREATE TABLE gr_graph_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_graph_id UUID NOT NULL REFERENCES gr_knowledge_graphs(id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    entity_id VARCHAR(500) NOT NULL, -- Canonical entity identifier
    entity_type VARCHAR(100) NOT NULL,
    canonical_name VARCHAR(1000) NOT NULL,
    aliases TEXT[] DEFAULT '{}',
    properties JSONB DEFAULT '{}',
    embeddings vector(1024), -- bge-m3 embeddings
    confidence_score DECIMAL(5,4) DEFAULT 0.0000,
    evidence_sources TEXT[] DEFAULT '{}',
    provenance JSONB DEFAULT '{}',
    quality_score DECIMAL(5,4) DEFAULT 0.0000,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT gr_entities_kg_entity_unique UNIQUE (knowledge_graph_id, entity_id),
    CONSTRAINT gr_entities_confidence_check CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT gr_entities_quality_check CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    CONSTRAINT gr_entities_status_check CHECK (status IN ('active', 'inactive', 'pending', 'merged'))
);

-- Graph Relationships - Edges in the knowledge graph
CREATE TABLE gr_graph_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_graph_id UUID NOT NULL REFERENCES gr_knowledge_graphs(id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    relationship_id VARCHAR(500) NOT NULL, -- Canonical relationship identifier
    source_entity_id UUID NOT NULL REFERENCES gr_graph_entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES gr_graph_entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    strength DECIMAL(5,4) DEFAULT 0.0000,
    context JSONB DEFAULT '{}',
    properties JSONB DEFAULT '{}',
    evidence_sources TEXT[] DEFAULT '{}',
    provenance JSONB DEFAULT '{}',
    temporal_validity JSONB, -- Start/end timestamps for temporal relationships
    confidence_score DECIMAL(5,4) DEFAULT 0.0000,
    quality_score DECIMAL(5,4) DEFAULT 0.0000,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT gr_rel_kg_rel_unique UNIQUE (knowledge_graph_id, relationship_id),
    CONSTRAINT gr_rel_strength_check CHECK (strength >= 0.0 AND strength <= 1.0),
    CONSTRAINT gr_rel_confidence_check CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT gr_rel_quality_check CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    CONSTRAINT gr_rel_status_check CHECK (status IN ('active', 'inactive', 'pending', 'deprecated')),
    CONSTRAINT gr_rel_no_self_loop CHECK (source_entity_id != target_entity_id)
);

-- Graph Communities - Community detection results
CREATE TABLE gr_graph_communities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_graph_id UUID NOT NULL REFERENCES gr_knowledge_graphs(id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    community_id VARCHAR(500) NOT NULL,
    name VARCHAR(500),
    description TEXT,
    algorithm VARCHAR(100) NOT NULL, -- louvain, leiden, etc.
    members JSONB NOT NULL, -- Array of entity IDs
    centrality_metrics JSONB DEFAULT '{}',
    cohesion_score DECIMAL(5,4),
    size_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT gr_comm_kg_comm_unique UNIQUE (knowledge_graph_id, community_id),
    CONSTRAINT gr_comm_cohesion_check CHECK (cohesion_score >= 0.0 AND cohesion_score <= 1.0)
);

-- ============================================================================
-- GRAPHRAG QUERY & RESPONSE TABLES
-- ============================================================================

-- GraphRAG Queries - Query processing and caching
CREATE TABLE gr_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_graph_id UUID NOT NULL REFERENCES gr_knowledge_graphs(id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    query_text TEXT NOT NULL,
    query_type VARCHAR(100) DEFAULT 'question_answering',
    query_embedding vector(1024), -- bge-m3 query embedding
    context JSONB DEFAULT '{}',
    retrieval_config JSONB DEFAULT '{}',
    reasoning_config JSONB DEFAULT '{}',
    explanation_level VARCHAR(50) DEFAULT 'standard',
    max_hops INTEGER DEFAULT 3,
    status VARCHAR(50) DEFAULT 'pending',
    processing_time_ms BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT gr_queries_max_hops_check CHECK (max_hops >= 1 AND max_hops <= 10),
    CONSTRAINT gr_queries_status_check CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cached')),
    CONSTRAINT gr_queries_explanation_check CHECK (explanation_level IN ('minimal', 'standard', 'detailed', 'comprehensive'))
);

-- GraphRAG Responses - Generated responses with reasoning
CREATE TABLE gr_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID NOT NULL REFERENCES gr_queries(id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    answer TEXT NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    reasoning_chain JSONB NOT NULL, -- Complete reasoning process
    supporting_evidence JSONB DEFAULT '[]',
    graph_paths JSONB DEFAULT '[]', -- Paths through the graph
    entity_mentions JSONB DEFAULT '[]',
    source_attribution JSONB DEFAULT '[]',
    quality_indicators JSONB DEFAULT '{}',
    processing_metrics JSONB DEFAULT '{}',
    model_used VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT gr_resp_confidence_check CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0)
);

-- ============================================================================
-- KNOWLEDGE CURATION TABLES
-- ============================================================================

-- Curation Workflows - Collaborative knowledge improvement
CREATE TABLE gr_curation_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_graph_id UUID NOT NULL REFERENCES gr_knowledge_graphs(id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    workflow_type VARCHAR(100) NOT NULL,
    participants JSONB NOT NULL, -- Expert users and roles
    consensus_threshold DECIMAL(3,2) DEFAULT 0.80,
    status VARCHAR(50) DEFAULT 'active',
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT gr_curation_consensus_check CHECK (consensus_threshold >= 0.50 AND consensus_threshold <= 1.00),
    CONSTRAINT gr_curation_status_check CHECK (status IN ('active', 'paused', 'completed', 'archived'))
);

-- Knowledge Edits - Proposed changes to knowledge
CREATE TABLE gr_knowledge_edits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES gr_curation_workflows(id) ON DELETE CASCADE,
    knowledge_graph_id UUID NOT NULL REFERENCES gr_knowledge_graphs(id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    editor_id VARCHAR(255) NOT NULL, -- User making the edit
    edit_type VARCHAR(100) NOT NULL,
    target_type VARCHAR(100) NOT NULL, -- entity, relationship, graph
    target_id UUID NOT NULL,
    proposed_changes JSONB NOT NULL,
    justification TEXT,
    evidence JSONB DEFAULT '[]',
    status VARCHAR(50) DEFAULT 'pending',
    reviews JSONB DEFAULT '[]',
    consensus_score DECIMAL(5,4),
    applied_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT gr_edits_type_check CHECK (edit_type IN ('create', 'update', 'delete', 'merge', 'split')),
    CONSTRAINT gr_edits_target_check CHECK (target_type IN ('entity', 'relationship', 'community', 'graph')),
    CONSTRAINT gr_edits_status_check CHECK (status IN ('pending', 'reviewing', 'approved', 'rejected', 'applied')),
    CONSTRAINT gr_edits_consensus_check CHECK (consensus_score IS NULL OR (consensus_score >= 0.0 AND consensus_score <= 1.0))
);

-- ============================================================================
-- PERFORMANCE & ANALYTICS TABLES
-- ============================================================================

-- Graph Analytics - Performance and usage metrics
CREATE TABLE gr_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_graph_id UUID NOT NULL REFERENCES gr_knowledge_graphs(id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(200) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_data JSONB DEFAULT '{}',
    time_period VARCHAR(50), -- hour, day, week, month
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT gr_analytics_type_check CHECK (metric_type IN ('performance', 'usage', 'quality', 'accuracy', 'efficiency'))
);

-- Query Performance Logs - Detailed query performance tracking
CREATE TABLE gr_query_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID NOT NULL REFERENCES gr_queries(id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    retrieval_time_ms BIGINT,
    reasoning_time_ms BIGINT,
    generation_time_ms BIGINT,
    total_time_ms BIGINT,
    entities_retrieved INTEGER,
    relationships_traversed INTEGER,
    graph_hops INTEGER,
    memory_usage_mb INTEGER,
    cache_hits INTEGER,
    cache_misses INTEGER,
    model_tokens INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR HIGH PERFORMANCE
-- ============================================================================

-- Knowledge Graphs Indexes
CREATE INDEX idx_gr_kg_tenant ON gr_knowledge_graphs(tenant_id);
CREATE INDEX idx_gr_kg_status ON gr_knowledge_graphs(status);
CREATE INDEX idx_gr_kg_updated ON gr_knowledge_graphs(updated_at);

-- Entities Indexes
CREATE INDEX idx_gr_entities_kg ON gr_graph_entities(knowledge_graph_id);
CREATE INDEX idx_gr_entities_tenant ON gr_graph_entities(tenant_id);
CREATE INDEX idx_gr_entities_type ON gr_graph_entities(entity_type);
CREATE INDEX idx_gr_entities_name ON gr_graph_entities(canonical_name);
CREATE INDEX idx_gr_entities_confidence ON gr_graph_entities(confidence_score);
CREATE INDEX idx_gr_entities_status ON gr_graph_entities(status);
-- Vector similarity index for embeddings
CREATE INDEX idx_gr_entities_embeddings ON gr_graph_entities USING ivfflat (embeddings vector_cosine_ops);

-- Relationships Indexes
CREATE INDEX idx_gr_rel_kg ON gr_graph_relationships(knowledge_graph_id);
CREATE INDEX idx_gr_rel_tenant ON gr_graph_relationships(tenant_id);
CREATE INDEX idx_gr_rel_source ON gr_graph_relationships(source_entity_id);
CREATE INDEX idx_gr_rel_target ON gr_graph_relationships(target_entity_id);
CREATE INDEX idx_gr_rel_type ON gr_graph_relationships(relationship_type);
CREATE INDEX idx_gr_rel_strength ON gr_graph_relationships(strength);
CREATE INDEX idx_gr_rel_status ON gr_graph_relationships(status);
-- Compound index for graph traversal
CREATE INDEX idx_gr_rel_traversal ON gr_graph_relationships(source_entity_id, relationship_type, status);

-- Communities Indexes
CREATE INDEX idx_gr_comm_kg ON gr_graph_communities(knowledge_graph_id);
CREATE INDEX idx_gr_comm_tenant ON gr_graph_communities(tenant_id);
CREATE INDEX idx_gr_comm_algorithm ON gr_graph_communities(algorithm);

-- Queries Indexes
CREATE INDEX idx_gr_queries_kg ON gr_queries(knowledge_graph_id);
CREATE INDEX idx_gr_queries_tenant ON gr_queries(tenant_id);
CREATE INDEX idx_gr_queries_type ON gr_queries(query_type);
CREATE INDEX idx_gr_queries_status ON gr_queries(status);
CREATE INDEX idx_gr_queries_created ON gr_queries(created_at);
-- Vector similarity index for query embeddings
CREATE INDEX idx_gr_queries_embeddings ON gr_queries USING ivfflat (query_embedding vector_cosine_ops);

-- Responses Indexes
CREATE INDEX idx_gr_resp_query ON gr_responses(query_id);
CREATE INDEX idx_gr_resp_tenant ON gr_responses(tenant_id);
CREATE INDEX idx_gr_resp_confidence ON gr_responses(confidence_score);
CREATE INDEX idx_gr_resp_created ON gr_responses(created_at);

-- Performance Indexes
CREATE INDEX idx_gr_analytics_kg ON gr_analytics(knowledge_graph_id);
CREATE INDEX idx_gr_analytics_tenant ON gr_analytics(tenant_id);
CREATE INDEX idx_gr_analytics_type ON gr_analytics(metric_type);
CREATE INDEX idx_gr_analytics_created ON gr_analytics(created_at);

CREATE INDEX idx_gr_perf_query ON gr_query_performance(query_id);
CREATE INDEX idx_gr_perf_tenant ON gr_query_performance(tenant_id);
CREATE INDEX idx_gr_perf_total_time ON gr_query_performance(total_time_ms);
CREATE INDEX idx_gr_perf_created ON gr_query_performance(created_at);

-- ============================================================================
-- APACHE AGE GRAPH FUNCTIONS
-- ============================================================================

-- Function to create entity vertex in Apache AGE
CREATE OR REPLACE FUNCTION gr_create_entity_vertex(
    graph_name VARCHAR,
    entity_data JSONB
) RETURNS TABLE(vertex_id agtype) AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM cypher(graph_name, $$
        CREATE (e:Entity $entity_data)
        RETURN id(e)
    $$, entity_data) AS (vertex_id agtype);
END;
$$ LANGUAGE plpgsql;

-- Function to create relationship edge in Apache AGE
CREATE OR REPLACE FUNCTION gr_create_relationship_edge(
    graph_name VARCHAR,
    source_id BIGINT,
    target_id BIGINT,
    rel_type VARCHAR,
    rel_data JSONB
) RETURNS TABLE(edge_id agtype) AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM cypher(graph_name, $$
        MATCH (s) WHERE id(s) = $source_id
        MATCH (t) WHERE id(t) = $target_id
        CREATE (s)-[r:$rel_type $rel_data]->(t)
        RETURN id(r)
    $$, source_id, target_id, rel_type, rel_data) AS (edge_id agtype);
END;
$$ LANGUAGE plpgsql;

-- Function for multi-hop graph traversal
CREATE OR REPLACE FUNCTION gr_multi_hop_traversal(
    graph_name VARCHAR,
    start_entity_id BIGINT,
    max_hops INTEGER DEFAULT 3,
    relationship_types VARCHAR[] DEFAULT NULL
) RETURNS TABLE(path agtype) AS $$
DECLARE
    query_text TEXT;
BEGIN
    -- Build dynamic query based on parameters
    query_text := format('
        MATCH path = (start)-[*1..%s]-(end) 
        WHERE id(start) = $start_entity_id
        RETURN path
    ', max_hops);
    
    -- Add relationship type filtering if specified
    IF relationship_types IS NOT NULL THEN
        query_text := replace(query_text, '-[*', format('-[:%s*', array_to_string(relationship_types, '|')));
    END IF;
    
    RETURN QUERY
    SELECT * FROM cypher(graph_name, query_text, start_entity_id) AS (path agtype);
END;
$$ LANGUAGE plpgsql;

-- Function to find shortest path between entities
CREATE OR REPLACE FUNCTION gr_shortest_path(
    graph_name VARCHAR,
    source_id BIGINT,
    target_id BIGINT
) RETURNS TABLE(shortest_path agtype) AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM cypher(graph_name, $$
        MATCH (s), (t), path = shortestPath((s)-[*]-(t))
        WHERE id(s) = $source_id AND id(t) = $target_id
        RETURN path
    $$, source_id, target_id) AS (shortest_path agtype);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC MAINTENANCE
-- ============================================================================

-- Update timestamps trigger function
CREATE OR REPLACE FUNCTION gr_update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update timestamp triggers
CREATE TRIGGER gr_kg_update_timestamp 
    BEFORE UPDATE ON gr_knowledge_graphs
    FOR EACH ROW EXECUTE FUNCTION gr_update_timestamp();

CREATE TRIGGER gr_entities_update_timestamp 
    BEFORE UPDATE ON gr_graph_entities
    FOR EACH ROW EXECUTE FUNCTION gr_update_timestamp();

CREATE TRIGGER gr_rel_update_timestamp 
    BEFORE UPDATE ON gr_graph_relationships
    FOR EACH ROW EXECUTE FUNCTION gr_update_timestamp();

CREATE TRIGGER gr_comm_update_timestamp 
    BEFORE UPDATE ON gr_graph_communities
    FOR EACH ROW EXECUTE FUNCTION gr_update_timestamp();

CREATE TRIGGER gr_curation_update_timestamp 
    BEFORE UPDATE ON gr_curation_workflows
    FOR EACH ROW EXECUTE FUNCTION gr_update_timestamp();

-- ============================================================================
-- PARTITIONING FOR SCALABILITY (Optional - for large deployments)
-- ============================================================================

-- Partition analytics table by month for better performance
-- CREATE TABLE gr_analytics_y2025m01 PARTITION OF gr_analytics
--     FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- ============================================================================
-- INITIAL DATA AND SETUP
-- ============================================================================

-- Insert default entity types
INSERT INTO gr_knowledge_graphs (tenant_id, name, description, schema_version) VALUES
('system', 'default_entity_types', 'System-defined entity types', '1.0.0'),
('system', 'default_relationship_types', 'System-defined relationship types', '1.0.0');

COMMENT ON DATABASE current_database() IS 'APG GraphRAG Database with Apache AGE Integration';
COMMENT ON SCHEMA public IS 'APG GraphRAG tables and Apache AGE graph operations';

-- Performance optimization settings
ALTER SYSTEM SET shared_preload_libraries = 'age';
ALTER SYSTEM SET max_connections = '1000';
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET random_page_cost = '1.1';

SELECT pg_reload_conf();

-- Verify Apache AGE installation
SELECT * FROM ag_catalog.ag_graph WHERE name = 'graphrag_knowledge';

-- End of schema