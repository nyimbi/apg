-- APG Capability Registry - Database Initialization Script
-- Creates necessary extensions and initial configuration for PostgreSQL

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create custom types for enums (will be managed by Alembic, but defined here for reference)
-- These will be created by Alembic migrations

-- Create indexes for improved performance (will be created by Alembic)
-- Listed here for documentation purposes

-- Capability indexes:
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_capabilities_category ON capabilities(category);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_capabilities_status ON capabilities(status);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_capabilities_tenant ON capabilities(tenant_id);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_capabilities_search ON capabilities USING gin(to_tsvector('english', capability_name || ' ' || COALESCE(description, '')));
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_capabilities_quality ON capabilities(quality_score DESC) WHERE quality_score > 0;
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_capabilities_popularity ON capabilities(popularity_score DESC) WHERE popularity_score > 0;
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_capabilities_created ON capabilities(created_at DESC);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_capabilities_updated ON capabilities(updated_at DESC);

-- Composition indexes:
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compositions_type ON compositions(composition_type);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compositions_status ON compositions(validation_status);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compositions_tenant ON compositions(tenant_id);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compositions_template ON compositions(is_template) WHERE is_template = true;
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compositions_public ON compositions(is_public) WHERE is_public = true;

-- Dependency indexes:
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dependencies_dependent ON dependencies(dependent_id);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dependencies_dependency ON dependencies(dependency_id);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dependencies_type ON dependencies(dependency_type);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dependencies_tenant ON dependencies(tenant_id);

-- Version indexes:
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_versions_capability ON versions(capability_id);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_versions_number ON versions(version_number);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_versions_status ON versions(status);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_versions_tenant ON versions(tenant_id);

-- Analytics indexes:
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_analytics_capability ON usage_analytics(capability_id);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_analytics_timestamp ON usage_analytics(timestamp DESC);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_analytics_tenant ON usage_analytics(tenant_id);

-- Registry indexes:
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_registry_tenant ON registry(tenant_id);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_registry_namespace ON registry(namespace);

-- Create roles for different access levels
DO $$
BEGIN
    -- Read-only role for analytics and reporting
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'registry_readonly') THEN
        CREATE ROLE registry_readonly;
        COMMENT ON ROLE registry_readonly IS 'Read-only access to capability registry';
    END IF;
    
    -- Analytics role for metrics collection
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'registry_analytics') THEN
        CREATE ROLE registry_analytics;
        COMMENT ON ROLE registry_analytics IS 'Analytics access to capability registry';
    END IF;
    
    -- Backup role for database backups
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'registry_backup') THEN
        CREATE ROLE registry_backup;
        COMMENT ON ROLE registry_backup IS 'Backup access to capability registry';
    END IF;
END
$$;

-- Create schema for partitioning (if needed in the future)
CREATE SCHEMA IF NOT EXISTS partitions;
COMMENT ON SCHEMA partitions IS 'Schema for table partitions';

-- Create schema for temporary tables
CREATE SCHEMA IF NOT EXISTS temp_tables;
COMMENT ON SCHEMA temp_tables IS 'Schema for temporary tables and operations';

-- Set up row-level security (RLS) preparation
-- Note: RLS policies will be created by the application based on tenant requirements

-- Create custom functions for common operations
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

COMMENT ON FUNCTION update_updated_at_column() IS 'Automatically update updated_at timestamp';

-- Create function for generating capability codes
CREATE OR REPLACE FUNCTION generate_capability_code(capability_name TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Convert to uppercase, replace spaces with underscores, remove special characters
    RETURN UPPER(REGEXP_REPLACE(REGEXP_REPLACE(capability_name, '[^a-zA-Z0-9\s]', '', 'g'), '\s+', '_', 'g'));
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION generate_capability_code(TEXT) IS 'Generate standardized capability code from name';

-- Create function for capability search ranking
CREATE OR REPLACE FUNCTION capability_search_rank(
    search_query TEXT,
    capability_name TEXT,
    description TEXT,
    keywords TEXT[]
)
RETURNS REAL AS $$
DECLARE
    name_rank REAL := 0;
    desc_rank REAL := 0;
    keyword_rank REAL := 0;
BEGIN
    -- Calculate ranking based on different fields
    name_rank := ts_rank(to_tsvector('english', capability_name), plainto_tsquery('english', search_query));
    desc_rank := ts_rank(to_tsvector('english', COALESCE(description, '')), plainto_tsquery('english', search_query)) * 0.5;
    
    -- Check keyword matches
    IF keywords IS NOT NULL THEN
        SELECT COALESCE(
            (SELECT COUNT(*) FROM unnest(keywords) AS keyword WHERE keyword ILIKE '%' || search_query || '%') * 0.3,
            0
        ) INTO keyword_rank;
    END IF;
    
    RETURN name_rank + desc_rank + keyword_rank;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION capability_search_rank(TEXT, TEXT, TEXT, TEXT[]) IS 'Calculate search ranking for capabilities';

-- Create materialized view for capability statistics (will be refreshed periodically)
-- This will be created by migrations, but documented here
/*
CREATE MATERIALIZED VIEW capability_stats AS
SELECT 
    category,
    COUNT(*) as total_capabilities,
    AVG(quality_score) as avg_quality_score,
    AVG(popularity_score) as avg_popularity_score,
    COUNT(*) FILTER (WHERE status = 'active') as active_capabilities,
    MAX(created_at) as latest_capability
FROM capabilities
GROUP BY category;

CREATE UNIQUE INDEX ON capability_stats (category);
*/

-- Set up database maintenance
-- Create function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_capability_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY capability_stats;
    PERFORM pg_stat_reset_single_table_counters('public'::regclass, 'capabilities'::regclass);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_capability_stats() IS 'Refresh capability statistics materialized view';

-- Create function for cleanup old analytics data
CREATE OR REPLACE FUNCTION cleanup_old_analytics(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM usage_analytics 
    WHERE timestamp < CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_analytics(INTEGER) IS 'Clean up old analytics data beyond retention period';

-- Grant permissions to roles
-- Note: Specific table permissions will be granted by migrations

-- Set database parameters for optimal performance
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log slow queries
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';

-- Note: These settings require a PostgreSQL restart to take effect
-- They should be configured in postgresql.conf in production

-- Create comment on database
COMMENT ON DATABASE apg_capability_registry IS 'APG Capability Registry - Foundation infrastructure for capability discovery and orchestration';

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'APG Capability Registry database initialization completed successfully';
    RAISE NOTICE 'Extensions enabled: uuid-ossp, pg_trgm, btree_gin, btree_gist, pg_stat_statements';
    RAISE NOTICE 'Custom functions created for automated maintenance and search';
    RAISE NOTICE 'Database ready for Alembic migrations';
END $$;