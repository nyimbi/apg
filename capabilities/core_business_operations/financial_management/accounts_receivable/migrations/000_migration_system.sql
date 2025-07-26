--
-- APG Accounts Receivable - Migration System Setup
-- Migration tracking and management system for APG multi-tenant architecture
--
-- © 2025 Datacraft. All rights reserved.
-- Author: Nyimbi Odero <nyimbi@gmail.com>
--

-- Create migration tracking table if it doesn't exist
CREATE TABLE IF NOT EXISTS apg_schema_migrations (
	id SERIAL PRIMARY KEY,
	schema_name VARCHAR(100) NOT NULL,
	migration_version VARCHAR(50) NOT NULL,
	migration_name VARCHAR(255),
	applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	applied_by VARCHAR(255) DEFAULT CURRENT_USER,
	execution_time_ms INTEGER,
	checksum VARCHAR(64),
	
	CONSTRAINT apg_migrations_unique UNIQUE (schema_name, migration_version)
);

-- Create migration log table for detailed tracking
CREATE TABLE IF NOT EXISTS apg_migration_log (
	id SERIAL PRIMARY KEY,
	migration_id INTEGER REFERENCES apg_schema_migrations(id),
	log_level VARCHAR(20) NOT NULL DEFAULT 'INFO',
	message TEXT NOT NULL,
	logged_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	
	CONSTRAINT migration_log_level_check CHECK (log_level IN ('DEBUG', 'INFO', 'WARN', 'ERROR'))
);

-- Function to log migration steps
CREATE OR REPLACE FUNCTION log_migration_step(
	p_migration_id INTEGER,
	p_level VARCHAR(20),
	p_message TEXT
) RETURNS VOID AS $$
BEGIN
	INSERT INTO apg_migration_log (migration_id, log_level, message)
	VALUES (p_migration_id, p_level, p_message);
END;
$$ LANGUAGE plpgsql;

-- Function to check if migration has been applied
CREATE OR REPLACE FUNCTION migration_applied(
	p_schema_name VARCHAR(100),
	p_migration_version VARCHAR(50)
) RETURNS BOOLEAN AS $$
BEGIN
	RETURN EXISTS (
		SELECT 1 FROM apg_schema_migrations 
		WHERE schema_name = p_schema_name 
		AND migration_version = p_migration_version
	);
END;
$$ LANGUAGE plpgsql;

-- Function to record migration application
CREATE OR REPLACE FUNCTION record_migration(
	p_schema_name VARCHAR(100),
	p_migration_version VARCHAR(50),
	p_migration_name VARCHAR(255) DEFAULT NULL,
	p_execution_time_ms INTEGER DEFAULT NULL
) RETURNS INTEGER AS $$
DECLARE
	migration_id INTEGER;
BEGIN
	INSERT INTO apg_schema_migrations (
		schema_name, 
		migration_version, 
		migration_name,
		execution_time_ms
	)
	VALUES (
		p_schema_name, 
		p_migration_version, 
		p_migration_name,
		p_execution_time_ms
	)
	RETURNING id INTO migration_id;
	
	PERFORM log_migration_step(
		migration_id, 
		'INFO', 
		format('Migration %s.%s applied successfully', p_schema_name, p_migration_version)
	);
	
	RETURN migration_id;
END;
$$ LANGUAGE plpgsql;

-- Function to validate migration prerequisites
CREATE OR REPLACE FUNCTION validate_migration_prerequisites(
	p_schema_name VARCHAR(100),
	p_required_migrations VARCHAR(50)[]
) RETURNS BOOLEAN AS $$
DECLARE
	required_migration VARCHAR(50);
BEGIN
	-- Check that all required migrations have been applied
	FOREACH required_migration IN ARRAY p_required_migrations
	LOOP
		IF NOT migration_applied(p_schema_name, required_migration) THEN
			RAISE EXCEPTION 'Required migration % has not been applied', required_migration;
			RETURN FALSE;
		END IF;
	END LOOP;
	
	RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to get pending migrations
CREATE OR REPLACE FUNCTION get_pending_migrations(p_schema_name VARCHAR(100))
RETURNS TABLE(migration_version VARCHAR(50), migration_name VARCHAR(255)) AS $$
BEGIN
	-- This is a placeholder - in a real implementation, this would
	-- scan the migrations directory and compare with applied migrations
	RETURN QUERY
	SELECT 
		'pending'::VARCHAR(50) as migration_version,
		'No pending migrations found'::VARCHAR(255) as migration_name
	WHERE FALSE; -- Return empty set for now
END;
$$ LANGUAGE plpgsql;

-- Create indexes for migration tracking performance
CREATE INDEX IF NOT EXISTS idx_schema_migrations_schema_version 
	ON apg_schema_migrations(schema_name, migration_version);
CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at 
	ON apg_schema_migrations(applied_at DESC);
CREATE INDEX IF NOT EXISTS idx_migration_log_migration_id 
	ON apg_migration_log(migration_id);

-- Record the migration system setup
INSERT INTO apg_schema_migrations (
	schema_name, 
	migration_version, 
	migration_name
) VALUES (
	'apg_migration_system', 
	'000_migration_system', 
	'Migration tracking system setup'
) ON CONFLICT DO NOTHING;