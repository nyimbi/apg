-- Time & Attendance Database Schema Creation
-- Multi-tenant schema setup and basic structure
-- Copyright © 2025 Datacraft

-- Create core application schemas
CREATE SCHEMA IF NOT EXISTS ta_core;
CREATE SCHEMA IF NOT EXISTS ta_tenant_management;
CREATE SCHEMA IF NOT EXISTS ta_analytics;
CREATE SCHEMA IF NOT EXISTS ta_monitoring;

-- Set default search path
ALTER DATABASE time_attendance_db SET search_path TO ta_core, ta_tenant_management, public;

-- Create tenant registry table
CREATE TABLE IF NOT EXISTS ta_tenant_management.tenants (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    schema_name VARCHAR(63) NOT NULL UNIQUE,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    configuration JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT chk_tenant_status CHECK (status IN ('active', 'suspended', 'inactive')),
    CONSTRAINT chk_schema_name_format CHECK (schema_name ~ '^[a-z][a-z0-9_]*$')
);

-- Create indexes for tenant management
CREATE INDEX IF NOT EXISTS idx_tenants_status ON ta_tenant_management.tenants(status);
CREATE INDEX IF NOT EXISTS idx_tenants_schema_name ON ta_tenant_management.tenants(schema_name);
CREATE INDEX IF NOT EXISTS idx_tenants_created_at ON ta_tenant_management.tenants(created_at);

-- Insert default tenant for development
INSERT INTO ta_tenant_management.tenants (id, name, schema_name) 
VALUES ('default-tenant-id', 'Default Tenant', 'ta_default')
ON CONFLICT (id) DO NOTHING;

-- Create default tenant schema
CREATE SCHEMA IF NOT EXISTS ta_default;