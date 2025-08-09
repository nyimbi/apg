# APG Budgeting & Forecasting - Database Migration Guide

**Comprehensive guide for setting up and managing the APG Budgeting & Forecasting database schema.**

Version 1.0 | © 2025 Datacraft | Author: Nyimbi Odero

---

## Overview

The APG Budgeting & Forecasting capability uses a sophisticated multi-tenant database architecture built on PostgreSQL. This guide covers the complete migration process, from initial schema setup to tenant management and maintenance operations.

### Architecture Highlights

- **Multi-tenant design** with schema-based isolation
- **Row-level security (RLS)** for data protection
- **Comprehensive indexing** for high performance
- **APG platform integration** with audit trails
- **Automated tenant provisioning** with management functions

---

## Prerequisites

### System Requirements

```bash
# PostgreSQL 13+ with required extensions
PostgreSQL >= 13.0
- UUID extension (uuid-ossp)
- JSONB support
- Row-level security support

# Python environment
Python >= 3.9
asyncpg >= 0.25.0
pydantic >= 2.0.0
alembic >= 1.8.0  # Optional for Alembic integration
```

### Environment Variables

```bash
# Required
export DATABASE_URL="postgresql://user:password@host:port/database"

# Optional configuration
export BF_SCHEMA_NAME="bf_shared"          # Shared schema name
export BF_APP_ROLE="app_role"              # Application role
export BF_AUDIT_ROLE="audit_role"          # Audit role
export BF_CREATE_ROLES="true"              # Create database roles
export BF_INSERT_DEFAULT_DATA="true"       # Insert reference data
export BF_ENABLE_RLS="true"                # Enable row-level security
```

### Database Permissions

The migration user requires the following PostgreSQL permissions:

```sql
-- Grant schema creation and management permissions
GRANT CREATE ON DATABASE your_database TO migration_user;
GRANT CREATE ON SCHEMA public TO migration_user;

-- Grant role creation permissions (for app_role and audit_role)
GRANT CREATEROLE TO migration_user;

-- Grant function creation permissions
GRANT CREATE ON SCHEMA public TO migration_user;
```

---

## Migration Files

### Core Migration Files

1. **`migration_bf_complete_schema.py`** - Alembic-compatible migration script
2. **`run_migration.py`** - Standalone migration runner
3. **`MIGRATION_GUIDE.md`** - This documentation file

### Migration Components

```
Migration Structure:
├── Shared Schema (bf_shared)
│   ├── tenant_config           # Tenant metadata and configuration
│   ├── budget_templates        # Shared budget templates
│   ├── account_categories      # Chart of accounts reference
│   ├── currency_rates          # Exchange rate data
│   └── industry_benchmarks     # Anonymized benchmarking data
│
├── Tenant Schemas (bf_{tenant_id})
│   ├── budgets                 # Budget master records
│   ├── budget_lines           # Detailed budget line items
│   ├── forecasts              # Forecast master records
│   ├── forecast_data_points   # Individual forecast data points
│   ├── variance_analysis      # Budget vs actual analysis
│   └── scenarios              # Scenario planning data
│
├── Management Functions
│   ├── setup_tenant()         # Create new tenant schema
│   ├── create_tenant_rls_policies() # Row-level security setup
│   └── create_tenant_indexes() # Performance optimization
│
└── Security & Performance
    ├── Row-level security policies
    ├── Comprehensive indexing strategy
    └── Database roles and permissions
```

---

## Migration Execution

### Method 1: Standalone Migration Runner (Recommended)

The standalone runner provides complete control and detailed logging:

```bash
# 1. Initial schema setup
python run_migration.py --action upgrade

# 2. Create your first tenant
python run_migration.py --action create-tenant \
    --tenant-id "acme_corp" \
    --tenant-name "ACME Corporation"

# 3. List all tenants
python run_migration.py --action list-tenants
```

### Method 2: Alembic Integration

For integration with existing Alembic workflows:

```bash
# 1. Add to your Alembic environment
cp migration_bf_complete_schema.py alembic/versions/

# 2. Run migration
alembic upgrade bf001_complete_2025

# 3. Create tenants using the runner
python run_migration.py --action create-tenant \
    --tenant-id "new_tenant" \
    --tenant-name "New Tenant Name"
```

### Method 3: Direct SQL Execution

For advanced users who need direct control:

```sql
-- 1. Create shared schema
CREATE SCHEMA bf_shared;

-- 2. Execute schema creation from migration file
-- (Copy and execute the SQL from migration_bf_complete_schema.py)

-- 3. Create tenant using function
SELECT bf_shared.setup_tenant('tenant_id', 'Tenant Name', 'creator_user_id');
```

---

## Tenant Management

### Creating New Tenants

Each tenant gets a complete isolated schema with all necessary tables:

```bash
# Create tenant with full schema
python run_migration.py --action create-tenant \
    --tenant-id "global_inc" \
    --tenant-name "Global Inc" \
    --created-by "admin_user_123"
```

**What happens during tenant creation:**

1. **Schema Creation**: Creates `bf_global_inc` schema
2. **Table Creation**: All BF tables with proper constraints
3. **Index Creation**: Performance-optimized indexes
4. **RLS Setup**: Row-level security policies
5. **Configuration**: Entry in `bf_shared.tenant_config`

### Tenant Schema Structure

Each tenant schema contains these tables:

```sql
-- Core budget management
bf_{tenant_id}.budgets                 # Budget master records
bf_{tenant_id}.budget_lines           # Detailed budget lines

-- Forecasting capabilities  
bf_{tenant_id}.forecasts              # Forecast configurations
bf_{tenant_id}.forecast_data_points   # Time series forecast data

-- Analysis and reporting
bf_{tenant_id}.variance_analysis      # Budget vs actual analysis
bf_{tenant_id}.scenarios              # Scenario planning models
```

### Tenant Configuration

Tenant-specific settings are managed in `bf_shared.tenant_config`:

```sql
-- Example tenant configuration
SELECT * FROM bf_shared.tenant_config WHERE tenant_id = 'acme_corp';

-- Update tenant features
UPDATE bf_shared.tenant_config 
SET features_enabled = '["advanced_forecasting", "ai_insights", "scenario_planning"]'
WHERE tenant_id = 'acme_corp';
```

---

## Performance Optimization

### Index Strategy

The migration creates comprehensive indexes for optimal performance:

```sql
-- Budget performance indexes
CREATE INDEX CONCURRENTLY idx_{tenant}_budgets_tenant_fiscal 
    ON bf_{tenant}.budgets(tenant_id, fiscal_year);

CREATE INDEX CONCURRENTLY idx_{tenant}_budget_lines_period 
    ON bf_{tenant}.budget_lines(period_start, period_end);

-- Forecast performance indexes  
CREATE INDEX CONCURRENTLY idx_{tenant}_forecasts_tenant_type 
    ON bf_{tenant}.forecasts(tenant_id, forecast_type, status);

-- Analysis performance indexes
CREATE INDEX CONCURRENTLY idx_{tenant}_variance_significance 
    ON bf_{tenant}.variance_analysis(significance_level, requires_investigation);
```

### Query Optimization

**Recommended query patterns:**

```sql
-- Always include tenant_id in WHERE clauses
SELECT * FROM bf_acme.budgets 
WHERE tenant_id = 'acme_corp' AND fiscal_year = 2025;

-- Use period-based queries for time series data
SELECT * FROM bf_acme.forecast_data_points 
WHERE forecast_id = 'forecast_123' 
AND period_date BETWEEN '2025-01-01' AND '2025-12-31';

-- Leverage indexes for analysis queries
SELECT * FROM bf_acme.variance_analysis 
WHERE significance_level IN ('high', 'critical') 
AND requires_investigation = true;
```

---

## Security Implementation

### Row-Level Security (RLS)

Every tenant table has RLS policies for data isolation:

```sql
-- Tenant isolation policy
CREATE POLICY tenant_isolation_budgets ON bf_tenant.budgets 
FOR ALL TO app_role 
USING (tenant_id = current_setting('app.current_tenant'));

-- Audit access policy
CREATE POLICY audit_access_budgets ON bf_tenant.budgets 
FOR SELECT TO audit_role 
USING (true);
```

### Application Context

Set tenant context in your application:

```python
# Set tenant context for the session
await connection.execute("SET app.current_tenant = 'acme_corp'")

# All subsequent queries are automatically filtered by RLS
budgets = await connection.fetch("SELECT * FROM budgets WHERE fiscal_year = 2025")
```

### Database Roles

Two primary roles are created:

- **`app_role`**: Full CRUD access with RLS enforcement
- **`audit_role`**: Read-only access across all tenants

```sql
-- Grant application role to database user
GRANT app_role TO your_app_user;

-- Grant audit role to monitoring user
GRANT audit_role TO your_audit_user;
```

---

## Monitoring and Maintenance

### Health Checks

Monitor migration and tenant health:

```bash
# Check all tenants
python run_migration.py --action list-tenants

# Verify schema integrity
psql -d your_db -c "
SELECT schemaname, tablename, hasindexes 
FROM pg_tables 
WHERE schemaname LIKE 'bf_%' 
ORDER BY schemaname, tablename;
"
```

### Performance Monitoring

Key metrics to monitor:

```sql
-- Table sizes by tenant
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname LIKE 'bf_%'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index usage statistics
SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE schemaname LIKE 'bf_%'
ORDER BY idx_tup_read DESC;

-- Query performance
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements 
WHERE query LIKE '%bf_%'
ORDER BY mean_time DESC;
```

---

## Troubleshooting

### Common Issues

**1. Permission Denied Errors**

```bash
# Solution: Ensure proper database permissions
GRANT CREATE ON DATABASE your_db TO migration_user;
GRANT CREATEROLE TO migration_user;
```

**2. Schema Already Exists**

```bash
# Check existing schemas
psql -d your_db -c "\dn bf_*"

# Clean up if needed (WARNING: destroys data)
python run_migration.py --action downgrade --confirm-destroy
```

**3. Missing Dependencies**

```bash
# Install required packages
pip install asyncpg pydantic

# For Alembic integration
pip install alembic
```

**4. Connection Issues**

```bash
# Test connection
psql -d "$DATABASE_URL" -c "SELECT version();"

# Check SSL requirements
export DATABASE_URL="postgresql://user:pass@host:port/db?sslmode=require"
```

### Migration Recovery

**Partial Migration Failure:**

```bash
# Check migration status
python run_migration.py --action list-tenants

# Recreate failed tenant
python run_migration.py --action create-tenant \
    --tenant-id "failed_tenant" \
    --tenant-name "Failed Tenant"
```

**Complete Rollback:**

```bash
# WARNING: Destroys ALL data
python run_migration.py --action downgrade --confirm-destroy

# Clean restart
python run_migration.py --action upgrade
```

---

## Advanced Configuration

### Custom Tenant Features

Configure tenant-specific features:

```sql
-- Enable AI features for specific tenant
UPDATE bf_shared.tenant_config 
SET features_enabled = features_enabled || '["ai_forecasting"]'
WHERE tenant_id = 'tech_company';

-- Set custom limits
UPDATE bf_shared.tenant_config 
SET usage_limits = '{"max_budgets": 500, "max_forecast_horizon": 36}'
WHERE tenant_id = 'enterprise_client';
```

### Integration Settings

Configure APG platform integrations:

```sql
-- Configure APG capability integrations
UPDATE bf_shared.tenant_config 
SET apg_integrations = '{
    "auth_rbac": {"enabled": true, "endpoint": "https://auth.apg.platform"},
    "audit_compliance": {"enabled": true, "level": "detailed"},
    "ai_orchestration": {"enabled": true, "models": ["forecasting", "variance_analysis"]}
}'
WHERE tenant_id = 'integrated_tenant';
```

### Performance Tuning

Adjust performance settings per tenant:

```sql
-- High-performance tenant configuration
UPDATE bf_shared.tenant_config 
SET 
    max_budget_lines = 500000,
    concurrent_users_limit = 500,
    forecast_horizon_limit = 60
WHERE tenant_id = 'large_enterprise';
```

---

## Best Practices

### Migration Management

1. **Always backup before migration**
   ```bash
   pg_dump -Fc your_database > backup_$(date +%Y%m%d_%H%M%S).dump
   ```

2. **Test migrations in staging first**
   ```bash
   # Run on staging environment
   DATABASE_URL="postgresql://staging..." python run_migration.py --action upgrade
   ```

3. **Monitor during migration**
   ```bash
   # Run with verbose logging
   export BF_LOG_LEVEL="DEBUG"
   python run_migration.py --action upgrade
   ```

### Tenant Management

1. **Use descriptive tenant IDs**
   ```bash
   # Good: descriptive and unique
   --tenant-id "acme_corp_usa"
   
   # Avoid: generic or unclear
   --tenant-id "tenant1"
   ```

2. **Configure features at creation**
   ```sql
   -- Set up tenant immediately after creation
   UPDATE bf_shared.tenant_config 
   SET features_enabled = '["advanced_forecasting", "ai_insights"]'
   WHERE tenant_id = 'new_tenant';
   ```

3. **Regular maintenance**
   ```bash
   # Monthly health check
   python run_migration.py --action list-tenants
   
   # Monitor performance
   psql -f performance_check.sql
   ```

---

## Integration with APG Platform

The migration sets up complete integration with the APG platform ecosystem:

### APG Capability Integration Points

1. **auth_rbac**: User authentication and role-based permissions
2. **audit_compliance**: Complete audit trails and compliance reporting
3. **ai_orchestration**: ML model management and inference
4. **document_management**: Budget template and document storage
5. **notification_engine**: Workflow alerts and notifications

### APG Base Model Compliance

All models follow APG standards:

```python
# Every model includes APG integration fields
class BFBudget(APGBaseModel):
    # APG tenant isolation
    tenant_id: str = Field(..., description="APG tenant identifier")
    
    # APG audit compliance
    created_by: str = Field(..., description="User who created the record")
    updated_by: str = Field(..., description="User who last updated the record")
    
    # APG integration fields
    document_folder_id: Optional[str] = Field(None, description="Document management folder")
    workflow_instance_id: Optional[str] = Field(None, description="Workflow engine instance")
    ai_job_id: Optional[str] = Field(None, description="AI orchestration job")
```

---

## Support and Resources

### Documentation

- **APG Platform Documentation**: [APG Integration Analysis](apg_integration_analysis.md)
- **Multi-Tenant Architecture**: [Multi-Tenant Architecture Guide](multi_tenant_architecture.md)
- **Database Schema**: [Complete Schema Documentation](database_schema.sql)

### Contact Information

- **Support**: nyimbi@gmail.com
- **Website**: www.datacraft.co.ke
- **Repository**: [APG Capabilities Repository]

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-26 | Initial migration implementation |

---

© 2025 Datacraft. All rights reserved.  
Contact: nyimbi@gmail.com | www.datacraft.co.ke