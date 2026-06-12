# APG Multi-Tenancy Capability

**Enterprise multi-tenancy framework providing tenant isolation, management, and context switching for the APG platform.**

## Overview

The Multi-Tenancy capability (`mten`) is a critical enterprise infrastructure capability that provides comprehensive multi-tenant architecture support for the APG platform. It enables secure tenant isolation, resource management, and context-aware operations across all system components.

## Current Executable Package

The package-backed MTEN slice is dependency-light and can be composed by
generated APG applications without starting cloud providers, DNS services,
deployment bundles, analytics engines, AI providers, web servers, or production
databases. The local runtime is `mten_runtime.MtenService`; generated
applications should use `api_helpers.py` and `view_models.py` for stable
package surfaces.

It currently provides:

- tenant-qualified tenant registration and activation;
- DNS, isolation, capacity, suspension, reactivation, and live migration
  guardrails;
- first-class tenant agent registration for `codex`, `claude_code`,
  `opencode`, and `pi`;
- role and approval guardrails for tenant agents;
- Bytewax lifecycle-stream metadata and batch validation guardrails;
- durable review evidence for capacity approvals, live migrations, privileged
  tenant agents, denied lifecycle batches, and governance events;
- UI/view-model surfaces for tenant portfolio, provisioning, capacity,
  isolation, migration, agents, governance, analytics, optimization, and
  settings.

### Dependency-Light Usage

```python
from capabilities.common.mten.mten_runtime import MtenService

service = MtenService()

service.validate_lifecycle_batch(
    tenant_id="platform",
    record_count=25,
    event_stream="bytewax",
)

agent = service.register_tenant_agent(
    agent_id="agent-001",
    tenant_id="platform",
    name="Tenant Capacity Reviewer",
    runtime="codex",
    role="capacity_reviewer",
    purpose="Review tenant overcommit and capacity approvals.",
    owner="capacity-reviewer",
)

tenant = service.register_tenant(
    target_tenant_id="tenant-alpha",
    tenant_id="platform",
    name="tenant-alpha",
    owner="tenant-owner",
    tier="enterprise",
    primary_domain="alpha.example.com",
    projected_compute_units=900,
)
```

Use `service.describe("platform")` to inspect configuration, rules, UI routes,
theme tokens, tenant-agent metadata, and Bytewax stream metadata.

### Review Evidence

MTEN records `policy_decision`, `matched_rules`, `review_reasons`, and
`governance_evidence` on executable tenant-control records. Generated
applications can compose review queues through:

```python
service.list_pending_reviews("platform")
service.list_lifecycle_batches("platform")
```

Privileged tenant agents without human approval are retained as
`pending_review` records. Denied non-Bytewax lifecycle batches are retained as
`denied` evidence before `PermissionError` is raised.

## Features

### ✅ Core Features
- **Complete Tenant Isolation**: Secure data and resource separation between tenants
- **Hierarchical Tenant Management**: Full tenant lifecycle from creation to archival
- **5-Tier Subscription Model**: Free, Basic, Premium, Enterprise, and Custom tiers
- **Resource Limit Enforcement**: Configurable limits per tenant tier with usage tracking
- **Context-Aware Operations**: Thread-local tenant context for all operations
- **Database Persistence**: SQLite-based tenant data persistence with querying
- **Feature Flag Management**: Per-tenant feature enablement and configuration
- **Audit Integration**: Complete audit trail for all tenant operations

### 🔐 Security & Isolation
- **Data Isolation**: Complete separation of tenant data and configurations
- **Access Control**: Tenant-scoped access controls and authorization
- **Resource Boundaries**: Strict enforcement of tenant resource limits
- **Context Security**: Secure tenant context switching and validation
- **Audit Trail**: Complete logging of all tenant management operations

### 📊 Management & Analytics
- **Usage Tracking**: Real-time monitoring of tenant resource usage
- **Billing Integration**: Support for subscription billing and tier management
- **Statistics & Reporting**: Comprehensive tenant analytics and reporting
- **Lifecycle Management**: Complete tenant lifecycle from provisioning to cleanup
- **Tier Management**: Dynamic tenant tier upgrades and feature management

## Architecture Components

- **TenantManager** - Core tenant management engine with in-memory operations
- **TenantService** - Business logic layer with validation and tier management
- **PersistentTenantService** - Database-backed service with querying capabilities
- **TenantContext** - Thread-local context management for tenant operations
- **REST API** - Complete HTTP API for tenant management operations
- **Data Models** - Pydantic models for validation and serialization

## Installation & Setup

### Prerequisites
- Python 3.9+
- AsyncIO support
- SQLite3 (included with Python)
- Pydantic for data validation
- aiosqlite for async database operations

### Basic Setup

```python
from apg.capabilities.common.mten import (
    init_multi_tenancy, get_current_tenant, set_current_tenant, tenant_context
)

# Initialize multi-tenancy system
tenant_service = await init_multi_tenancy()

# Create a new tenant
tenant = await tenant_service.create_tenant(
    name="Acme Corporation",
    slug="acme-corp",
    email="admin@acme-corp.com",
    tier=TenantTier.PREMIUM
)
print(f"Created tenant: {tenant.id}")

# Set tenant context for operations
set_current_tenant(tenant.id)

# All subsequent operations will be tenant-aware
current_tenant_id = get_current_tenant()
print(f"Current tenant: {current_tenant_id}")
```

### Advanced Setup with Context Management

```python
from apg.capabilities.common.mten import tenant_context, TenantTier
from apg.capabilities.common.mten.service import get_persistent_tenant_service

# Initialize persistent service
service = await init_persistent_tenant_service()

# Create multiple tenants
tenant_a = await service.create_tenant("Company A", "company-a", "admin@a.com", TenantTier.ENTERPRISE)
tenant_b = await service.create_tenant("Company B", "company-b", "admin@b.com", TenantTier.BASIC)

# Use context manager for tenant-scoped operations
async with tenant_context(tenant_a.id):
    # All operations here are scoped to tenant A
    await perform_tenant_specific_operations()
    
async with tenant_context(tenant_b.id):
    # All operations here are scoped to tenant B
    await perform_other_tenant_operations()
```

## Tenant Tiers & Features

### Tier Structure

| Tier | Users | Storage | API Calls/Month | Features |
|------|-------|---------|-----------------|----------|
| **Free** | 5 | 1GB | 1,000 | Basic features |
| **Basic** | 25 | 10GB | 10,000 | Email support, data export |
| **Premium** | 100 | 100GB | 100,000 | Advanced analytics, integrations |
| **Enterprise** | Unlimited | Unlimited | Unlimited | SSO, audit logs, custom branding |
| **Custom** | Configurable | Configurable | Configurable | Custom feature set |

### Free Tier
```python
limits = {
    "users": 5,
    "storage_gb": 1,
    "api_calls_per_month": 1000,
    "concurrent_sessions": 2
}
features = {"basic_features"}
```

### Enterprise Tier
```python
limits = {
    "users": -1,  # unlimited
    "storage_gb": -1,
    "api_calls_per_month": -1,
    "concurrent_sessions": -1
}
features = {
    "basic_features", "email_support", "data_export",
    "advanced_analytics", "integrations", "priority_support",
    "sso", "audit_logs", "custom_branding", "dedicated_support"
}
```

## Usage Examples

### Tenant Management

```python
import asyncio
from apg.capabilities.common.mten import TenantTier, TenantStatus
from apg.capabilities.common.mten.service import get_persistent_tenant_service

async def tenant_management_examples():
    service = get_persistent_tenant_service()
    await service.initialize()
    
    # Create tenant
    tenant = await service.create_tenant(
        name="Tech Startup Inc",
        slug="techstartup",
        email="admin@techstartup.com",
        tier=TenantTier.BASIC,
        created_by="admin_user_id"
    )
    
    print(f"Created tenant: {tenant.name} ({tenant.id})")
    print(f"Tier: {tenant.tier.value}, Features: {list(tenant.features)}")
    
    # Update tenant information
    updated = await service.update_tenant(tenant.id, {
        "description": "A fast-growing technology startup",
        "primary_phone": "+1-555-0123"
    })
    
    # Upgrade tenant tier
    upgraded = await service.upgrade_tenant(tenant.id, TenantTier.PREMIUM)
    print(f"Upgraded to {upgraded.tier.value}")
    print(f"New limits: {upgraded.limits}")
    print(f"New features: {list(upgraded.features)}")
    
    # Check resource usage
    usage = await service.get_tenant_usage_stats(tenant.id)
    print(f"Current usage: {usage}")
    
    # Verify limits
    limit_check = await service.check_tenant_limits(tenant.id, "users", 50)
    print(f"User limit check: {limit_check}")

asyncio.run(tenant_management_examples())
```

### Tenant Context Operations

```python
from apg.capabilities.common.mten import (
    get_current_tenant, set_current_tenant, tenant_context
)

async def context_examples():
    # Method 1: Manual context management
    set_current_tenant("tenant-123")
    
    # All operations now scoped to tenant-123
    current = get_current_tenant()
    print(f"Current tenant: {current}")
    
    # Method 2: Context manager (recommended)
    async with tenant_context("tenant-456"):
        # Operations scoped to tenant-456
        await perform_database_operations()
        await call_external_apis()
        
        # Context automatically restored after block
    
    # Method 3: Nested contexts
    set_current_tenant("tenant-123")
    async with tenant_context("tenant-456"):
        # Inner context: tenant-456
        inner_tenant = get_current_tenant()
        print(f"Inner tenant: {inner_tenant}")
    
    # Restored to outer context: tenant-123
    outer_tenant = get_current_tenant() 
    print(f"Outer tenant: {outer_tenant}")

async def perform_database_operations():
    tenant_id = get_current_tenant()
    # All database queries automatically scoped to tenant_id
    pass

async def call_external_apis():
    tenant_id = get_current_tenant()
    # API calls include tenant context
    pass
```

### Advanced Querying

```python
from apg.capabilities.common.mten.service import TenantQueryFilter
from datetime import datetime, timedelta

async def query_examples():
    service = get_persistent_tenant_service()
    
    # Query active premium tenants
    premium_filter = TenantQueryFilter(
        status=TenantStatus.ACTIVE,
        tier=TenantTier.PREMIUM,
        limit=50
    )
    result = await service.query_tenants(premium_filter)
    print(f"Found {result.total_count} premium tenants")
    
    # Search by name
    search_filter = TenantQueryFilter(
        name_search="Tech",
        limit=20
    )
    result = await service.query_tenants(search_filter)
    print(f"Found {len(result.tenants)} tenants matching 'Tech'")
    
    # Recent tenants
    recent_filter = TenantQueryFilter(
        created_after=datetime.utcnow() - timedelta(days=30),
        limit=100
    )
    result = await service.query_tenants(recent_filter)
    print(f"Found {result.total_count} tenants created in last 30 days")
    
    # Pagination
    page1_filter = TenantQueryFilter(limit=25, offset=0)
    page1 = await service.query_tenants(page1_filter)
    
    page2_filter = TenantQueryFilter(limit=25, offset=25)
    page2 = await service.query_tenants(page2_filter)
    
    print(f"Page 1: {len(page1.tenants)} tenants")
    print(f"Page 2: {len(page2.tenants)} tenants")
    print(f"Has more: {page2.has_more}")
```

### Resource Management

```python
async def resource_management_examples():
    service = get_persistent_tenant_service()
    tenant_id = "tenant-123"
    
    # Check current usage against limits
    user_check = await service.check_tenant_limits(tenant_id, "users", 45)
    print(f"User limit check:")
    print(f"  Allowed: {user_check['allowed']}")
    print(f"  Limit: {user_check['limit']}")
    print(f"  Current: {user_check['current_usage']}")
    print(f"  Remaining: {user_check['remaining']}")
    print(f"  Usage %: {user_check['usage_percentage']}")
    
    # Get comprehensive usage stats
    usage = await service.get_tenant_usage_stats(tenant_id)
    print(f"Complete usage statistics:")
    print(f"  Current users: {usage['current_users']}")
    print(f"  Storage used: {usage['storage_used_gb']} GB")
    print(f"  API calls this month: {usage['api_calls_this_month']}")
    print(f"  Active sessions: {usage['active_sessions']}")
    
    # Usage percentages by resource
    for resource, percentage in usage['usage_percentage'].items():
        print(f"  {resource}: {percentage}%")
```

### Tenant Lifecycle Management

```python
async def lifecycle_examples():
    service = get_persistent_tenant_service()
    
    # Create tenant
    tenant = await service.create_tenant(
        name="Lifecycle Demo Corp",
        slug="lifecycle-demo", 
        email="admin@lifecycle.com"
    )
    
    # Provision resources
    provisioning = await service.provision_tenant_resources(tenant.id)
    print(f"Provisioned resources: {provisioning['resources']}")
    
    # Suspend tenant
    suspended = await service.suspend_tenant(tenant.id, "Non-payment")
    print(f"Tenant suspended: {suspended.status}")
    
    # Reactivate tenant
    reactivated = await service.reactivate_tenant(tenant.id)
    print(f"Tenant reactivated: {reactivated.status}")
    
    # Archive tenant (soft delete)
    archived = await service.delete_tenant(tenant.id)
    print(f"Tenant archived: {archived}")
    
    # Clean up archived tenant data
    cleanup = await service.cleanup_tenant_data(tenant.id)
    print(f"Data cleanup: {cleanup}")
```

## REST API

### Endpoints

The multi-tenancy capability provides a comprehensive REST API:

#### Create Tenant
```bash
POST /api/tenants/
Content-Type: application/json

{
  "name": "Acme Corporation",
  "slug": "acme-corp",
  "primary_email": "admin@acme-corp.com",
  "tier": "premium",
  "description": "Leading software company"
}
```

#### List Tenants
```bash
GET /api/tenants/?status=active&tier=premium&limit=50&offset=0
```

#### Get Tenant
```bash
GET /api/tenants/{tenant_id}
```

#### Update Tenant
```bash
PUT /api/tenants/{tenant_id}
Content-Type: application/json

{
  "description": "Updated description",
  "primary_phone": "+1-555-0123"
}
```

#### Upgrade Tenant
```bash
POST /api/tenants/{tenant_id}/upgrade
Content-Type: application/json

{
  "tier": "enterprise"
}
```

#### Suspend Tenant
```bash
POST /api/tenants/{tenant_id}/suspend
Content-Type: application/json

{
  "reason": "Policy violation"
}
```

#### Get Usage Statistics
```bash
GET /api/tenants/{tenant_id}/usage
```

#### Check Resource Limits
```bash
GET /api/tenants/{tenant_id}/limits/users/check?current_usage=45
```

#### Set Tenant Context
```bash
POST /api/tenants/context/{tenant_id}
```

#### Get Current Tenant
```bash
GET /api/tenants/current
```

### API Response Format

```json
{
  "success": true,
  "tenant": {
    "id": "01HKQR9Z8XVJ2K3L4M5N6P7Q8R",
    "name": "Acme Corporation",
    "slug": "acme-corp",
    "description": "Leading software company",
    "status": "active",
    "tier": "premium",
    "created_at": "2025-01-08T10:30:00.123Z",
    "updated_at": "2025-01-08T15:45:30.456Z",
    "primary_email": "admin@acme-corp.com",
    "limits": {
      "users": 100,
      "storage_gb": 100,
      "api_calls_per_month": 100000
    },
    "features": [
      "basic_features",
      "advanced_analytics",
      "integrations",
      "priority_support"
    ]
  }
}
```

## Configuration

### Environment Variables
- `APG_MTEN_ENABLED` - Enable/disable multi-tenancy (default: true)
- `APG_MTEN_DEFAULT_TIER` - Default tier for new tenants (default: free)
- `APG_MTEN_AUTO_PROVISION` - Auto-provision tenant resources (default: true)
- `APG_MTEN_ISOLATION_LEVEL` - Tenant isolation level (default: database)

### Configuration Templates

```python
from apg.capabilities.common.mten.models import MULTITENANCY_CONFIGURATION_TEMPLATES

# Load development configuration
dev_config = MULTITENANCY_CONFIGURATION_TEMPLATES["development"]

# Load production configuration  
prod_config = MULTITENANCY_CONFIGURATION_TEMPLATES["production"]
```

## Security Considerations

### Data Isolation
- Complete tenant data separation at all levels
- Schema-based or database-based isolation options
- Tenant context validation for all operations
- No cross-tenant data access possible

### Access Control
- Tenant-scoped authentication and authorization
- Admin-level operations require elevated privileges
- User access restricted to assigned tenants only
- Secure tenant context switching

### Resource Protection
- Strict enforcement of per-tenant resource limits
- Usage monitoring and alerting capabilities
- Automatic limit enforcement prevents resource abuse
- Billing integration for usage-based pricing

### Audit & Compliance
- Complete audit trail for all tenant operations
- Integration with APG audit logging capability
- Compliance reporting and data retention policies
- Tenant lifecycle audit logs

## Integration with Other Capabilities

The multi-tenancy capability integrates with all other APG capabilities:

### Configuration Management Integration
```python
from apg.capabilities.common.conf import get_config
from apg.capabilities.common.mten import get_current_tenant

# Get tenant-scoped configuration
async def get_tenant_config(key: str, default=None):
    tenant_id = get_current_tenant()
    if tenant_id:
        tenant_key = f"tenant.{tenant_id}.{key}"
        return await get_config(tenant_key, default)
    return await get_config(key, default)
```

### Audit Logging Integration
```python
from apg.capabilities.common.audl import audit_log, AuditLevel, AuditEventType
from apg.capabilities.common.mten import get_current_tenant

async def audit_tenant_action(action: str, resource: str = None):
    tenant_id = get_current_tenant()
    await audit_log(
        level=AuditLevel.INFO,
        event_type=AuditEventType.CUSTOM_EVENT,
        component="application",
        action=action,
        tenant_id=tenant_id,
        resource=resource
    )
```

### Database Integration
```python
class TenantAwareModel:
    """Base model with automatic tenant scoping"""
    
    def __init__(self):
        self.tenant_id = get_current_tenant()
    
    async def save(self):
        # Automatically include tenant_id in all saves
        if not self.tenant_id:
            raise ValueError("No tenant context available")
        # Save with tenant isolation
        
    @classmethod
    async def query(cls, **filters):
        # Automatically scope queries to current tenant
        tenant_id = get_current_tenant()
        if tenant_id:
            filters['tenant_id'] = tenant_id
        return await cls._query(**filters)
```

## Development & Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run multi-tenancy tests
pytest apg/capabilities/common/mten/tests/ -v

# Run with coverage
pytest --cov=apg.capabilities.common.mten apg/capabilities/common/mten/tests/
```

### Development Setup

```bash
# Install in development mode
pip install -e .

# Set development environment
export APG_ENVIRONMENT=development
export APG_MTEN_ENABLED=true
export APG_MTEN_DEFAULT_TIER=free
```

## Performance Considerations

- **In-Memory Caching**: Active tenant data cached for fast access
- **Async Operations**: All operations are non-blocking and concurrent
- **Database Indexing**: Optimized indexes for common query patterns
- **Context Efficiency**: Thread-local storage for minimal overhead
- **Lazy Loading**: Tenant data loaded on-demand to minimize memory usage

## Troubleshooting

### Common Issues

1. **Tenant Not Found**
   - Verify tenant ID is correct
   - Check if tenant has been archived
   - Ensure proper tenant context is set

2. **Context Issues**
   - Use context managers for reliable context switching
   - Check thread-local storage in multi-threaded environments
   - Verify context is set before tenant-aware operations

3. **Resource Limit Exceeded**
   - Check current usage against tenant limits
   - Consider upgrading tenant tier
   - Review resource allocation patterns

### Debug Configuration

```python
# Enable debug logging for tenant operations
from apg.capabilities.common.mten import get_tenant_manager

manager = get_tenant_manager()

# Get current context info
current_tenant = manager.get_current_tenant()
print(f"Current tenant: {current_tenant}")

# List all tenants for debugging
all_tenants = await manager.list_tenants()
for tenant in all_tenants:
    print(f"Tenant: {tenant.name} ({tenant.id}) - {tenant.status}")
```

## Best Practices

### Context Management
- Always use context managers (`tenant_context`) for operations
- Set tenant context as early as possible in request lifecycle
- Validate tenant access before setting context
- Clear context appropriately to prevent leaks

### Resource Management
- Regularly monitor tenant usage against limits
- Implement usage alerts and notifications
- Plan for tenant growth and tier upgrades
- Implement graceful limit enforcement

### Security
- Never bypass tenant isolation mechanisms
- Validate all tenant IDs from external inputs
- Use proper authentication for tenant operations
- Implement audit logging for all tenant changes

### Performance
- Cache frequently accessed tenant data
- Use efficient database queries with proper indexing
- Implement pagination for large tenant lists
- Monitor and optimize tenant-scoped operations

## License

© 2025 Datacraft. All rights reserved.

---

**Next Steps**: After implementing the Multi-Tenancy capability, proceed with [Authentication & RBAC (`auth`)](../auth/README.md) as defined in the development order plan.

---

## World-Class Enhancements (v2.0)

- **I1.** Multi-Tenancy (mten) — World-Class Improvements
- **I2.** Hierarchical Tenant Trees (Sub-Tenants)
- **I3.** Quota Ledger with Real-Time Enforcement
- **I4.** Policy-as-Code Engine (OPA/Rego Bridge)
- **I5.** Tenant Namespace Namespacing at the Database Layer
- **I6.** Graceful Tier Downgrade with Usage Conflict Resolution
- **I7.** Cross-Tenant Data Masking and Tokenisation
- **I8.** Event-Driven Tenant Lifecycle Webhooks
- **I9.** Tenant-Scoped Secret Management
- **I10.** Tenant Activity Fingerprinting and Behaviour Baselining
- **I11.** Immutable Tenant Configuration Snapshots
- **I12.** SLA-Aware Provisioning with Circuit Breaker
- **I13.** Cost Attribution and Showback / Chargeback Reports
- **I14.** Zero-Downtime Live Tenant Migration with State Sync
- **I15.** Tenant-Scoped Rate Limiting with Token Bucket and Burst Control

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
