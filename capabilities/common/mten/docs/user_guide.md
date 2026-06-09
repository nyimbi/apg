# Multi-Tenant Management

**Capability ID**: `mten` | **Domain**: `common` | **Version**: `1.0.0`

## Description

**Enterprise multi-tenancy framework providing tenant isolation, management, and context switching for the APG platform.**

## Installation

```bash
pip install apg-common-mten
```

## Provides

- `mten_operations`
- `tenant_agents`
- `review_evidence`

## Requires

_(none)_

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mten/dashboard` | `mten:view` | Overview |
| `/mten/tenants` | `mten:view` | Operations |
| `/mten/provisioning` | `mten:provision` | Operations |
| `/mten/capacity/approvals` | `mten:approve_capacity` | Governance |
| `/mten/isolation` | `mten:admin` | Governance |
| `/mten/migrations` | `mten:migrate` | Operations |
| `/mten/templates` | `mten:manage_templates` | Build |
| `/mten/analytics` | `mten:view_analytics` | Intelligence |

## Key Service Methods

- `model_dump()`
- `get_tenant_permissions()`
- `log_event()`
- `status()`
- `initialize()`
- `_initialize_apg_integrations()`
- `_load_default_templates()`
- `create_tenant()`
- `_provision_tenant_async()`
- `_allocate_compute_resources()`

_(See `service.py` for complete API.)_

## Interoperability

`mten` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mten;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MTEN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
