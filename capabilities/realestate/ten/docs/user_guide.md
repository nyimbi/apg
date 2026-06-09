# Tenant Management

**Capability ID**: `realestate_ten` | **Domain**: `realestate` | **Version**: `1.0.0`

## Description

Full tenant lifecycle from prospect registration through onboarding (10-step workflow with mandatory-step gating), service request management with SLA enforcement, multi-channel communication portal, satisfaction surveying with automatic review triggers, tenant scoring and credit grading, escalation management, and retention risk analytics.

## Installation

```bash
pip install apg-realestate-ten
```

## Provides

- `tenant_onboarding_workflow`
- `tenant_communication_portal`
- `service_request_management`
- `tenant_scoring_engine`
- `satisfaction_tracking`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/ten/dashboard` | `realestate_ten:view` | Overview |
| `/realestate/ten/tenants` | `realestate_ten:tenants` | Tenants |
| `/realestate/ten/tenants/<id>` | `realestate_ten:tenants` | Tenants |
| `/realestate/ten/onboarding` | `realestate_ten:onboarding` | Onboarding |
| `/realestate/ten/service-requests` | `realestate_ten:service_requests` | Services |
| `/realestate/ten/communications` | `realestate_ten:communications` | Communications |
| `/realestate/ten/satisfaction` | `realestate_ten:satisfaction` | Analytics |
| `/realestate/ten/scoring` | `realestate_ten:scoring` | Analytics |

## Key Service Methods

- `register_tenant()`
- `get_tenant()`
- `list_tenants()`
- `update_tenant()`
- `activate_tenant()`
- `blacklist_tenant()`
- `complete_onboarding_step()`
- `get_onboarding_progress()`
- `raise_service_request()`
- `get_service_request()`

_(See `service.py` for complete API.)_

## Interoperability

`realestate_ten` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use realestate_ten;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `REALESTATE_TEN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
