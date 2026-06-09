# Tenants Legacy

**Capability ID**: `tens` | **Domain**: `common` | **Version**: `1.0.0`

## Description

TENS is the APG capability for legacy tenant compatibility and migration governance. It gives generated applications a composable runtime for legacy tenant registration, APG tenant mapping, access-boundary validation, migration approval, migration completion, deprecation planning, AI-assisted review, and Bytewax lifecycle events.

## Installation

```bash
pip install apg-common-tens
```

## Provides

- `legacy_tenant_registry`
- `tenant_mapping`
- `migration_controls`
- `access_boundaries`
- `deprecation_governance`

## Requires

- `mten`
- `auth`
- `audl`
- `idfd`
- `usrm`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/tens/dashboard` | `tens:view` | Overview |
| `/tens/tenants` | `tens:view` | Tenants |
| `/tens/mappings` | `tens:map` | Mapping |
| `/tens/migrations` | `tens:migrate` | Migration |
| `/tens/boundaries` | `tens:approve` | Access |
| `/tens/deprecation` | `tens:approve` | Governance |
| `/tens/agents` | `tens:admin` | Automation |
| `/tens/policy` | `tens:admin` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_legacy_tenant()`
- `map_tenant()`
- `validate_access_boundary()`
- `create_migration_plan()`
- `complete_migration()`
- `record_deprecation_plan()`
- `create_record()`
- `list_records()`

_(See `service.py` for complete API.)_

## Interoperability

`tens` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use tens;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TENS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
