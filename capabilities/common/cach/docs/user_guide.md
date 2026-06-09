# Cache Management

**Capability ID**: `cach` | **Domain**: `common` | **Version**: `1.0.0`

## Description

CACH is APG's cache governance and runtime-adapter capability. It gives generated applications a tenant-aware way to register cache namespaces, enforce entry admission rules, manage warming and eviction reviews, publish UI metadata,

## Installation

```bash
pip install apg-common-cach
```

## Provides

- `cache_governance`
- `cache_runtime_adapters`
- `cache_agent_composition`
- `review_evidence`

## Requires

- `conf`
- `auth`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/cach/dashboard` | `cach:view` | Overview |
| `/cach/namespaces` | `cach:manage_namespaces` | Operations |
| `/cach/entries` | `cach:read` | Operations |
| `/cach/policies` | `cach:manage_policies` | Governance |
| `/cach/warming` | `cach:warm` | Operations |
| `/cach/evictions` | `cach:review_eviction` | Governance |
| `/cach/hierarchy` | `cach:view` | Architecture |
| `/cach/analytics` | `cach:view_analytics` | Intelligence |

## Key Service Methods

- `uuid7str()`
- `_audit()`
- `cache_set()`
- `cache_get()`
- `cache_delete()`
- `cache_exists()`
- `bulk_set()`
- `bulk_get()`
- `cache_flush()`
- `ttl_update()`

_(See `service.py` for complete API.)_

## Interoperability

`cach` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use cach;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `CACH_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
