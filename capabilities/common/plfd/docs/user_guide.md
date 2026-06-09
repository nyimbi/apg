# Platform Foundation

**Capability ID**: `plfd` | **Domain**: `common` | **Version**: `1.0.0`

## Description

PLFD provides APG applications with a tenant-scoped foundation governance runtime: platform service registry, dependency posture, required baselines, readiness gates, platform change approval, foundation agents, UI metadata,

## Installation

```bash
pip install apg-common-plfd
```

## Provides

- `foundation_registry`
- `dependency_posture`
- `configuration_baselines`
- `readiness_gates`
- `platform_governance`

## Requires

- `conf`
- `mten`
- `auth`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/plfd/dashboard` | `plfd:view` | Overview |
| `/plfd/services` | `plfd:manage_services` | Services |
| `/plfd/dependencies` | `plfd:view` | Readiness |
| `/plfd/baselines` | `plfd:manage_baselines` | Baselines |
| `/plfd/readiness` | `plfd:view` | Readiness |
| `/plfd/changes` | `plfd:approve_changes` | Governance |
| `/plfd/agents` | `plfd:admin` | Operations |
| `/plfd/governance` | `plfd:admin` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `health_check_all_services()`
- `platform_configuration()`
- `feature_flag_set()`
- `feature_flag_check()`
- `circuit_breaker_status()`
- `circuit_breaker_reset()`
- `dependency_graph()`
- `service_discovery_register()`

_(See `service.py` for complete API.)_

## Interoperability

`plfd` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use plfd;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PLFD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
