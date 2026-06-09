# Configuration Management

**Capability ID**: `conf` | **Domain**: `common` | **Version**: `1.0.0`

## Description

**System-wide configuration store providing centralized, hierarchical configuration management with environment-specific overrides, validation, and real-time updates.**

## Installation

```bash
pip install apg-common-conf
```

## Provides

- `conf_operations`
- `conf_agents`
- `review_evidence`

## Requires

_(none)_

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/config/dashboard` | `conf:view` | Operations |
| `/config/resources` | `conf:view` | Author |
| `/config/templates` | `conf:create` | Author |
| `/config/changes` | `conf:edit` | Governance |
| `/config/approvals` | `conf:approve` | Governance |
| `/config/policies` | `conf:admin` | Governance |
| `/config/deployments` | `conf:deploy` | Operations |
| `/config/drift` | `conf:view` | Governance |

## Key Service Methods

- `set_config_manager()`
- `set_gitops_manager()`
- `set_nlp_service()`
- `describe_runtime()`
- `_maybe_await()`
- `_maybe_initialize()`
- `_maybe_shutdown()`
- `initialize()`
- `create_configuration()`
- `deploy_configuration()`

_(See `service.py` for complete API.)_

## Interoperability

`conf` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use conf;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `CONF_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
