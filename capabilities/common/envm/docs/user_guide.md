# Environment Management

**Capability ID**: `envm` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`envm` is the APG common environment management capability. It lets generated applications compose tenant-scoped environment inventory, stage and region policy, governed promotion paths, promotion runs, configuration drift reports,

## Installation

```bash
pip install apg-common-envm
```

## Provides

- `environment_inventory`
- `environment_promotion`
- `configuration_drift`
- `secret_scopes`
- `environment_policy`

## Requires

- `auth`
- `conf`
- `audl`
- `depl`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/envm/dashboard` | `envm:view` | Overview |
| `/envm/environments` | `envm:manage_environments` | Inventory |
| `/envm/promotion` | `envm:promote` | Promotion |
| `/envm/drift` | `envm:view` | Governance |
| `/envm/secrets` | `envm:manage_secrets` | Security |
| `/envm/agents` | `envm:govern` | Governance |
| `/envm/policies` | `envm:admin` | Governance |
| `/envm/rules` | `envm:govern` | Governance |

## Key Service Methods

- `uuid7str()`
- `uuid7str()`
- `put()`
- `get()`
- `list()`
- `delete()`
- `log_event()`
- `send()`
- `env_create()`
- `env_clone()`

_(See `service.py` for complete API.)_

## Interoperability

`envm` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use envm;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ENVM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
