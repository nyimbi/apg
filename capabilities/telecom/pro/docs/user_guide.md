# Service Provisioning

**Capability ID**: `telecom_pro` | **Domain**: `telecom` | **Version**: `1.0.0`

## Description

Service activation and provisioning engine covering workflow orchestration, network resource reservation, configuration push to network elements via multiple protocols (NETCONF, RESTCONF, CLI, REST API), end-to-end activation verification, automated rollback on failure, and bulk provisioning with pre-approval gating.

## Installation

```bash
pip install apg-telecom-pro
```

## Provides

- `service_activation_workflow`
- `network_resource_allocation`
- `configuration_push_workflow`
- `activation_confirmation_workflow`
- `rollback_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-pro/dashboard` | `telecom_pro:view` | Overview |
| `/telecom-pro/workflows` | `telecom_pro:workflows` | Provisioning |
| `/telecom-pro/workflows/<id>` | `telecom_pro:workflows` | Provisioning |
| `/telecom-pro/resources` | `telecom_pro:resources` | Resources |
| `/telecom-pro/config-push` | `telecom_pro:config_push` | Configuration |
| `/telecom-pro/network-elements` | `telecom_pro:network_elements` | Configuration |
| `/telecom-pro/activation` | `telecom_pro:activation` | Provisioning |
| `/telecom-pro/rollback` | `telecom_pro:rollback` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `start_workflow()`
- `update_workflow_status()`
- `reserve_resource()`
- `release_resource()`
- `push_config()`
- `confirm_activation()`
- `trigger_rollback()`
- `complete_rollback()`

_(See `service.py` for complete API.)_

## Interoperability

`telecom_pro` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use telecom_pro;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_PRO_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
