# Order Management

**Capability ID**: `telecom_ord` | **Domain**: `telecom` | **Version**: `1.0.0`

## Description

End-to-end service order management covering order capture, validation, decomposition into provisioning tasks, orchestration, fallout management, number portability, bulk order processing, and real-time order tracking. Enforces duplicate detection and requires explicit approval for bulk operations.

## Installation

```bash
pip install apg-telecom-ord
```

## Provides

- `order_capture_workflow`
- `order_validation_workflow`
- `order_decomposition_workflow`
- `provisioning_orchestration_workflow`
- `fallout_management_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-ord/dashboard` | `telecom_ord:view` | Overview |
| `/telecom-ord/orders` | `telecom_ord:orders` | Orders |
| `/telecom-ord/orders/<id>` | `telecom_ord:orders` | Orders |
| `/telecom-ord/decomposition` | `telecom_ord:decomposition` | Processing |
| `/telecom-ord/tasks` | `telecom_ord:tasks` | Processing |
| `/telecom-ord/fallout` | `telecom_ord:fallout` | Operations |
| `/telecom-ord/provisioning` | `telecom_ord:provisioning` | Processing |
| `/telecom-ord/portability` | `telecom_ord:portability` | Special Orders |

## Key Service Methods

- `describe()`
- `evaluate()`
- `submit_order()`
- `validate_order()`
- `decompose_order()`
- `create_task()`
- `complete_task()`
- `record_fallout()`
- `retry_fallout()`
- `resolve_fallout()`

_(See `service.py` for complete API.)_

## Interoperability

`telecom_ord` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use telecom_ord;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_ORD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
