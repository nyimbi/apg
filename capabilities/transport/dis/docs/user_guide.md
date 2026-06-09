# Dispatch Operations

**Capability ID**: `transport_dis` | **Domain**: `transport` | **Version**: `1.0.0`

## Description

The Dispatch Operations capability manages load planning, driver assignment with hours-of-service compliance, dispatch optimisation, real-time GPS tracking updates, and exception management. It enforces vehicle capacity limits, driver hours regulations, and provides multi-channel driver communication.

## Installation

```bash
pip install apg-transport-dis
```

## Provides

- `load_planning_workflow`
- `driver_assignment_workflow`
- `dispatch_optimisation_workflow`
- `real_time_tracking_workflow`
- `exception_management_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/transport-dispatch/dashboard` | `transport_dis:view` | Overview |
| `/transport-dispatch/loads` | `transport_dis:loads` | Planning |
| `/transport-dispatch/loads/create` | `transport_dis:loads_write` | Planning |
| `/transport-dispatch/board` | `transport_dis:dispatch` | Operations |
| `/transport-dispatch/drivers` | `transport_dis:drivers` | Operations |
| `/transport-dispatch/tracking` | `transport_dis:tracking` | Operations |
| `/transport-dispatch/exceptions` | `transport_dis:exceptions` | Exceptions |
| `/transport-dispatch/optimisation` | `transport_dis:optimisation` | Planning |

## Key Service Methods

- `describe()`
- `evaluate()`
- `plan_load()`
- `assign_driver()`
- `create_dispatch()`
- `update_dispatch_status()`
- `update_tracking()`
- `raise_exception()`
- `resolve_exception()`
- `send_communication()`

_(See `service.py` for complete API.)_

## Interoperability

`transport_dis` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use transport_dis;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_DIS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
