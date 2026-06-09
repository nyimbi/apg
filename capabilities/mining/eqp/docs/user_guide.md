# Equipment & Plant Management

**Capability ID**: `mining_eqp` | **Domain**: `mining` | **Version**: `1.0.0`

## Description

Manages the full lifecycle of mining fleet and processing plant equipment including registration, dispatch, maintenance work orders, preventive maintenance scheduling, pre-shift inspections, fuel consumption tracking, fault reporting, and fleet KPI reporting. Enforces equipment availability guardrails: breakdown equipment cannot be dispatched, operators must hold valid licences, and pre-shift inspections must pass before daily dispatch.

## Installation

```bash
pip install apg-mining-eqp
```

## Provides

- `fleet_register_management`
- `equipment_lifecycle_tracking`
- `maintenance_work_order_workflow`
- `preventive_maintenance_scheduling`
- `equipment_dispatch_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mining-eqp/dashboard` | `mining_eqp:view` | Overview |
| `/mining-eqp/fleet` | `mining_eqp:view` | Fleet |
| `/mining-eqp/fleet/create` | `mining_eqp:write` | Fleet |
| `/mining-eqp/fleet/:id` | `mining_eqp:view` | Fleet |
| `/mining-eqp/maintenance` | `mining_eqp:view` | Maintenance |
| `/mining-eqp/maintenance/create` | `mining_eqp:maintenance` | Maintenance |
| `/mining-eqp/maintenance/schedule` | `mining_eqp:maintenance` | Maintenance |
| `/mining-eqp/dispatch` | `mining_eqp:dispatch` | Dispatch |

## Key Service Methods

- `register_equipment()`
- `get_equipment()`
- `get_equipment_by_asset_number()`
- `update_equipment()`
- `decommission_equipment()`
- `list_equipment()`
- `dispatch_equipment()`
- `create_work_order()`
- `approve_work_order()`
- `complete_work_order()`

_(See `service.py` for complete API.)_

## Interoperability

`mining_eqp` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mining_eqp;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MINING_EQP_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
