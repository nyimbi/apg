# Transport Scheduling

**Capability ID**: `transport_sch` | **Domain**: `transport` | **Version**: `1.0.0`

## Description

The Transport Scheduling capability manages load scheduling, driver shift planning with tachograph and HOS compliance, vehicle assignment, charter management (school, corporate, tourist, medical), schedule optimisation, and conflict detection. It blocks schedule publication when unresolved conflicts exist and enforces tacho compliance on all shifts.

## Installation

```bash
pip install apg-transport-sch
```

## Provides

- `load_scheduling_workflow`
- `driver_shift_planning_workflow`
- `vehicle_assignment_workflow`
- `charter_management_workflow`
- `schedule_optimisation_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/transport-scheduling/dashboard` | `transport_sch:view` | Overview |
| `/transport-scheduling/schedules` | `transport_sch:schedules` | Schedules |
| `/transport-scheduling/schedules/create` | `transport_sch:schedules_write` | Schedules |
| `/transport-scheduling/calendar` | `transport_sch:view` | Overview |
| `/transport-scheduling/shifts` | `transport_sch:shifts` | Drivers |
| `/transport-scheduling/vehicles` | `transport_sch:vehicles` | Vehicles |
| `/transport-scheduling/charters` | `transport_sch:charters` | Charters |
| `/transport-scheduling/optimisation` | `transport_sch:optimisation` | Optimisation |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_schedule()`
- `publish_schedule()`
- `create_shift()`
- `assign_vehicle()`
- `create_charter()`
- `record_conflict()`
- `resolve_conflict()`
- `send_notification()`

_(See `service.py` for complete API.)_

## Interoperability

`transport_sch` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use transport_sch;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_SCH_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
