# Space Planning & Management

**Capability ID**: `realestate_spa` | **Domain**: `realestate` | **Version**: `1.0.0`

## Description

Comprehensive workplace and space management: versioned floor plans, space allocation and deallocation, move management with headcount-threshold approvals, conflict-checked space bookings, anonymised sensor-data ingestion for occupancy analytics, workplace density planning, and space chargeback calculation.

## Installation

```bash
pip install apg-realestate-spa
```

## Provides

- `floor_plan_management`
- `space_allocation_engine`
- `move_management_workflow`
- `occupancy_analytics`
- `workplace_density_planning`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/spa/dashboard` | `realestate_spa:view` | Overview |
| `/realestate/spa/floor-plans` | `realestate_spa:floor_plans` | Floor Plans |
| `/realestate/spa/spaces` | `realestate_spa:spaces` | Spaces |
| `/realestate/spa/allocations` | `realestate_spa:allocations` | Allocation |
| `/realestate/spa/moves` | `realestate_spa:moves` | Moves |
| `/realestate/spa/bookings` | `realestate_spa:bookings` | Bookings |
| `/realestate/spa/occupancy` | `realestate_spa:occupancy` | Analytics |
| `/realestate/spa/density` | `realestate_spa:density` | Planning |

## Key Service Methods

- `upload_floor_plan()`
- `get_floor_plan()`
- `list_floor_plans()`
- `create_space()`
- `get_space()`
- `list_spaces()`
- `update_space()`
- `get_available_spaces()`
- `allocate_space()`
- `deallocate_space()`

_(See `service.py` for complete API.)_

## Interoperability

`realestate_spa` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use realestate_spa;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `REALESTATE_SPA_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
