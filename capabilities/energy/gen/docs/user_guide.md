# Generation Management

**Capability ID**: `energy_gen` | **Domain**: `energy` | **Version**: `1.0.0`

## Description

Generation Management provides end-to-end lifecycle management of power generation assets including thermal, hydro, and renewable plants. It covers plant registration, economic dispatch scheduling, outage management with approval workflows, KPI calculation (availability, capacity factor, heat rate), capacity planning, and fuel stock monitoring with low-supply alerting.

## Installation

```bash
pip install apg-energy-gen
```

## Provides

- `plant_registry`
- `dispatch_scheduling`
- `outage_management`
- `capacity_planning`
- `generation_kpis`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/energy-gen/dashboard` | `energy_gen:view` | Overview |
| `/energy-gen/plants` | `energy_gen:plants` | Assets |
| `/energy-gen/plants/<id>` | `energy_gen:plants` | Assets |
| `/energy-gen/dispatch` | `energy_gen:dispatch` | Operations |
| `/energy-gen/schedules` | `energy_gen:dispatch` | Operations |
| `/energy-gen/outages` | `energy_gen:outages` | Maintenance |
| `/energy-gen/outages/<id>` | `energy_gen:outages` | Maintenance |
| `/energy-gen/kpis` | `energy_gen:kpis` | Performance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_plant()`
- `update_plant_status()`
- `list_plants()`
- `get_plant()`
- `decommission_plant()`
- `create_dispatch_schedule()`
- `approve_dispatch_schedule()`
- `list_dispatch_schedules()`

_(See `service.py` for complete API.)_

## Interoperability

`energy_gen` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use energy_gen;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ENERGY_GEN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
