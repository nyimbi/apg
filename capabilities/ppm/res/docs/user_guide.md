# Resource Management

**Capability ID**: `ppm_res` | **Domain**: `ppm` | **Version**: `1.0.0`

## Description

Resource Management (res) manages the full resource lifecycle: pool registration, skill cataloguing with evidence-backed proficiency, allocation to projects with over-allocation controls, capacity planning, utilisation band tracking, demand forecasting, leave management, and cost rate governance with finance-approval gates.

## Installation

```bash
pip install apg-ppm-res
```

## Provides

- `resource_pool_management`
- `skill_matching_engine`
- `capacity_planning`
- `utilisation_tracking`
- `demand_forecasting`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ppm-res/dashboard` | `ppm_res:view` | Overview |
| `/ppm-res/resources` | `ppm_res:resources` | Resources |
| `/ppm-res/resources/<id>` | `ppm_res:resources` | Resources |
| `/ppm-res/skills` | `ppm_res:skills` | Skills |
| `/ppm-res/skill-match` | `ppm_res:skill_match` | Skills |
| `/ppm-res/allocations` | `ppm_res:allocations` | Allocations |
| `/ppm-res/capacity` | `ppm_res:capacity` | Planning |
| `/ppm-res/utilisation` | `ppm_res:utilisation` | Analytics |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_resource()`
- `get_resource()`
- `list_resources()`
- `register_resource()`
- `skill_search()`
- `assign_resource()`
- `resource_utilisation()`
- `team_capacity()`

_(See `service.py` for complete API.)_

## Interoperability

`ppm_res` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use ppm_res;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PPM_RES_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
