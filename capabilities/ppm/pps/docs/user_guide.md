# Project Planning & Scheduling

**Capability ID**: `ppm_pps` | **Domain**: `ppm` | **Version**: `1.0.0`

## Description

Project Planning & Scheduling (pps) manages the full project schedule lifecycle: WBS decomposition, task definition, dependency linking with circular-dependency prevention, critical path calculation (CPM/PERT/CCPM/Monte Carlo), resource levelling, calendar management, and milestone tracking. Retroactive edits are blocked to maintain schedule integrity.

## Installation

```bash
pip install apg-ppm-pps
```

## Provides

- `wbs_creation_and_management`
- `critical_path_analysis`
- `resource_levelling`
- `dependency_management`
- `timeline_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ppm-pps/dashboard` | `ppm_pps:view` | Overview |
| `/ppm-pps/projects` | `ppm_pps:projects` | Projects |
| `/ppm-pps/projects/<id>` | `ppm_pps:projects` | Projects |
| `/ppm-pps/projects/<id>/wbs` | `ppm_pps:wbs` | Planning |
| `/ppm-pps/projects/<id>/gantt` | `ppm_pps:gantt` | Planning |
| `/ppm-pps/projects/<id>/critical-path` | `ppm_pps:critical_path` | Analysis |
| `/ppm-pps/projects/<id>/dependencies` | `ppm_pps:dependencies` | Planning |
| `/ppm-pps/projects/<id>/levelling` | `ppm_pps:levelling` | Resources |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_project()`
- `get_project()`
- `list_projects()`
- `add_wbs_element()`
- `list_wbs_elements()`
- `add_task()`
- `update_task_status()`
- `list_tasks()`

_(See `service.py` for complete API.)_

## Interoperability

`ppm_pps` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use ppm_pps;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PPM_PPS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
