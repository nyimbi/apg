# Project Baseline Management

**Capability ID**: `ppm_pbl` | **Domain**: `ppm` | **Version**: `1.0.0`

## Description

Project Baseline Management (pbl) establishes and protects the scope, schedule, and cost baselines for projects. It enforces formal change control, calculates earned value metrics, detects variance threshold breaches, and prevents retroactive baseline manipulation — providing the performance measurement baseline required for EVM compliance.

## Installation

```bash
pip install apg-ppm-pbl
```

## Provides

- `scope_baseline_management`
- `schedule_baseline_management`
- `cost_baseline_management`
- `change_control_workflow`
- `earned_value_analysis`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ppm-pbl/dashboard` | `ppm_pbl:view` | Overview |
| `/ppm-pbl/baselines` | `ppm_pbl:baselines` | Baselines |
| `/ppm-pbl/baselines/<id>` | `ppm_pbl:baselines` | Baselines |
| `/ppm-pbl/scope` | `ppm_pbl:scope` | Baselines |
| `/ppm-pbl/schedule` | `ppm_pbl:schedule` | Baselines |
| `/ppm-pbl/cost` | `ppm_pbl:cost` | Baselines |
| `/ppm-pbl/changes` | `ppm_pbl:changes` | Change Control |
| `/ppm-pbl/changes/<id>` | `ppm_pbl:changes` | Change Control |

## Key Service Methods

- `describe()`
- `evaluate()`
- `set_scope_baseline()`
- `set_schedule_baseline()`
- `set_cost_baseline()`
- `change_request()`
- `approve_change()`
- `baseline_comparison()`
- `variance_analysis()`
- `change_log()`

_(See `service.py` for complete API.)_

## Interoperability

`ppm_pbl` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use ppm_pbl;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PPM_PBL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
