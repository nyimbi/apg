# Scheduling and Job Orchestration

**Capability ID**: `schd` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`schd` is the APG common capability for governed schedules, calendar triggers, job definitions, worker pools, run recovery, and scheduler operations. It gives generated applications a dependency-light runtime that can define jobs, attach

## Installation

```bash
pip install apg-common-schd
```

## Provides

- `job_scheduling`
- `calendar_triggers`
- `worker_orchestration`
- `retry_policies`
- `job_monitoring`

## Requires

- `wflo`
- `mqeb`
- `moni`
- `audl`
- `aicr`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/schd/dashboard` | `schd:view` | Overview |
| `/schd/schedules` | `schd:schedule` | Schedules |
| `/schd/jobs` | `schd:run_jobs` | Jobs |
| `/schd/runs` | `schd:view` | Runtime |
| `/schd/workers` | `schd:manage_workers` | Workers |
| `/schd/calendars` | `schd:schedule` | Schedules |
| `/schd/agents` | `schd:run_jobs` | Runtime |
| `/schd/lifecycle` | `schd:admin` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_calendar_policy()`
- `register_worker_pool()`
- `change_worker_state()`
- `define_job()`
- `create_schedule()`
- `trigger_run()`
- `complete_run()`
- `retry_run()`

_(See `service.py` for complete API.)_

## Interoperability

`schd` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use schd;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `SCHD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
