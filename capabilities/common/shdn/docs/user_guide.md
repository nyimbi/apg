# Shutdown and Lifecycle Control

**Capability ID**: `shdn` | **Domain**: `common` | **Version**: `1.0.0`

## Description

SHDN is the APG capability for governed service lifecycle control. It gives generated applications a composable runtime for registering lifecycle targets, building shutdown plans, draining services, enforcing backup and health gates, executing shutdowns, recording recovery evidence, composing AI-assisted review, and emitting Bytewax lifecycle events.

## Installation

```bash
pip install apg-common-shdn
```

## Provides

- `service_lifecycle`
- `shutdown_orchestration`
- `restart_plans`
- `backup_gates`
- `operational_safety`

## Requires

- `moni`
- `hlth`
- `bkup`
- `audl`
- `envm`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/shdn/dashboard` | `shdn:view` | Overview |
| `/shdn/services` | `shdn:view` | Services |
| `/shdn/plans` | `shdn:plan` | Planning |
| `/shdn/executions` | `shdn:execute` | Execution |
| `/shdn/approvals` | `shdn:approve` | Governance |
| `/shdn/recovery` | `shdn:execute` | Recovery |
| `/shdn/agents` | `shdn:admin` | Automation |
| `/shdn/policy` | `shdn:admin` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_service()`
- `create_shutdown_plan()`
- `start_drain()`
- `record_backup_snapshot()`
- `execute_shutdown()`
- `record_recovery()`
- `create_record()`
- `list_records()`

_(See `service.py` for complete API.)_

## Interoperability

`shdn` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use shdn;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `SHDN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
