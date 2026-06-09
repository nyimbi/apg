# Deployment Management

**Capability ID**: `depl` | **Domain**: `common` | **Version**: `1.0.0`

## Description

DEPL is the APG capability for governed release, deployment, health-gate, rollback, deployment-agent, audit, and deployment-evidence workflows. It gives generated APG applications a tenant-aware deployment lifecycle that can be

## Installation

```bash
pip install apg-common-depl
```

## Provides

- `release_management`
- `deployment_rollouts`
- `health_gates`
- `rollback_control`
- `deployment_audit`

## Requires

- `logt`
- `moni`
- `hlth`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/depl/dashboard` | `depl:view` | Overview |
| `/depl/releases` | `depl:plan` | Releases |
| `/depl/deployments` | `depl:deploy` | Runtime |
| `/depl/rollouts` | `depl:deploy` | Runtime |
| `/depl/health` | `depl:view` | Quality |
| `/depl/rollback` | `depl:rollback` | Recovery |
| `/depl/agents` | `depl:deploy` | Agents |
| `/depl/evidence` | `depl:view` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_environment()`
- `create_release()`
- `attach_rollback_plan()`
- `record_health_gate()`
- `create_deployment_plan()`
- `approve_deployment_plan()`
- `execute_deployment()`
- `execute_rollback()`

_(See `service.py` for complete API.)_

## Interoperability

`depl` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use depl;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `DEPL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
