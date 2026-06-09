# AI Agent Composition

**Capability ID**: `agnt` | **Domain**: `common` | **Version**: `1.0.0`

## Description

AGNT makes AI agents first-class APG citizens. It gives generated applications a provider-neutral way to register agent runtimes, request approval for external providers, declare agents with models and contracts, compose teams

## Installation

```bash
pip install apg-common-agnt
```

## Provides

- `agent_registry`
- `runtime_registry`
- `agent_teams`
- `handoff_graphs`
- `execution_plans`

## Requires

- `aicr`
- `sbox`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/agnt/dashboard` | `agnt:view` | Overview |
| `/agnt/agents` | `agnt:compose` | Agents |
| `/agnt/teams` | `agnt:compose` | Teams |
| `/agnt/handoffs` | `agnt:compose` | Teams |
| `/agnt/runtimes` | `agnt:manage_runtimes` | Runtimes |
| `/agnt/executions` | `agnt:run` | Operations |
| `/agnt/runs` | `agnt:run` | Operations |
| `/agnt/memory` | `agnt:admin` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_runtime()`
- `list_runtimes()`
- `request_runtime_approval()`
- `decide_runtime_approval()`
- `list_runtime_approvals()`
- `list_audit_events()`
- `register_agent()`
- `list_agents()`

_(See `service.py` for complete API.)_

## Interoperability

`agnt` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use agnt;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `AGNT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
