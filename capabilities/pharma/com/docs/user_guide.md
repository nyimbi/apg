# Commercial Operations

**Capability ID**: `pharma_com` | **Domain**: `pharma` | **Version**: `1.0.0`

## Description

Manages pharmaceutical field force activities including territory management, sales rep assignments, physician call recording, PDMA-compliant sample dispensing, HCP interaction tracking, aggregate spend management, and commercial planning. Enforces Sunshine Act reporting and PDMA compliance rules at every transactional boundary.

## Installation

```bash
pip install apg-pharma-com
```

## Provides

- `territory_management_workflow`
- `sales_rep_management_workflow`
- `call_activity_workflow`
- `sample_management_workflow`
- `hcp_interaction_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-com/dashboard` | `pharma_com:view` | Overview |
| `/pharma-com/territories` | `pharma_com:territories` | Territory |
| `/pharma-com/territories/<id>` | `pharma_com:territories` | Territory |
| `/pharma-com/reps` | `pharma_com:reps` | Field Force |
| `/pharma-com/calls` | `pharma_com:calls` | Field Force |
| `/pharma-com/samples` | `pharma_com:samples` | Samples |
| `/pharma-com/samples/reconcile` | `pharma_com:samples_admin` | Samples |
| `/pharma-com/interactions` | `pharma_com:interactions` | HCP Engagement |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_territory()`
- `get_territory()`
- `list_territories()`
- `update_territory()`
- `assign_rep()`
- `get_rep()`
- `list_reps()`
- `list_reps_by_territory()`

_(See `service.py` for complete API.)_

## Interoperability

`pharma_com` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pharma_com;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PHARMA_COM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
