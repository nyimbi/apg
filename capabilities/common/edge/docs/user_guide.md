# Edge Computing

**Capability ID**: `edge` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`edge` is the APG common edge computing capability. It lets generated applications compose tenant-scoped edge nodes, fleets, signed workloads, deployments, offline execution, state synchronization, resource pressure,

## Installation

```bash
pip install apg-common-edge
```

## Provides

- `edge_nodes`
- `edge_fleets`
- `edge_workloads`
- `edge_deployments`
- `offline_execution`

## Requires

- `auth`
- `conf`
- `audl`
- `dist`
- `cach`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/edge/dashboard` | `edge:view` | Overview |
| `/edge/nodes` | `edge:manage_nodes` | Nodes |
| `/edge/fleets` | `edge:manage_nodes` | Nodes |
| `/edge/workloads` | `edge:deploy_workloads` | Workloads |
| `/edge/deployments` | `edge:deploy_workloads` | Workloads |
| `/edge/sync` | `edge:sync` | Synchronization |
| `/edge/agents` | `edge:govern` | Governance |
| `/edge/rules` | `edge:govern` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_edge_node()`
- `node_health_monitor()`
- `deploy_workload()`
- `workload_status()`
- `offload_computation()`
- `edge_to_cloud_sync()`
- `auto_scaling()`
- `failover()`

_(See `service.py` for complete API.)_

## Interoperability

`edge` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use edge;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `EDGE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
