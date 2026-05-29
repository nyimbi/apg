# Edge Computing Capability Specification

- **Capability Name**: Edge Computing
- **Capability ID**: `edge`
- **Category**: common
- **Version**: 1.0.0

## Purpose

EDGE provides a tenant-aware, dependency-light runtime for edge nodes,
edge fleets, signed workload artifacts, edge deployments, offline-first state
synchronization, review-required offline windows, and audit evidence. It turns
the executable APG edge contract into service, API, and view behavior that
generated APG applications can compose without requiring a live edge cluster.

## Current Runtime Behavior

The package currently supports:

- registering attested edge nodes with owners, location policy, health state,
  secure transport, capacity, and capabilities;
- grouping nodes into policy-versioned fleets;
- registering signed workload artifacts with deployment policies, resource
  quotas, and offline-mode posture;
- deploying workloads to healthy attested nodes when resource quotas fit
  available node capacity;
- synchronizing edge state with conflict policy, cache policy, secure
  transport, replay counts, conflict lists, and offline-window review state;
- reviewing long offline windows after the rule engine marks a sync session as
  review-required;
- computing node resource pressure for operational dashboards;
- exposing dashboard, node manager, workload console, sync monitor, route,
  theme, and audit view models;
- recording deterministic audit digests for node, fleet, workload, deployment,
  sync, and review operations.

Runtime state is in-memory and tenant-scoped. This keeps the package executable
for compiler smoke tests, publish planning, and APG composition while preserving
clear adapter boundaries for production infrastructure.

## Provided Services

- `edge_nodes`
- `edge_workloads`
- `offline_execution`
- `edge_sync`
- `edge_deployment`
- `edge_operations`

## Required Services

The package contract depends on APG platform capabilities rather than direct
network services:

- `tenant_context`
- `dist` for distributed execution composition
- `cach` for cache policy composition
- `conf` for configuration policy composition

Optional integration boundaries are intentionally external:

- `iotd` for IoT device ingestion and device registry adapters;
- `cicd` for artifact build, signing, and promotion pipelines;
- `moni` for production telemetry and alerting;
- `geos` for geospatial policy enrichment;
- physical edge runtimes, Kubernetes distributions, device agents, message
  brokers, and remote attestation providers.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

Key configuration sections:

- `nodes`: owner, attestation, health-check, and location-policy requirements;
- `workloads`: deployment policy, resource quota, offline-mode, and artifact
  signature requirements;
- `sync`: conflict policy, cache policy, maximum offline hours, and replay
  support;
- `governance`: tenant context, audit events, configuration policy, and secure
  transport requirements;
- `ui`: dashboard, node manager, workload console, and sync monitor controls;
- `theme`: `edge_operations_console` theme metadata.

## Rules

The deterministic rule engine enforces:

- `tenant_context_required`
- `node_requires_attestation`
- `workload_requires_signed_artifact`
- `sync_requires_conflict_policy`
- `edge_transport_requires_security`
- `long_offline_window_requires_review`

`EdgeService` evaluates these rules before registering nodes, registering or
deploying workloads, and synchronizing edge state. Deny decisions raise
`PermissionError`. Long offline windows create sync sessions with
`review_required` status until reviewed.

## UI

The package exposes 8 APG Python UI route contracts through `views.py` and the
package semantic model:

- `/edge/dashboard`
- `/edge/nodes`
- `/edge/fleets`
- `/edge/workloads`
- `/edge/deployments`
- `/edge/sync`
- `/edge/analytics`
- `/edge/settings`

The dependency-light view helpers provide dashboard, node manager, workload
console, and sync monitor models.

## Theme

The package uses the `edge_operations_console` APG theme contract with route,
node-map, workload-table, sync-monitor, and fleet-panel component metadata.

## Verification

Recommended focused verification for this package:

```bash
./.venv/bin/python -m py_compile capabilities/common/edge/__init__.py capabilities/common/edge/models.py capabilities/common/edge/edge_engine.py capabilities/common/edge/service.py capabilities/common/edge/api.py capabilities/common/edge/views.py capabilities/common/edge/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/edge/test_capability_contract.py capabilities/common/edge/tests
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities publish-plan capabilities/common/edge --json
```

## Known Integration Boundary

EDGE does not yet start a physical edge runtime, open remote network tunnels,
perform remote attestation against a hardware root of trust, sign artifacts via
a CI/CD provider, persist node telemetry to a time-series database, or execute
Bytewax flows. Those should remain behind composed APG capabilities and
adapters so the current package stays deterministic, testable, and publish-plan
ready.
