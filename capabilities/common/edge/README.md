# EDGE Edge Computing

`edge` is the APG common edge computing capability. It lets generated
applications compose tenant-scoped edge nodes, fleets, signed workloads,
deployments, offline execution, state synchronization, resource pressure,
audit evidence, Bytewax stream governance, visual theme metadata, and AI-agent
assistance.

The package is dependency-light. It defines the executable service, rule
engine, UI route metadata, theme metadata, Bytewax stream declaration, API
helpers, view models, and semantic evidence. Physical device enrollment,
container runtimes, model runtimes, durable telemetry stores, remote update
systems, and stream-worker deployments are adapter responsibilities.

## What It Provides

- Edge node registration with owner, location policy, health, secure transport,
  attestation, capacity, and capabilities.
- Fleet creation and node membership governance.
- Signed workload registration with resource quota and deployment policy.
- Workload placement on healthy attested nodes with capacity checks.
- Offline-first synchronization with conflict policy, cache policy, replay
  counts, and review for long offline windows.
- Resource pressure summaries and audit digests.
- AI edge-agent registration for Codex, Claude Code, OpenCode, Pi, and future
  runtimes behind the same contract.
- Bytewax stream guardrail for batch edge mutation.
- UI routes and visual theme tokens for generated APG applications.

## Quick Use

```python
from capabilities.common.edge import EdgeService

service = EdgeService()

service.register_node(
    node_id="node-plant-a-01",
    tenant_id="tenant-acme",
    name="Plant A Gateway",
    owner="edge-ops",
    node_type="gateway",
    location={"site": "plant-a", "zone": "line-1"},
    location_policy="site-policy-plant-a",
    attested=True,
    capacity={"cpu": 8, "memory": 16384, "storage": 512},
    capabilities=["sensor_aggregation", "local_inference"],
)

service.register_workload(
    workload_id="wl-line-monitor",
    tenant_id="tenant-acme",
    name="Line Monitor",
    version="1.0.0",
    owner="automation",
    artifact_payload={"image": "line-monitor:1.0.0"},
    artifact_signed=True,
    deployment_policy="signed-canary",
    resource_quota={"cpu": 2, "memory": 1024, "storage": 10},
)

service.deploy_workload(
    deployment_id="dep-line-monitor",
    tenant_id="tenant-acme",
    workload_id="wl-line-monitor",
    node_id="node-plant-a-01",
    deployed_by="release-manager",
)
```

## AI Agent Registration

AI agents are first-class edge contributors only after registration:

```python
agent = service.register_edge_agent(
    tenant_id="tenant-acme",
    name="Placement reviewer",
    runtime="codex",
    role="workload_placement_reviewer",
    scope="review workload placement, capacity, and attestation evidence",
    contribution_disclosed=True,
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles are `fleet_optimizer`, `node_health_reviewer`,
`workload_placement_reviewer`, `offline_sync_reviewer`, and
`security_reviewer`.

## Guardrails

The deterministic rules deny or require review when:

- tenant context is missing;
- edge node owner, attestation, or location policy is missing;
- edge transport is not secure;
- fleet owner or policy version is missing;
- workload owner, artifact signature, or resource quota is missing;
- sync conflict policy or cache policy is missing;
- offline window exceeds the configured review threshold without review;
- an AI edge agent is unregistered, unsupported, unscoped, or undisclosed;
- lifecycle state changes lack audit evidence;
- batch edge mutation does not use Bytewax.

## Bytewax Batch Mutation

Batch edge mutation must use the Bytewax event stream:

```python
allowed = service.validate_batch_edge_mutation("bytewax")
blocked = service.validate_batch_edge_mutation("other-stream")

assert allowed["decision"] == "allow"
assert blocked["decision"] == "deny"
```

The contract declares topic `apg.edge.lifecycle` and state for nodes, fleets,
workloads, deployments, sync sessions, edge agents, and audit events.

## Composition

Generated APG applications should compose `edge` through:

- capability ID: `edge`;
- provided services: edge nodes, fleets, workloads, deployments, offline
  execution, sync, and edge agents;
- required services: `auth`, `conf`, `audl`, `dist`, `cach`, and `moni`;
- API prefix: `/edge/api/v1`;
- UI routes: dashboard, nodes, fleets, workloads, deployments, sync, agents,
  rules, analytics, audit, and settings;
- theme: `edge_operations_console`;
- stream processor: `bytewax`.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/edge/__init__.py capabilities/common/edge/capability_contract.py capabilities/common/edge/models.py capabilities/common/edge/service.py capabilities/common/edge/api.py capabilities/common/edge/views.py capabilities/common/edge/app.py capabilities/common/edge/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/edge/test_capability_contract.py
./.venv/bin/python -c "from capabilities.common.edge import EdgeService; service = EdgeService(); service.register_edge_agent('tenant-proof', 'Proof agent', 'codex', 'security_reviewer', 'review edge security'); print(service.dashboard_summary('tenant-proof'))"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/edge --json
./.venv/bin/apg capabilities publish-plan capabilities/common/edge --json
```
