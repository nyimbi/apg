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
- Workload scheduling via cron expressions across target node sets.
- Offline-first synchronization with conflict policy, cache policy, replay
  counts, and review for long offline windows.
- Edge-local data caching with TTL and invalidation.
- Local ML inference execution (Ollama/ONNX adapter target).
- Computation offload with latency-requirement routing.
- Latency monitoring with running statistics (min/max/avg/p95).
- Federated metric aggregation across nodes without centralizing raw data.
- Power-aware workload scheduling against a watt budget.
- Firmware staging with hash verification and rollback version tracking.
- Locality routing scored by resource pressure.
- Data sovereignty compliance checking against required residency codes.
- Edge security event recording with severity classification.
- Bandwidth optimization policies (compression, deduplication, QoS).
- Fleet-wide health summaries and per-node health probes.
- Auto-scaling with scale-out/scale-in decisions and cooldown.
- Failover with workload migration to a healthy target node.
- Bulk node registration and workload deployment.
- JSON and CSV export for nodes and deployments.
- Resource pressure summaries and audit digests.
- AI edge-agent registration for Codex, Claude Code, OpenCode, Pi, and future
  runtimes behind the same contract.
- Bytewax stream guardrail for batch edge mutation.
- UI routes and visual theme tokens for generated APG applications.

## Quick Start

```python
from capabilities.common.edge import EdgeService

service = EdgeService()

# Register a node
service.register_edge_node(
    node_id="node-plant-a-01",
    tenant_id="tenant-acme",
    location={"site": "plant-a", "zone": "line-1"},
    capabilities=["sensor_aggregation", "local_inference"],
    network_type="ethernet",
    name="Plant A Gateway",
    owner="edge-ops",
    node_type="gateway",
    location_policy="site-policy-plant-a",
    attested=True,
    capacity={"cpu": 8, "memory": 16384, "storage": 512},
)

# Deploy a workload
service.deploy_workload(
    workload_id="wl-line-monitor",
    tenant_id="tenant-acme",
    target_nodes=["node-plant-a-01"],
    constraints={"required_capabilities": ["sensor_aggregation"]},
    name="Line Monitor",
    version="1.0.0",
    owner="automation",
    artifact_payload={"image": "line-monitor:1.0.0"},
    artifact_signed=True,
    deployment_policy="signed-canary",
    resource_quota={"cpu": 2, "memory": 1024, "storage": 10},
)

# Dashboard summary
print(service.dashboard_summary("tenant-acme"))
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

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
Supported roles: `fleet_optimizer`, `node_health_reviewer`,
`workload_placement_reviewer`, `offline_sync_reviewer`, `security_reviewer`.

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
blocked  = service.validate_batch_edge_mutation("other-stream")

assert allowed["decision"] == "allow"
assert blocked["decision"]  == "deny"
```

The contract declares topic `apg.edge.lifecycle` and state for nodes, fleets,
workloads, deployments, sync sessions, edge agents, and audit events.

## Core API

| Method | Purpose |
|---|---|
| `register_edge_node(...)` | Register a node with location, capabilities, capacity |
| `node_health_monitor(node_id, tenant_id)` | Synthetic health probes: connectivity, resources, security |
| `deploy_workload(workload_id, target_nodes, constraints, ...)` | Register workload and place on matching healthy nodes |
| `workload_status(workload_id, tenant_id)` | Active deployment count and per-deployment detail |
| `workload_schedule(...)` | Cron-based scheduling across a node set |
| `offload_computation(request_id, payload, latency_requirement_ms, ...)` | Route computation to lowest-latency node or cloud fallback |
| `edge_to_cloud_sync(node_id, data_type, data, ...)` | Initiate edge-to-cloud data sync session |
| `sync_state(...)` | Full sync session record with conflict and review tracking |
| `review_offline_window(sync_id, tenant_id, reviewer)` | Clear review flag on a long-offline sync session |
| `sync_when_connected(...)` | Queue data sync for offline nodes; immediate if online |
| `auto_scaling(workload_id, metric, threshold, ...)` | Evaluate scale-out/scale-in decision |
| `failover(node_id, failover_target, ...)` | Migrate workloads from failed node to healthy target |
| `create_fleet(...)` | Create a fleet with owner and policy version |
| `attach_node_to_fleet(node_id, fleet_id, tenant_id)` | Add node to fleet membership |
| `edge_cache(tenant_id, node_id, cache_key, data, ttl_seconds)` | Write data to edge cache with TTL |
| `edge_cache_invalidate(...)` | Invalidate a cache entry by key |
| `local_inference(...)` | Execute ML inference at edge node (Ollama/ONNX adapter) |
| `offline_mode(tenant_id, node_id, enabled, ...)` | Enable/disable offline operation for all node workloads |
| `bandwidth_optimise(...)` | Apply compression + deduplication bandwidth policy |
| `bandwidth_optimisation(node_id, policy, ...)` | Full-control bandwidth policy application |
| `power_aware_compute(...)` | Schedule workloads against a watt budget by priority |
| `federated_aggregate(...)` | Aggregate a metric across nodes (mean/sum/min/max/count) |
| `edge_health(tenant_id)` | Fleet-wide health counts and average pressure |
| `node_pressure(node_id, tenant_id)` | Per-resource pressure for a single node |
| `firmware_update(...)` | Stage firmware with hash verification and rollback version |
| `locality_routing(...)` | Route request to lowest-pressure capable node |
| `latency_monitor(...)` | Record latency sample; returns running min/max/avg/p95 |
| `edge_security(...)` | Record security event (info/warning/critical/breach) |
| `data_sovereignty_check(...)` | Verify data residency compliance for a node |
| `edge_analytics(period, tenant_id)` | Full analytics: nodes, deployments, scaling, failovers, inference |
| `dashboard_summary(tenant_id)` | Counts for all entity types, suitable for a dashboard header |
| `health_check(tenant_id)` | Service liveness check |
| `bulk_register_nodes(tenant_id, nodes)` | Register a list of nodes in one call |
| `bulk_deploy_workloads(tenant_id, workloads)` | Deploy a list of workloads in one call |
| `export_nodes(tenant_id, fmt)` | Export nodes as JSON or CSV |
| `export_deployments(tenant_id, fmt)` | Export deployments as JSON or CSV |
| `register_edge_agent(...)` | Register an AI agent as an edge contributor |
| `validate_batch_edge_mutation(event_stream)` | Enforce Bytewax-only batch mutation |

## New Methods

### Local Inference

Run an ML model at the edge node. Production adapters target Ollama or an ONNX
runtime; the service layer provides the audit trail regardless.

```python
result = service.local_inference(
    tenant_id="tenant-acme",
    node_id="node-plant-a-01",
    request_id="inf-001",
    model_name="defect-classifier-v2",
    input_data={"image_hash": "abc123", "region": "top-left"},
    inference_type="classification",
    executed_by="line-monitor",
)
# result["output"] = {"label": "class_A", "confidence": 0.87, "latency_ms": 12}
```

### Latency Monitor

Record telemetry samples and get running statistics per node:

```python
for ms in [12.3, 14.1, 11.8, 13.0]:
    stats = service.latency_monitor(
        tenant_id="tenant-acme",
        node_id="node-plant-a-01",
        latency_ms=ms,
        operation="inference",
    )
# stats["statistics"] = {"min_ms": 11.8, "max_ms": 14.1, "avg_ms": 12.8, ...}
```

### Power-Aware Compute

Schedule workloads in priority order against an available watt budget. Workloads
that don't fit are deferred, not dropped:

```python
result = service.power_aware_compute(
    tenant_id="tenant-acme",
    node_id="node-solar-01",
    power_budget_watts=40.0,
    workload_priority={"wl-critical": 10, "wl-batch": 3, "wl-reporting": 1},
)
# result["scheduled_count"], result["deferred_count"]
```

### Data Sovereignty Check

Verify that data processed at a node satisfies residency requirements before
syncing or storing it:

```python
check = service.data_sovereignty_check(
    tenant_id="tenant-acme",
    node_id="node-plant-a-01",
    data_classification="PII",
    data_country_codes=["KE", "NG"],
    required_residency=["KE"],
)
if not check["compliant"]:
    raise PermissionError(f"Sovereignty violation: {check['country_violations']}")
```

### Federated Aggregate

Compute a metric across multiple nodes without transmitting raw node data:

```python
agg = service.federated_aggregate(
    tenant_id="tenant-acme",
    aggregation_id="agg-cpu-fleet-001",
    node_ids=["node-plant-a-01", "node-plant-b-01"],
    aggregation_fn="mean",
    metric="cpu",
)
# agg["result"] = 0.42  (mean cpu utilisation ratio across both nodes)
```

### Firmware Update

Stage a firmware version with hash and rollback version for safe OTA delivery:

```python
update = service.firmware_update(
    tenant_id="tenant-acme",
    node_id="node-plant-a-01",
    firmware_version="2.4.1",
    firmware_hash="sha256:deadbeef...",
    rollback_version="2.4.0",
    initiated_by="release-bot",
    staged=True,
)
# update["status"] == "staged"
```

## World-Class Enhancements (v2.0)

The following 15 improvements represent the target production architecture.
The v1 service ships the adapter-ready scaffold; each item identifies what the
corresponding adapter or future service version must implement.

1. **Async-First Service Layer** — All I/O-bound methods should be `async def`
   with an `asyncio`-native concurrency model. A `run_sync()` shim serves
   synchronous callers. Eliminates event-loop bridging when MQTT/aiohttp/asyncpg
   adapters go async.

2. **Streaming Telemetry with AsyncGenerator Drain** — Replace list accumulation
   in `latency_monitor` with an `asyncio.Queue`-backed drain that yields
   batch-compressed telemetry windows. Provides natural back-pressure; zero
   overhead on quiet nodes.

3. **Cryptographic Node Attestation Pipeline** — `attested: bool` is caller-set
   today. The production path: send challenge nonce → receive TPM2-signed
   response → verify against hardware root-of-trust cert → set `attested=True`
   with a trust score and evidence record.

4. **CRDT Sync** — Replace `last_write_wins` with pluggable CRDT strategies:
   G-Counter for monotonic sensor accumulations, OR-Set for capability
   registries, LWW-Register for config keys. `sync_state` negotiates merge
   strategy per field type; no data loss on partition reconciliation.

5. **Geo-Aware Locality Routing** — Add lat/lon to `EdgeNode.location`, compute
   Haversine distance, and score as `α * pressure + β * normalized_distance`.
   Expose `α`/`β` as per-tenant routing weights.

6. **OTA Firmware Delta Updates** — `firmware_update` stages full blobs today.
   Delta OTA computes a bsdiff between rollback and target versions, transmits
   2–15% of full size, and the node applies the patch locally.

7. **Federated Learning Aggregation (FedAvg)** — `federated_aggregate` operates
   on load ratios today. True FL collects model weight gradients from nodes
   without sharing raw training data, applies FedAvg/FedProx, and returns an
   updated global model manifest.

8. **Zero-Trust mTLS Certificate Lifecycle** — `secure_transport: bool` is
   binary. Production requires per-node short-lived mTLS certs issued by an
   internal CA, rotated on a configurable TTL, with revocation checking on every
   sync. Certificate fingerprints become mandatory audit fields.

9. **Predictive Auto-Scaling** — `auto_scaling` reacts to instantaneous
   threshold breaches. Predictive scaling trains a Holt-Winters model on
   historical metrics, forecasts load 5–15 minutes ahead, and pre-provisions
   replicas before the spike with confidence intervals.

10. **Energy-Aware Workload Migration** — `power_aware_compute` uses a static
    watt budget. The dynamic version reads real-time solar/battery
    state-of-charge, migrates non-critical workloads to grid-connected nodes
    when battery drops below threshold, and restores on renewable recovery.

11. **Distributed Tracing Integration** — No W3C Traceparent propagates today.
    Every service method should accept `trace_context: dict[str, str] | None`,
    create a child span, attach it to audit events, and propagate through
    federated aggregate calls.

12. **Policy-as-Code with OPA** — The capability contract is hand-coded Python
    today. The production path externalises policy to OPA bundles evaluated via
    `opa_client.evaluate_bundle(policy_bundle, input_data)`. Tenants upload
    custom `.rego` policies; the service stays policy-agnostic.

13. **Data Lineage and Provenance Tracking** — `local_inference` results have no
    lineage today. A `DataProvenance` model should record: input dataset hash,
    model version, node ID, inference timestamp, and a DAG of upstream
    transformations. `get_data_lineage(artifact_id)` returns the full DAG for
    regulatory audit trails.

14. **Multi-Region Fleet Replication** — `EdgeFleet` is single-region. Add
    `replicate_fleet(fleet_id, target_regions, replication_policy)` that mirrors
    topology across regions, applies geo-fencing per region, and uses a
    CRDT-backed membership set for consensus-free eventual consistency.

15. **Canary Deploy with Automatic Rollback** — `deploy_workload` has no
    post-deploy health loop. `canary_deploy(workload_id, canary_node_id,
    full_node_ids, health_check_fn, rollback_threshold_pct)` routes configurable
    traffic to the canary, monitors error rate and p95 latency, and
    automatically rolls back the full fleet if error rate exceeds threshold.

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
