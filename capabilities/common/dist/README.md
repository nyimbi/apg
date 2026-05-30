# DIST - Distributed Computing

DIST is the APG capability for governed worker pools, partitioned jobs,
distributed execution, result aggregation, scaling decisions, compute-agent
governance, audit, and lifecycle stream metadata. It gives generated APG
applications a tenant-aware compute runtime that can be composed with
scheduling, monitoring, logging, cache, edge, configuration, audit, and
Bytewax-backed stream adapters.

The implementation is dependency-light and side-effect free. It records
distributed-compute state and evidence without starting live Kubernetes, Ray,
Dask, Spark, Slurm, cloud workers, queues, caches, or external schedulers.

## What DIST Provides

- Worker pools with tenant, owner, capacity quota, health check, queue name,
  autoscaling flag, and status.
- Worker nodes with host, CPU, memory, labels, health state, and partition
  assignment evidence.
- Idempotent distributed jobs with owner, retry policy, partition count, quota
  policy, Bytewax event stream, aggregation strategy, review state, and status.
- Partition records with stable shard keys, assigned workers, attempt counts,
  result hashes, and completion state.
- Result aggregations with partition totals, failed/completed counts,
  deterministic result hashes, and status.
- Scaling decisions based on queued partitions, active workers, and tenant
  capacity quota.
- First-class AI compute agents for Codex, Claude Code, OpenCode, Pi, and
  compatible runtime adapters.
- Deterministic rules for tenant context, job ownership, idempotency keys,
  retry policies, event streams, aggregation strategy, worker health, quota
  policy, partition count, large partition review, AI compute agents,
  state-change audit, cross-tenant isolation, and Bytewax batch mutation
  streams.
- View models for dashboards, jobs, workers, partitions, queues, scaling,
  agents, audit, analytics, and settings.
- Theme metadata for APG Studio and generated Python applications.

## How To Use It

```python
from capabilities.common.dist.service import DistService

service = DistService()
tenant_id = "tenant-dist"

pool = service.create_worker_pool(
    pool_id="pool-main",
    tenant_id=tenant_id,
    name="Main compute pool",
    owner="compute-owner",
    capacity_quota=4,
    health_check="worker-heartbeat",
    queue_name="dist.jobs",
)

worker = service.register_worker(
    worker_id="worker-a",
    tenant_id=tenant_id,
    pool_id=pool["id"],
    hostname="worker-a.internal",
    cpu_slots=4,
    memory_gb=16,
    labels={"zone": "a"},
)

agent = service.register_compute_agent(
    tenant_id=tenant_id,
    agent_id="codex-partition-reviewer",
    name="Codex Partition Reviewer",
    runtime="codex",
    role="result_reviewer",
    scope="Review partition completion evidence and aggregation readiness.",
    contribution_disclosed=True,
    policy_ref="policy:dist:agents:v1",
)

job = service.submit_job(
    job_id="job-001",
    tenant_id=tenant_id,
    name="Reprice portfolio",
    owner="risk-owner",
    worker_pool_id=pool["id"],
    idempotency_key="idempotency-001",
    retry_policy="retry-3-exponential",
    partition_count=4,
    quota_policy="tenant-quota-standard",
    event_bus_topic="bytewax",
    aggregation_strategy="merge_hashes",
)

for partition in service.dispatch_partitions(job["id"], tenant_id):
    service.complete_partition(
        partition_id=partition["id"],
        tenant_id=tenant_id,
        result_payload={"ok": True, "partition": partition["ordinal"]},
    )

aggregation = service.aggregate_results("agg-001", tenant_id, job["id"])
```

Use `api.py` when composing generated application handlers, and use `views.py`
for framework-neutral screen state:

```python
from capabilities.common.dist.views import dashboard_model, compute_agents_model

dashboard = dashboard_model(service, tenant_id)
agents = compute_agents_model(service, tenant_id)
```

## Contract And Composition

`get_capability_contract()` publishes:

- configuration for jobs, workers, coordination, compute agents, governance,
  observability, adapters, UI, and theme;
- JSON-schema-style configuration requirements;
- deterministic rule engine;
- UI routes under `/dist`;
- theme tokens under `dist_compute_grid`;
- Bytewax lifecycle-stream metadata.

DIST depends on `mqeb`, `moni`, and `conf`. Optional adapter boundaries are
`cach`, `logt`, `edge`, `schd`, `bytewax`, and `audl`.

## Guardrail Summary

DIST denies or requires review when:

- tenant context is missing;
- a job lacks owner, idempotency key, retry policy, event stream, aggregation
  strategy, quota policy, or positive partition count;
- a worker pool lacks owner, capacity quota, health check, or queue name;
- a worker lacks hostname, CPU slots, or memory;
- a large partitioned job lacks review evidence;
- dispatch starts without healthy workers;
- an AI compute agent is unregistered, uses an unsupported runtime or role,
  lacks explicit scope, or has undisclosed contributions;
- a job state change lacks reason or audit evidence;
- a cross-tenant access attempt is detected;
- a batch compute mutation does not declare Bytewax.

## Focused Verification

Battery-conscious DIST checks:

```bash
./.venv/bin/python -m py_compile capabilities/common/dist/__init__.py capabilities/common/dist/models.py capabilities/common/dist/distributed_engine.py capabilities/common/dist/service.py capabilities/common/dist/api.py capabilities/common/dist/views.py capabilities/common/dist/capability_contract.py capabilities/common/dist/app.py capabilities/common/dist/test_capability_contract.py capabilities/common/dist/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/dist/test_capability_contract.py capabilities/common/dist/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/dist --json
./.venv/bin/apg capabilities publish-plan capabilities/common/dist --json
```
