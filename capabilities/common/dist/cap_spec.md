# Distributed Computing Capability Specification

- **Capability Name**: Distributed Computing
- **Capability ID**: `dist`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`dist` provides dependency-light distributed execution state for APG
applications. It owns tenant worker pools, worker nodes, idempotent distributed
jobs, partitions, result aggregations, scaling decisions, queue state, and audit
evidence behind the executable capability contract.

The package is intentionally local and deterministic. It does not start live
Kubernetes, Ray, Dask, Spark, Slurm, Bytewax, Redis, RabbitMQ, Kafka, or cloud
compute workers. External schedulers, queues, observability systems, caches, and
edge runtimes should be composed through APG capabilities such as `mqeb`,
`schd`, `moni`, `cach`, `edge`, `logt`, `conf`, and Bytewax-backed runtime
adapters.

## Provided Services

- `distributed_jobs`
- `worker_pools`
- `partitioned_execution`
- `coordination`
- `distributed_scaling`

## Required Services

- `mqeb`
- `moni`
- `conf`

Optional composition partners include `cach`, `logt`, `edge`, and `schd`.

## Runtime Behavior

The service layer exposes an in-memory distributed-computing runtime for package
evidence and generated-application composition:

1. Create tenant worker pools with owners, health checks, queue names, and
   capacity quotas.
2. Register healthy worker nodes with resource slots and labels.
3. Submit idempotent distributed jobs with owners, retry policies, event-bus
   topics, quota policy, aggregation strategy, and partition counts.
4. Hold large partition jobs for explicit review when the rule engine requires
   it.
5. Create deterministic partition records and dispatch them across healthy
   workers.
6. Complete or fail partitions with deterministic result hashes.
7. Aggregate completed partition results into a stable job result hash.
8. Record scaling decisions from queue pressure, active workers, and tenant
   quota.
9. Publish dashboard, job-console, worker-pool, queue, partition, scaling, and
   audit state.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context, job ownership, idempotency keys,
retry policy, worker health checks, capacity quotas, event-bus coordination,
distributed locking, result aggregation, dead-letter queues, monitoring, UI,
and theme metadata are explicit contract concerns.

## Rules

- `tenant_context_required`
- `job_requires_owner`
- `idempotency_key_required`
- `worker_pool_requires_health`
- `quota_policy_required`
- `large_partition_job_requires_review`

The service calls the rule engine before job submission and tenant-sensitive
operations. Deny decisions raise `PermissionError`; review decisions create
pending-review jobs until explicitly approved.

## UI

The package exposes 8 APG Python UI route contract(s) through `views.py` and the
package semantic model:

- `/dist/dashboard`
- `/dist/jobs`
- `/dist/workers`
- `/dist/partitions`
- `/dist/queues`
- `/dist/scaling`
- `/dist/analytics`
- `/dist/settings`

`views.py` provides dashboard and job-detail models for compute dashboards, job
consoles, worker pools, partition monitors, queue monitors, scaling panels,
result aggregation, and audit timelines.

## Theme

The package uses the `dist_compute_grid` APG theme contract.
