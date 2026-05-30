# DIST Capability Specification

## Purpose

DIST defines the APG distributed-computing capability. It gives generated
applications an executable, governed lifecycle for defining worker pools,
registering workers, submitting partitioned jobs, dispatching partitions,
recording partition results, aggregating results, recording scaling decisions,
registering AI compute agents, composing distributed-compute UIs, and auditing
every meaningful compute transition.

## Scope

In scope:

- Tenant-aware in-memory service state for package checks and generated apps.
- Worker-pool, worker-node, distributed-job, job-partition,
  result-aggregation, scaling-decision, compute-agent, and audit models.
- Deterministic partition IDs, partition result hashes, aggregation hashes, and
  audit payload hashes.
- Deterministic rule evaluation for distributed-compute guardrails.
- First-class AI compute-agent registration with runtime, role, scope,
  disclosure, and policy reference.
- Bytewax stream contract metadata for lifecycle and batch mutation events.
- Framework-neutral API helpers and UI view models.
- Theme tokens and component metadata for generated APG applications.
- Package evidence through `app.py`, `semantic_model.json`,
  `package_manifest.json`, and `release_report.json`.

Out of scope for this dependency-light package:

- Live Kubernetes, Ray, Dask, Spark, Slurm, or cloud compute execution.
- External queue, scheduler, cache, or database mutation.
- Worker process management.
- Persistent database storage.
- Rendered browser UI.

Those behaviors must attach through explicit adapters so local APG tooling stays
safe, deterministic, and side-effect free.

## Functional Requirements

### Worker Pools

- Create tenant-scoped worker pools with ID, name, owner, capacity quota,
  health check, queue name, autoscaling flag, and status.
- Deny missing tenant context, missing owner, non-positive quota, missing
  health check, and missing queue name.
- Store duplicate worker-pool IDs safely across tenants.

### Workers

- Register tenant-scoped workers only under tenant-local pools.
- Require hostname, positive CPU slots, and positive memory.
- Store duplicate worker IDs safely across tenants.

### Jobs

- Submit tenant-scoped distributed jobs only to tenant-local worker pools.
- Require owner, idempotency key, retry policy, positive partition count, quota
  policy, event stream, and aggregation strategy.
- Return the existing job when the same tenant submits the same idempotency key.
- Route large partition jobs to review unless review evidence is recorded.
- Store duplicate job IDs safely across tenants.

### Partitions

- Create deterministic partition IDs for each accepted job.
- Dispatch partitions only after review is approved and healthy tenant-local
  workers are available.
- Assign partitions across healthy workers in the selected pool.
- Complete or fail partitions with deterministic result hashes.

### Aggregation

- Aggregate results only when every partition for the job has completed or
  failed.
- Store completed and failed partition counts.
- Generate deterministic aggregation result hashes.
- Mark jobs as completed or completed with failures.

### Scaling

- Record scaling decisions only for tenant-local worker pools.
- Derive scale up, scale down, or hold decisions from queued partitions,
  active workers, and tenant capacity quota.

### AI Compute Agents

- Register AI compute agents as first-class DIST records.
- Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
- Supported roles: `job_planner`, `partition_operator`,
  `worker_pool_operator`, `result_reviewer`, `incident_reviewer`.
- Require registration flag, supported runtime, supported role, explicit scope,
  and contribution disclosure.
- Isolate agent registrations by tenant even when agent IDs are reused.

### UI And Theme

DIST must expose route metadata for:

- `dashboard`
- `jobs`
- `workers`
- `partitions`
- `queues`
- `scaling`
- `agents`
- `audit`
- `analytics`
- `settings`

DIST must expose view-model functions for these operational surfaces and
publish the `dist_compute_grid` theme with job, worker, partition, scaling,
agent, and audit component metadata.

### Streaming

DIST must declare Bytewax as the lifecycle stream processor. The stream
contract must include worker-pool, worker, job, partition, aggregation,
scaling-decision, compute-agent, and audit state families. Batch compute
mutation must be denied unless the event stream is `bytewax`.

## Rule Engine Requirements

The deterministic rules must cover:

- tenant context;
- job owner, idempotency key, retry policy, partition count, quota policy,
  event stream, aggregation strategy, and large-partition review;
- worker-pool health;
- AI compute-agent registration, runtime, role, scope, and disclosure;
- state-change reason and audit evidence;
- cross-tenant access denial;
- Bytewax event stream requirement for batch mutation.

The rule evaluator must support equality plus numeric `_lt`, `_lte`, `_gt`,
`_gte`, and inequality `_ne` conditions.

## Non-Functional Requirements

- Importing the package must not require live adapters.
- Service operations must remain tenant-scoped.
- Generated package evidence must stay synchronized with the contract.
- API and view-model functions must return plain Python dictionaries/lists.
- Focused tests must cover the main lifecycle, guardrail failures, AI agents,
  tenant-safe duplicate IDs, Bytewax metadata, idempotency, and generated
  evidence.
- Documentation must explain use, architecture, boundaries, and verification.

## Acceptance Criteria

- `README.md`, `SPECIFICATION.md`, `PLAN.md`, and `cap_spec.md` describe the
  same executable packet.
- `register_capability()` exposes dependencies, optional adapters,
  permissions, endpoints, UI metadata, theme, and Bytewax stream contract.
- Focused DIST tests pass.
- `app.self_test()` passes.
- `semantic_model.json` exposes DIST routes, rules, configuration, theme, and
  Bytewax stream metadata.
- Implementation audit and publish-plan pass for DIST.
- Stale-marker search finds no unsupported overclaims, unfinished markers, or
  unsupported stream-provider references in DIST.
