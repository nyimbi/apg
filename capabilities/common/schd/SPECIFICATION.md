# Scheduling and Job Orchestration Capability Specification

## Purpose

`schd` is the APG common capability for governed scheduling and operational job
orchestration. It lets generated applications compose schedules, calendar
policies, job definitions, worker pools, runs, recovery flows, AI scheduler
agents, Bytewax lifecycle batches, audit events, UI screens, visual theming,
and Bytewax event-stream policy.

## Scope

The capability must support:

- Tenant-local calendar policies with timezone, business days, blackout
  windows, holiday calendars, owner, and audit events.
- Worker pools with queue, capacity, readiness state, health evidence,
  autoscaling metadata, drain/offline transitions, and state-change reasons.
- Job definitions with name, owner, command or adapter target, criticality,
  expected runtime, external-job approval, monitoring evidence, retry policy,
  attempt limits, tags, enablement, and dead-letter behavior.
- Schedules that bind a job, calendar policy, worker pool, trigger type,
  timezone, interval/cron/event/manual trigger evidence, next-run hint, active
  state, pause/resume, disable, and audit trail.
- Job runs with requested actor, Bytewax event stream, status, attempt,
  parent retry linkage, records processed, error count, exit code, blocked
  count, logs, completion evidence, cancellation reason, dead-letter reason,
  start and completion timestamps.
- Recovery flows for failed runs: retry, dead-letter, cancellation, and
  operator-visible state changes.
- AI scheduler agents as first-class records, with stable ID, readable name,
  supported provider-neutral runtime, supported role, owner, purpose, scope,
  registration actor, status, human-review treatment for privileged roles, and
  visible contribution disclosure.
- Bytewax-backed event-stream configuration for batch scheduler mutation,
  runtime events, and lifecycle batches across calendars, worker pools, jobs,
  schedules, runs, retries, dead letters, scheduler agents, and audit.
- UI route contracts and dependency-light view models for generated
  applications.

## Dependencies

Required:

- `wflo` for composing schedules with workflow execution.
- `mqeb` for production event/message composition.
- `moni` for production run and worker monitoring.
- `audl` for durable scheduler audit trails.

Optional:

- `ntfy`, `cach`, `comp`, and `them`.

## Configuration

The authoritative configuration lives in `capability_contract.py` and includes:

- `schedules`
- `jobs`
- `job_runs`
- `workers`
- `scheduler_agents`
- `agents`
- `governance`
- `observability`
- `streaming`
- `adapters`
- `ui`
- `theme`

## Rules

The deterministic rule engine covers:

- tenant context
- schedule owner, timezone, calendar policy, worker pool, manual-run reason,
  pause reason, and resume audit
- job owner, command, retry policy, critical monitoring, external approval,
  long-running runtime review, and dead-letter enablement
- worker queue, positive capacity, health evidence, drain reason, and ready
  state before runs
- active schedules before run start
- Bytewax event stream enforcement
- run audit evidence and non-negative metrics
- retry eligibility
- cancellation and dead-letter reasons
- scheduler-agent stable ID, readable name, runtime, role, scope, owner,
  purpose, contribution disclosure, and privileged-role human approval
- Bytewax lifecycle batch mutation count, supported operation, and lifecycle
  stream enforcement
- scheduler state-change audit
- tenant isolation
- Bytewax batch mutation enforcement

## Runtime

`service.SchdService` is the generated-application runtime. It stores
deterministic in-memory state for:

- calendar policies
- worker pools
- job definitions
- schedules
- job runs
- scheduler agents
- lifecycle batches
- audit events

The runtime enforces the same guardrails exposed by the contract rule engine
and keeps live providers behind adapter boundaries.

## UI

The UI contract exposes:

- dashboard
- schedules
- jobs
- runs
- workers
- calendars
- agents
- lifecycle
- audit
- analytics
- settings

## Production Boundary

This packet does not start live schedulers, durable queues, distributed worker
processes, notification providers, monitoring backends, audit stores, external
AI-agent CLIs, or live Bytewax workers. Those are production adapters behind the
APG composition layer.

## Acceptance Gates

- `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe the package clearly.
- `capability_contract.py` exposes configuration, deterministic rules, UI,
  theme, streaming, and adapter metadata.
- Runtime/API/view tests prove positive lifecycle behavior and negative
  guardrail behavior.
- First-class scheduler-agent composition is provider-neutral across `codex`,
  `claude_code`, `opencode`, and `pi`; external clients remain behind AICR
  adapter contracts.
- Lifecycle batch governance uses Bytewax metadata only and does not introduce
  broker-specific queue or broker-core processing.
- `semantic_model.json`, `package_manifest.json`, and `release_report.json`
  match the current contract.
- Focused compile, pytest, self-test, implementation audit, publish-plan,
  stale-marker scan, and diff check pass.
