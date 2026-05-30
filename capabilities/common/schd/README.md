# Scheduling and Job Orchestration Capability

`schd` is the APG common capability for governed schedules, calendar triggers,
job definitions, worker pools, run recovery, and scheduler operations. It gives
generated applications a dependency-light runtime that can define jobs, attach
them to tenant calendar policies and worker pools, trigger runs, recover failed
runs, expose scheduler UI models, and enforce deterministic rules before live
adapters are connected.

## What It Provides

- Tenant-scoped calendar policies with timezone, business-day, blackout, and
  holiday-calendar metadata.
- Job definitions with owner, command/adapter target, criticality, expected
  runtime, retry policy, monitoring, approval, and dead-letter controls.
- Worker pools with queues, capacity, health evidence, autoscaling metadata,
  state transitions, drain reasons, and audit events.
- Schedules that bind jobs, calendars, worker pools, trigger type, timezone,
  interval/cron/event/manual trigger evidence, enabled state, pause/resume, and
  disable controls.
- Run lifecycle behavior for start, completion, cancellation, retry,
  dead-letter, metrics, logs, completion evidence, parent retry linkage, and
  Bytewax stream policy.
- AI scheduler agents as first-class records for schedule design, run
  observation, retry advice, capacity planning, and calendar audit assistance.
- Rule-engine, UI-route, visual-theme, adapter, semantic-model, release, and
  publish-plan metadata for APG composition.

## Runtime Surface

Use `service.SchdService` for local generated-application behavior:

```python
from capabilities.common.schd.service import SchdService

service = SchdService()
calendar = service.create_calendar_policy("tenant-a", "weekday", "UTC", "ops")
worker = service.register_worker_pool("tenant-a", "etl", "etl.jobs", 4)
job = service.define_job(
    "tenant-a",
    "ledger-close",
    "python close.py",
    "finance",
    monitoring_attached=True,
)
schedule = service.create_schedule(
    "tenant-a",
    "ledger-close-hourly",
    job["id"],
    calendar["id"],
    worker["id"],
    "interval",
    "UTC",
    "finance",
    interval_minutes=60,
)
run = service.trigger_run("tenant-a", schedule["id"], "scheduler")
service.complete_run("tenant-a", run["id"], records_processed=100)
```

Dependency-light API helpers in `api.py` wrap the same service methods for
generated endpoints. `views.py` provides dashboard, schedule console, job
library, run monitor, worker dashboard, calendar manager, scheduler-agent,
audit, analytics, and settings view models.

## Guardrails

The deterministic rule engine blocks or flags:

- missing tenant context, owner, timezone, calendar policy, worker pool, queue,
  capacity, health evidence, job command, retry policy, monitoring, approval,
  runtime review, manual-run reason, pause reason, cancellation reason, and
  dead-letter reason;
- triggering disabled, paused, or offline-worker schedules;
- completing runs without valid non-negative metrics and audit evidence;
- retrying non-failed runs or exceeding attempt limits;
- unregistered, unsupported, unscoped, or undisclosed scheduler agents;
- cross-tenant access and batch scheduler mutation without Bytewax.

## Composition

Required capabilities are `wflo`, `mqeb`, `moni`, and `audl`. Optional
production adapters include `ntfy`, `cach`, `comp`, and `them`.

The local package does not start live schedulers, distributed workers, message
buses, audit stores, monitoring systems, notification providers, external AI
CLIs, or Bytewax workers. Those are adapter responsibilities behind the APG
composition layer. The package does enforce the same contract and policy shape
that production adapters must honor.

## Verification

Focused package proof:

```bash
./.venv/bin/python -m py_compile capabilities/common/schd/__init__.py capabilities/common/schd/models.py capabilities/common/schd/scheduling_runtime.py capabilities/common/schd/service.py capabilities/common/schd/api.py capabilities/common/schd/views.py capabilities/common/schd/capability_contract.py capabilities/common/schd/app.py capabilities/common/schd/test_capability_contract.py capabilities/common/schd/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/schd/test_capability_contract.py capabilities/common/schd/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.schd import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/schd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/schd --json
```
