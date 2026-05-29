# APG Capability Specification: SCHD - Scheduling and Job Orchestration

`schd` is the APG Scheduling and Job Orchestration capability. It provides
tenant-aware calendar policies, worker pools, job definitions, schedules,
runtime job attempts, retries, dead-letter posture, audit events, deterministic
rules, API helpers, route metadata, and scheduler operations theming.

## Executable Runtime

The package is implemented as a dependency-light Python runtime:

| Surface | File | Responsibility |
| --- | --- | --- |
| Contract | `capability_contract.py` | configuration, deterministic scheduler rules, UI routes, and theme tokens |
| Runtime helpers | `scheduling_runtime.py` | stable IDs, trigger/job/worker/retry normalization, run state, next-run hints, backoff |
| Models | `models.py` | calendar policies, worker pools, jobs, schedules, runs, and audit events |
| Service | `service.py` | tenant-scoped lifecycle methods and policy enforcement |
| API helpers | `api.py` | callable package API surface for composition and generated apps |
| View models | `views.py` | dashboard, schedule console, jobs, runs, workers, calendars, analytics, settings |
| Package entrypoint | `app.py` | publishable semantic model, component manifest, and self-test |

The runtime intentionally keeps external schedulers, queue brokers, worker
daemons, monitoring exporters, notification systems, CI/CD launchers, cache
stores, and compensation engines behind future adapters. The current local
contract must remain deterministic and runnable without live infrastructure.

## Domain Model

`SchdService` manages:

- calendar policies with timezone, business-day, blackout, and holiday metadata
- worker pools with queue, concurrency, health, capacity, state, and autoscaling metadata
- job definitions with command, owner, criticality, expected runtime, monitoring, approvals, retry policy, and tags
- schedules binding jobs to calendars, worker pools, triggers, owners, and next-run hints
- job runs with requested actor, status, attempt, processed records, errors, exit code, retry delay, logs, and timestamps
- audit events for calendar, worker, job, schedule, run, and disable operations

The compatibility `create_record` and `list_records` methods produce and list
schedules so existing package tooling can keep treating SCHD as a composable
APG package while richer scheduler APIs are used by new code.

## Rule Engine

SCHD uses deterministic rule evaluation from `capability_contract.py`.

| Rule | Enforced by |
| --- | --- |
| `tenant_context_required` | all service methods that create or mutate tenant-scoped objects |
| `schedule_requires_owner` | calendar policy and schedule creation |
| `timezone_required` | calendar policy and schedule creation |
| `critical_job_requires_monitoring` | critical job definition |
| `external_job_requires_approval` | external job definition |
| `long_running_job_requires_review` | long-running job definition above configured threshold |

Manual schedules also require a manual reason because SCHD governance declares
manual run reasons as required.

## UI And Theme Contract

The package publishes APG route metadata for:

- `/schd/dashboard`
- `/schd/schedules`
- `/schd/jobs`
- `/schd/runs`
- `/schd/workers`
- `/schd/calendars`
- `/schd/analytics`
- `/schd/settings`

The default theme is `schd_scheduler_ops`. Components include schedule
calendar, job run table, worker pool, and retry panel tokens. View models expose
plain dictionaries so generated Python apps, APG Studio, and future UI
adapters can compose the scheduler without framework-specific imports.

## Adapter Boundaries

Future live integrations should attach behind explicit adapters:

- workflow engine adapter for `wflo`
- queue/event bus adapter for `mqeb`
- monitoring exporter adapter for `moni`
- audit sink adapter for `audl`
- notification adapter for `ntfy`
- cache adapter for `cach`
- compensation adapter for `comp`
- worker daemon or CI/CD launcher adapters

Do not make package import, tests, publish-plan, or implementation audit depend
on those live providers.

## Focused Verification

Use focused checks while developing SCHD:

```bash
rg -n "<generated-baseline marker alternation>" capabilities/common/schd
./.venv/bin/python -m py_compile capabilities/common/schd/__init__.py capabilities/common/schd/models.py capabilities/common/schd/scheduling_runtime.py capabilities/common/schd/service.py capabilities/common/schd/api.py capabilities/common/schd/views.py capabilities/common/schd/capability_contract.py capabilities/common/schd/app.py capabilities/common/schd/test_capability_contract.py capabilities/common/schd/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/schd/test_capability_contract.py capabilities/common/schd/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/schd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/schd --json
```

The baseline-marker search should return no matches. The implementation audit
should classify `schd` as `domain_specific` and report no root warnings.
