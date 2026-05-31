# Scheduling and Job Orchestration Capability Plan

## Implementation Packet

Build one coherent lifecycle and guardrail packet that makes `schd` usable by
generated applications without external services:

1. Document the capability boundary, runtime shape, rules, UI, adapters, and
   verification gates.
2. Expand the executable contract with schedule, job, run, worker,
   first-class scheduler-agent, governance, observability, adapter, UI, theme,
   and Bytewax streaming surfaces.
3. Extend runtime models for worker health/state reasons, job retry policy,
   event-trigger policy, run event stream, cancellation, dead-letter, retry
   parent linkage, completion evidence, scheduler agents, and lifecycle batch
   records.
4. Enforce scheduling lifecycle guardrails in `SchdService`: tenant context,
   owner/timezone/calendar/worker requirements, worker readiness, job
   monitoring/approval/runtime review, manual reasons, run completion evidence,
   cancellation, retry, dead-letter, pause/resume, provider-neutral
   scheduler-agent registration, privileged-agent review, and lifecycle batch
   validation.
5. Expose dependency-light API helpers and view models for the new lifecycle
   surfaces, including the scheduler-agent roster and lifecycle batch monitor.
6. Refresh package semantic evidence from the live contract.
7. Run focused verification only, preserving battery.

## Review Checklist

- Every state-changing operation evaluates tenant context.
- Runtime checks match contract rule names and reasons.
- Schedules cannot run when disabled, paused, or assigned to offline workers.
- Critical jobs cannot bypass monitoring; external jobs cannot bypass approval.
- Run completion cannot bypass evidence or use negative counters.
- Retry cannot target a successful run or exceed attempt limits.
- Cancellation, pause, dead-letter, and drain transitions require reasons.
- AI scheduler agents cannot be registered without stable ID, readable name,
  supported runtime, supported role, scope, owner, purpose, and disclosure.
- Privileged scheduler-agent roles require human approval evidence or remain in
  `pending_review`.
- Lifecycle batches cannot be accepted without Bytewax routing, supported
  lifecycle operation, and at least one mutation.
- Bytewax is the only batch/runtime stream accepted by the contract.
- UI routes and view models expose all user-visible lifecycle surfaces.
- Documentation explains production adapter boundaries instead of implying
  external services run locally.
- No disallowed message-bus dependency or stale generated-baseline marker is
  introduced.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/schd/__init__.py capabilities/common/schd/models.py capabilities/common/schd/scheduling_runtime.py capabilities/common/schd/service.py capabilities/common/schd/api.py capabilities/common/schd/views.py capabilities/common/schd/capability_contract.py capabilities/common/schd/app.py capabilities/common/schd/test_capability_contract.py capabilities/common/schd/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/schd/test_capability_contract.py capabilities/common/schd/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.schd import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities inspect schd --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/schd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/schd --json
git diff --check -- capabilities/common/schd docs/progress_log.md
```

## Deliberately Out Of Scope

- Live scheduler loops and distributed worker execution.
- Durable queue/database migrations.
- Notification delivery and monitoring dashboards.
- External AI-agent CLI execution.
- Browser-rendered scheduler UI behavior.
- Live Bytewax worker execution.
