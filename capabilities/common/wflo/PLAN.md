# Workflow Orchestration Capability Plan

## Implementation Packet

Build one coherent lifecycle and guardrail packet that makes `wflo` usable by generated applications without external services:

1. Document the capability boundary, runtime shape, rules, UI, adapters, and verification gates.
2. Expand the executable contract with definition, step, execution, task, approval, first-class workflow-agent, governance, observability, adapter, UI, theme, and Bytewax streaming surfaces.
3. Extend runtime models for step policies, execution state-change reasons, compensation state, task claim/escalation data, approval decision evidence, workflow agents, and lifecycle batch records.
4. Enforce workflow lifecycle guardrails in `WfloService`: retry policy, trigger policy, AI/automation/event step policies, published execution, correlation ID, Bytewax events, task claims, approval evidence, cancellation/failure reason, compensation plan, provider-neutral agent registration, privileged-agent review, and lifecycle batch validation.
5. Preserve durable review evidence on review-required definitions, privileged workflow agents, lifecycle batches, approval decisions, and audit events.
6. Expose dependency-light API helpers and view models for the new lifecycle surfaces, including pending-review queues, the workflow-agent roster, and lifecycle batch monitor.
7. Refresh package semantic evidence from the live contract.
8. Run focused verification only, preserving battery.

## Review Checklist

- Every state-changing operation evaluates tenant context.
- Runtime checks match contract rule names and reasons.
- Workflow definitions cannot publish or retire without approval.
- Execution cannot start from unpublished definitions or without correlation ID.
- Task completion cannot bypass claim policy.
- Approval decisions cannot bypass evidence, and delegation cannot omit a delegate.
- Cancellation, failure, escalation, and compensation cannot omit required reason or plan evidence.
- AI workflow agents cannot be registered without stable ID, readable name, supported runtime, supported role, scope, owner, purpose, and disclosure.
- Privileged workflow-agent roles require human approval evidence or remain in `pending_review`.
- Review-required definitions, pending-review agents, and denied lifecycle batches expose matched rules, review reasons, and audit evidence.
- Lifecycle batches cannot be accepted without Bytewax routing, supported lifecycle operation, and at least one mutation.
- UI routes and view models expose all user-visible lifecycle surfaces.
- Documentation explains production adapter boundaries instead of implying external services run locally.
- No disallowed message-bus dependency or stale generated-baseline marker is introduced.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/wflo/__init__.py capabilities/common/wflo/models.py capabilities/common/wflo/workflow_runtime.py capabilities/common/wflo/service.py capabilities/common/wflo/api.py capabilities/common/wflo/views.py capabilities/common/wflo/capability_contract.py capabilities/common/wflo/app.py capabilities/common/wflo/test_capability_contract.py capabilities/common/wflo/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/wflo/test_capability_contract.py capabilities/common/wflo/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.wflo import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities inspect wflo --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/wflo --json
./.venv/bin/apg capabilities publish-plan capabilities/common/wflo --json
git diff --check -- capabilities/common/wflo docs/progress_log.md
```

## Deliberately Out Of Scope

- Live event bus and distributed executor operation.
- Scheduler and notification delivery.
- Script runtime and external AI-agent CLI execution.
- Durable workflow database and migration work.
- Browser-rendered workflow studio behavior.
- Live Bytewax worker execution.
- Live Codex, Claude Code, OpenCode, Pi, or other external AI-agent clients.
