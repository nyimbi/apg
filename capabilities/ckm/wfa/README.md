# CKM Workflow Automation

`ckm_wfa` is the APG Collaboration and Knowledge Management workflow
automation capability. It lets generated applications compose workflow
definitions, active process instances, human and service tasks, approval
governance, exception handling, analytics metadata, Bytewax stream guardrails,
visual theme metadata, and AI-agent assistance.

The package is dependency-light. It defines the executable lifecycle, rule
engine, UI route metadata, theme metadata, Bytewax stream declaration, and
semantic evidence. Visual designers, persistent storage, external connectors,
live schedulers, durable audit sinks, and stream-worker deployments are adapter
responsibilities.

## What It Provides

- Workflow definition lifecycle with owners, versions, triggers, and variable
  schemas.
- Activation control that requires approval evidence.
- Tenant-scoped process instances with initiator and context.
- Human, approval, service, decision, notification, and subprocess tasks.
- Task completion evidence and approval governance.
- Exception ownership and escalation metadata.
- AI WFA-agent registration for Codex, Claude Code, OpenCode, Pi, and future
  runtimes behind the same contract.
- Bytewax stream guardrail for batch workflow mutation.
- UI routes and visual theme tokens for generated APG applications.

## Quick Use

```python
from capabilities.ckm.wfa import WfaLifecycleService

service = WfaLifecycleService("tenant-acme")

service.create_process(
    process_id="proc-close-001",
    name="Month-end close approvals",
    owner_id="user-controller",
    version="1.0.0",
    variable_schema={"amount": {"type": "number"}, "period": {"type": "string"}},
    trigger="manual",
)

service.activate_process(
    process_id="proc-close-001",
    approval_recorded=True,
    reviewer_id="user-cfo",
)

service.start_instance(
    instance_id="inst-close-001",
    process_id="proc-close-001",
    initiated_by="user-controller",
    context={"period": "2026-05"},
    correlation_key="close/2026-05",
)

task = service.create_task(
    task_id="task-review-001",
    instance_id="inst-close-001",
    name="Review accrual batch",
    assignee_id="user-cfo",
)

service.complete_task(
    task_id=task["id"],
    completed_by="user-cfo",
    completion_evidence={"journal_batch": "JB-2026-05-A"},
)
```

## AI Agent Registration

AI agents are first-class workflow contributors only after registration:

```python
agent = service.register_wfa_agent(
    name="Approval reviewer",
    runtime="codex",
    role="approval_reviewer",
    scope="review workflow approvals for independence and evidence",
    contribution_disclosed=True,
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles are `process_designer`, `approval_reviewer`, `exception_reviewer`,
`sla_reviewer`, and `optimization_reviewer`.

## Guardrails

The deterministic rules deny or require review when:

- tenant context is missing;
- workflow definition owner or version is missing;
- workflow activation lacks approval evidence;
- an instance starts from an inactive definition;
- an instance lacks an initiator;
- a human task lacks assignee or queue ownership;
- an SLA-tracked task lacks a due time;
- task completion lacks evidence;
- approval reviewer is the same person as the requester;
- an approval decision lacks a reason;
- a rejected approval lacks a reason;
- SLA breach handling lacks review;
- an exception lacks an owner;
- an AI WFA agent is unregistered, unsupported, unscoped, or undisclosed;
- lifecycle state changes lack audit evidence;
- batch workflow mutation does not use Bytewax.

## Bytewax Batch Mutation

Batch workflow mutation must use the Bytewax event stream:

```python
allowed = service.validate_batch_wfa_mutation("bytewax")
blocked = service.validate_batch_wfa_mutation("other-stream")

assert allowed["decision"] == "allow"
assert blocked["decision"] == "deny"
```

The contract declares topic `apg.ckm_wfa.lifecycle` and state for definitions,
instances, tasks, approvals, exceptions, WFA agents, and audit events.

## Composition

Generated APG applications should compose `ckm_wfa` through:

- capability ID: `ckm_wfa`;
- provided services: workflow definitions, workflow instances, task
  orchestration, approval governance, exception management, workflow analytics,
  and WFA agents;
- required services: `auth`, `conf`, `audl`, `ckm_not`, and `ckm_rtc`;
- API prefix: `/ckm-wfa/api/v1`;
- UI routes: dashboard, designer, definitions, instances, tasks, approvals,
  exceptions, agents, rules, analytics, audit, and settings;
- theme: `ckm_wfa_workflow_ops`;
- stream processor: `bytewax`.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/ckm/wfa/__init__.py capabilities/ckm/wfa/capability_contract.py capabilities/ckm/wfa/lifecycle.py capabilities/ckm/wfa/app.py capabilities/ckm/wfa/test_capability_contract.py
./.venv/bin/pytest -q capabilities/ckm/wfa/test_capability_contract.py
./.venv/bin/python -c "import importlib; pkg = importlib.import_module('capabilities.ckm.wfa'); service = pkg.WfaLifecycleService('tenant-proof'); print(service.dashboard_summary())"
./.venv/bin/apg capabilities implementation-audit --root capabilities/ckm/wfa --json
./.venv/bin/apg capabilities publish-plan capabilities/ckm/wfa --json
```
