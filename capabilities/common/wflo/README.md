# Workflow Orchestration Capability

`wflo` provides APG's common capability for governed workflow and process automation. It composes workflow definition, versioning, publication approval, trigger policy, retry policy, task routing, approval gates, execution state, event streams, compensation, first-class provider-neutral AI workflow agents, UI route metadata, visual theming, and Bytewax lifecycle guardrails.

## What It Provides

- Tenant-scoped workflow definitions with owner, version, steps, triggers, retry policy, compensation plan, runtime estimate, review state, and publication state.
- Step modeling for human, automation, approval, AI, and event steps with policy references and compensation metadata.
- Execution lifecycle with published-definition enforcement, correlation IDs, Bytewax event stream policy, event history, cancellation, failure, completion, and compensation state.
- Task lifecycle with assignment, claim, completion, escalation reason, due date, and event emission.
- Approval lifecycle with approver, reason, decision evidence, delegation, decision state, and execution status updates.
- First-class workflow agents with provider-neutral runtime (`codex`, `claude_code`, `opencode`, `pi`), governance role, owner, purpose, scope, disclosure, privileged-role review, and audit events.
- Bytewax lifecycle batch validation for definition, publication, execution, task, approval, compensation, workflow-agent, and audit mutations.
- Deterministic rule decisions for workflow authoring, runtime, approvals, tasks, compensation, agents, tenant isolation, and batch mutation policy.
- Dependency-light API helpers, UI view models, package manifest, semantic model, and release evidence.

## Runtime Shape

The generated runtime is `service.WfloService`. It is deterministic and in-memory so generated applications can exercise workflow behavior without live event buses, schedulers, distributed executors, notification systems, AI providers, script runners, or durable workflow databases.

Primary methods:

- `create_workflow_definition(...)`
- `publish_workflow(...)`
- `retire_workflow(...)`
- `start_execution(...)`
- `create_task(...)`
- `claim_task(...)`
- `complete_task(...)`
- `escalate_task(...)`
- `request_approval(...)`
- `record_approval(...)`
- `complete_execution(...)`
- `cancel_execution(...)`
- `fail_execution(...)`
- `run_compensation(...)`
- `register_workflow_agent(...)`
- `validate_batch_mutation(...)`
- `validate_lifecycle_batch(...)`
- `dashboard_summary(...)`

## Configuration And Rules

`capability_contract.py` is the source of truth for:

- configuration defaults
- configuration schema
- deterministic rules
- UI route contracts
- theme tokens
- APG adapter map
- Bytewax streaming contract

The rule engine returns `allow`, `require_review`, or `deny` decisions with matched rules and required actions. Runtime methods enforce the same guardrails used by the contract.

## Workflow Agents

Workflow agents are first-class APG composition citizens. WFLO records their stable ID, name, runtime, role, owner, purpose, workflow scope, contribution disclosure, human-approval treatment, and review status. WFLO does not invoke external Codex, Claude Code, OpenCode, Pi, or other agent clients directly; those integrations belong behind the AICR provider-neutral adapter contract named in the capability contract.

Privileged roles such as `step_runner`, `approval_advisor`, `compensation_planner`, `integration_coordinator`, `lifecycle_batch_reviewer`, and `process_steward` require human approval evidence to become active immediately. Without that evidence, WFLO keeps the agent in `pending_review` rather than silently granting authority.

## Bytewax Lifecycle Batches

WFLO validates lifecycle batches through `validate_lifecycle_batch(...)`. Accepted batches must use the Bytewax stream, include at least one mutation, and name a configured operation such as `definition_batch`, `execution_batch`, `task_batch`, `approval_batch`, `compensation_batch`, or `workflow_agent_batch`. The packet deliberately exposes Bytewax metadata and guardrails without starting a live worker in dependency-light generated applications.

## UI Surfaces

The package exposes route contracts for:

- dashboard
- designer
- definitions
- executions
- tasks
- approvals
- agents
- lifecycle
- audit
- analytics
- settings

`views.py` provides dependency-light models for these screens.

## How To Use

```python
from capabilities.common.wflo.service import WfloService

service = WfloService()
definition = service.create_workflow_definition(
    tenant_id="tenant-1",
    name="Purchase Approval",
    owner_ref="process-owner",
    steps=[
        {"name": "review_request", "step_type": "human", "assignee_ref": "manager"},
        {"name": "approve_request", "step_type": "approval", "requires_approval": True},
    ],
    trigger_type="external",
    trigger_policy_ref="trigger-policy://purchase",
    retry_policy_ref="retry://default",
    compensation_ref="compensation://purchase",
)
published = service.publish_workflow("tenant-1", definition["id"], "approval://publish/1", "workflow-admin")
execution = service.start_execution("tenant-1", published["id"], "purchase-123", "requester-1")
task = service.create_task("tenant-1", execution["id"], published["steps"][0]["id"], "Review purchase", "manager")
service.claim_task("tenant-1", task["id"], "manager")
service.complete_task("tenant-1", task["id"], "manager")
approval = service.request_approval("tenant-1", execution["id"], "purchase-123", "approver-1", "High value purchase")
service.record_approval("tenant-1", approval["id"], "approved", "approver-1", "evidence://approval/1")
service.register_workflow_agent(
    "agent-1",
    "tenant-1",
    "Runtime observer",
    "codex",
    "runtime_observer",
    execution["id"],
    "workflow-admin",
    True,
    owner_ref="workflow-admin",
    purpose="Observe workflow execution state and flag blocked transitions.",
)
service.validate_lifecycle_batch("tenant-1", "bytewax", 1, "workflow_agent_batch")
completed = service.complete_execution("tenant-1", execution["id"], "workflow-admin")
```

Use `register_capability()` to expose the full APG registration payload to the composition engine.

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile capabilities/common/wflo/__init__.py capabilities/common/wflo/models.py capabilities/common/wflo/workflow_runtime.py capabilities/common/wflo/service.py capabilities/common/wflo/api.py capabilities/common/wflo/views.py capabilities/common/wflo/capability_contract.py capabilities/common/wflo/app.py capabilities/common/wflo/test_capability_contract.py capabilities/common/wflo/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/wflo/test_capability_contract.py capabilities/common/wflo/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/wflo --json
./.venv/bin/apg capabilities publish-plan capabilities/common/wflo --json
```
