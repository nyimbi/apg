# Workflow Low-Code (wflo) — User Guide

© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke

---

## Overview

The `wflo` capability provides APG's governed workflow and process automation engine. It supports visual workflow design, form-based process building, human task routing, approval gates, AI/automation step integration, SLA enforcement, compensation flows, and real-time event emission — all under a deterministic policy engine.

---

## Core Concepts

| Concept | Description |
|---|---|
| **Workflow Definition** | Template describing the process: steps, trigger, retry, compensation, version. |
| **Workflow Execution** | A running instance of a definition, scoped to a correlation ID. |
| **Step** | A discrete unit of work: `human`, `automation`, `approval`, `ai`, or `event`. |
| **Task** | A human step instantiated at runtime, assigned to a person or role. |
| **Approval** | A formal decision gate attached to an execution. |
| **Workflow Agent** | An AI or automation agent registered to participate in workflow steps. |
| **Lifecycle Batch** | A Bytewax-validated batch of workflow mutations. |
| **Audit Event** | An immutable log entry recording every material state change. |

---

## Installation

```bash
pip install apg-common-wflo
# or via uv in the APG monorepo:
uv pip install -e capabilities/common/wflo
```

---

## Quick Start

```python
from capabilities.common.wflo.service import WfloService

svc = WfloService()

# 1. Define a workflow
defn = svc.create_workflow_definition(
    tenant_id="tenant-1",
    name="Invoice Approval",
    owner_ref="finance-team",
    steps=[
        {"name": "review_invoice", "step_type": "human", "assignee_ref": "accountant", "sla_minutes": 480},
        {"name": "approve_payment", "step_type": "approval", "requires_approval": True, "sla_minutes": 240},
    ],
    trigger_type="external",
    trigger_policy_ref="trigger://invoice",
    retry_policy_ref="retry://default",
    compensation_ref="compensation://invoice",
    expected_runtime_minutes=720,
    runtime_review_recorded=True,
)

# 2. Publish it
published = svc.publish_workflow(defn["id"], defn["id"], "approval://publish/1", "workflow-admin")

# 3. Start an execution
execution = svc.start_execution("tenant-1", defn["id"], "invoice-2025-001", "requester")

# 4. Work the tasks
task = svc.create_task("tenant-1", execution["id"], defn["steps"][0]["id"], "Review invoice #2025-001", "accountant")
svc.claim_task("tenant-1", task["id"], "accountant")
svc.complete_task("tenant-1", task["id"], "accountant")

# 5. Gate with approval
approval = svc.request_approval("tenant-1", execution["id"], "invoice-2025-001", "cfo", "High value invoice")
svc.record_approval("tenant-1", approval["id"], "approved", "cfo", "evidence://cfo/sign/1")

# 6. Complete
svc.complete_execution("tenant-1", execution["id"], "workflow-admin")
```

---

## Async Usage

All core methods have `async_*` counterparts safe to await in FastAPI, Starlette, or any ASGI host.

```python
import asyncio
from capabilities.common.wflo.service import WfloService

svc = WfloService()

async def run():
    defn = await svc.async_create_workflow_definition(
        tenant_id="tenant-1",
        name="Onboarding",
        owner_ref="hr-team",
        steps=[{"name": "send_welcome", "step_type": "human"}],
        retry_policy_ref="retry://default",
        runtime_review_recorded=True,
    )
    execution = await svc.async_start_execution(
        "tenant-1", defn["id"], "onboard-emp-42", "hr-admin"
    )
    print(execution["id"])

asyncio.run(run())
```

### Idempotent Execution Starts

`async_start_execution` detects duplicate `correlation_id` values and returns the existing execution record instead of creating a duplicate:

```python
exec1 = await svc.async_start_execution("tenant-1", defn["id"], "corr-abc", "actor")
exec2 = await svc.async_start_execution("tenant-1", defn["id"], "corr-abc", "actor")
assert exec1["id"] == exec2["id"]  # same execution returned
```

### Bulk Task Creation

Create multiple tasks in one call:

```python
tasks = await svc.async_bulk_create_tasks(
    tenant_id="tenant-1",
    execution_id=execution["id"],
    task_specs=[
        {"step_id": step1_id, "title": "Legal review", "assignee_ref": "legal"},
        {"step_id": step2_id, "title": "Finance sign-off", "assignee_ref": "cfo", "due_at": "2026-06-15T00:00:00Z"},
    ],
)
```

---

## Visual Designer Serialization

Serialize any published workflow into a canvas-compatible node/edge graph for React Flow or similar renderers:

```python
graph = svc.serialize_designer_state("tenant-1", defn["id"])
# {
#   "nodes": [...],   # one node per step + start/end sentinels
#   "edges": [...],   # sequential + parallel group edges
#   "metadata": {...}
# }
```

The async variant:

```python
graph = await svc.async_serialize_designer_state("tenant-1", defn["id"])
```

---

## BPMN Import

Import a BPMN 2.0 XML document and create a workflow definition from it:

```python
with open("purchase_order.bpmn") as f:
    bpmn_xml = f.read()

result = svc.bpmn_import(
    tenant_id="tenant-1",
    bpmn_xml=bpmn_xml,
    owner_ref="process-team",
    actor="workflow-admin",
)
print(result["bpmn_task_count"], "tasks imported")
```

Async:

```python
result = await svc.async_bpmn_import("tenant-1", bpmn_xml, "process-team")
```

---

## Parallel Gateways

Fork a workflow into concurrent branches (AND-split):

```python
svc.parallel_gateway(
    tenant_id="tenant-1",
    definition_id=defn["id"],
    gateway_name="parallel_review",
    branch_step_names=["legal", "finance", "compliance"],
    owner_ref="process-owner",
)
```

---

## Inclusive Gateways

Create condition-based OR-splits:

```python
svc.inclusive_gateway(
    tenant_id="tenant-1",
    definition_id=defn["id"],
    gateway_name="value_routing",
    condition_steps=[
        {"name": "high_value", "condition": "amount > 10000", "event_policy_ref": "policy://high-value"},
        {"name": "standard",   "condition": "amount <= 10000"},
    ],
)
```

---

## SLA Enforcement

Check for breached tasks on any running execution:

```python
report = svc.sla_enforce("tenant-1", execution["id"])
if report["sla_status"] == "breach":
    for task_info in report["overdue_tasks"]:
        print(f"Task {task_info['task_id']} breached on step {task_info['step']}")
```

Async:

```python
report = await svc.async_sla_enforce("tenant-1", execution["id"])
```

---

## Process Simulation

Estimate throughput, average cycle time, and SLA pass rate before going live:

```python
sim = svc.process_simulate("tenant-1", defn["id"], simulation_runs=500)
print(sim["estimated_throughput_per_day"], "executions/day at this SLA config")
```

---

## Bottleneck Detection

Identify the top-3 slowest steps based on configured SLA windows:

```python
report = svc.bottleneck_detect("tenant-1", defn["id"])
for step in report["bottleneck_steps"]:
    print(step["name"], "—", step["sla_minutes"], "min SLA")
```

---

## Compensation Flows

Trigger rollback on a failed execution:

```python
result = svc.compensation_trigger("tenant-1", execution["id"], reason="payment gateway timeout")
print(result["compensation"]["compensation_status"])  # "completed"
```

---

## Boundary Events

Attach a timer, error, signal, or message event to a running step:

```python
svc.boundary_event("tenant-1", execution["id"], step_id, "timer", {"timeout_minutes": 60})
```

---

## Process Analytics

Aggregate metrics across all executions for a tenant:

```python
analytics = svc.process_analytics("tenant-1")
print(analytics["completion_rate"])  # e.g. 0.92
```

Async:

```python
analytics = await svc.async_process_analytics("tenant-1")
```

---

## Dashboard Summary

Full operational snapshot:

```python
summary = svc.dashboard_summary("tenant-1")
# Keys: definition_count, running_execution_count, open_task_count,
#       pending_approval_count, agent_count, event_count, ...
```

Async:

```python
summary = await svc.async_dashboard_summary("tenant-1")
```

---

## Workflow Agents

Register a provider-neutral AI agent into the workflow:

```python
svc.register_workflow_agent(
    agent_id="agent-llm-1",
    tenant_id="tenant-1",
    name="Invoice Classifier",
    runtime="claude_code",
    role="runtime_observer",
    scope_ref=execution["id"],
    registered_by="workflow-admin",
    contribution_disclosed=True,
    owner_ref="ai-team",
    purpose="Classify invoice priority and flag anomalies.",
)
```

Privileged roles (`step_runner`, `approval_advisor`, `compensation_planner`, etc.) require `human_approval_required=True` to become immediately active. Without it, the agent lands in `pending_review`.

---

## Lifecycle Batch Validation

Validate Bytewax lifecycle mutations before committing:

```python
result = svc.validate_lifecycle_batch(
    tenant_id="tenant-1",
    event_stream="bytewax",
    mutation_count=12,
    operation="execution_batch",
)
print(result["status"])  # "accepted" | "review_required" | "denied"
```

---

## Audit Trail

Every material state change is recorded as a `WorkflowAuditEventRecord`:

```python
audit = svc.list_audit_events("tenant-1")
for event in audit:
    print(event["event_type"], event["actor"], event["severity"])
```

---

## Policy Engine

All mutations pass through `capability_contract.evaluate_capability_rules`. Decisions are `allow`, `require_review`, or `deny`. Review-required records persist `decision`, `matched_rules`, `review_reasons`, and `audit_evidence` so approval queues can be built without replaying rules:

```python
pending = svc.list_pending_reviews("tenant-1")
```

---

## Testing

```bash
# Compile check
./.venv/bin/python -m py_compile capabilities/common/wflo/service.py

# Unit + contract tests
uv run pytest -vxs capabilities/common/wflo/tests/

# APG audit
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/wflo --json
```

---

## Configuration Reference

Key defaults from `capability_contract.DEFAULT_CONFIGURATION`:

| Parameter | Default | Notes |
|---|---|---|
| `max_steps_per_workflow` | 50 | Triggers review above this limit |
| `max_executions_per_definition` | 1000 | Throttle per definition |
| `default_sla_minutes` | 1440 | 24 hours |
| `event_stream` | `bytewax` | Only supported stream |
| `required_operations` | see contract | Allowed lifecycle batch ops |

---

## Troubleshooting

**`workflow_policy_blocked`** — A rule denied the operation. Check `result["matched_rules"]` for the specific rule name.

**`workflow_definition_not_found`** — The definition ID or name does not exist under the given tenant. Confirm `tenant_id` matches.

**`unsupported_workflow_step_type`** — Valid step types: `human`, `automation`, `approval`, `ai`, `event`.

**`task_already_completed`** — `claim_task` cannot claim a completed task. Check task status before claiming.
