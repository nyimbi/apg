# Workflow Low-Code (wflo)

`wflo` provides APG's common capability for governed workflow and process automation. It composes workflow definition, versioning, publication approval, trigger policy, retry policy, task routing, approval gates, execution state, event streams, compensation, provider-neutral AI workflow agents, visual designer serialization, UI route metadata, visual theming, and Bytewax lifecycle guardrails.

## What It Provides

- Tenant-scoped workflow definitions with owner, version, steps, triggers, retry policy, compensation plan, runtime estimate, review state, and publication state.
- Step modeling for human, automation, approval, AI, and event steps with policy references and compensation metadata.
- Execution lifecycle with published-definition enforcement, correlation IDs, idempotent starts, Bytewax event stream policy, event history, cancellation, failure, completion, and compensation state.
- Task lifecycle with assignment, claim, bulk creation, completion, escalation reason, due date, and event emission.
- Approval lifecycle with approver, reason, decision evidence, delegation, decision state, and execution status updates.
- Visual designer serialization via `serialize_designer_state` — emits a `{nodes, edges, metadata}` graph compatible with React Flow and similar canvas renderers.
- First-class workflow agents with provider-neutral runtime (`codex`, `claude_code`, `opencode`, `pi`), governance role, owner, purpose, scope, disclosure, privileged-role review, and audit events.
- Bytewax lifecycle batch validation for definition, publication, execution, task, approval, compensation, workflow-agent, and audit mutations.
- Durable review evidence for review-required workflow definitions, privileged workflow agents, denied lifecycle batches, approval decisions, and audit events.
- Deterministic rule decisions for workflow authoring, runtime, approvals, tasks, compensation, agents, tenant isolation, and batch mutation policy.
- Full async interface (`async_*` variants) for ASGI hosts, background schedulers, and async test suites.
- Dependency-light API helpers, UI view models, package manifest, semantic model, and release evidence.
- Process simulation, bottleneck detection, SLA enforcement, parallel/inclusive gateways, boundary events, and process analytics.

## Runtime Shape

The runtime is `service.WfloService`. It is deterministic and in-memory so generated applications can exercise workflow behavior without live event buses, schedulers, distributed executors, notification systems, AI providers, script runners, or durable workflow databases.

### Synchronous Methods

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
- `bpmn_import(...)`
- `process_simulate(...)`
- `bottleneck_detect(...)`
- `sla_enforce(...)`
- `compensation_trigger(...)`
- `parallel_gateway(...)`
- `inclusive_gateway(...)`
- `boundary_event(...)`
- `escalation_handle(...)`
- `process_analytics(...)`
- `serialize_designer_state(...)`
- `list_pending_reviews(...)`
- `dashboard_summary(...)`

### Async Methods

All sync methods that involve mutations or I/O-bound aggregation have async counterparts:

- `async_create_workflow_definition(...)`
- `async_start_execution(...)` — enforces idempotency on duplicate `correlation_id`
- `async_complete_task(...)`
- `async_record_approval(...)`
- `async_process_analytics(...)`
- `async_sla_enforce(...)`
- `async_bpmn_import(...)`
- `async_bulk_create_tasks(...)` — creates N tasks in one await
- `async_dashboard_summary(...)`
- `async_cancel_execution(...)`
- `async_process_simulate(...)`
- `async_serialize_designer_state(...)`

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

## Review Evidence

WFLO persists review and denial evidence instead of leaving it only in transient rule-engine responses. Long-running definitions and privileged workflow agents carry `decision`, `matched_rules`, `review_reasons`, and `audit_evidence`. Denied lifecycle batches are stored as `denied` before `PermissionError` is raised so generated applications can inspect rejected Bytewax lifecycle mutations.

Use `list_pending_reviews()` or the dashboard, definition library, agent panel, lifecycle monitor, and analytics view models to compose workflow approval queues without replaying rules.

## Workflow Agents

Workflow agents are first-class APG composition citizens. WFLO records their stable ID, name, runtime, role, owner, purpose, workflow scope, contribution disclosure, human-approval treatment, and review status. WFLO does not invoke external Codex, Claude Code, OpenCode, Pi, or other agent clients directly; those integrations belong behind the AICR provider-neutral adapter contract named in the capability contract.

Privileged roles (`step_runner`, `approval_advisor`, `compensation_planner`, `integration_coordinator`, `lifecycle_batch_reviewer`, `process_steward`) require human approval evidence to become active immediately. Without that evidence, WFLO keeps the agent in `pending_review` rather than silently granting authority.

## Bytewax Lifecycle Batches

WFLO validates lifecycle batches through `validate_lifecycle_batch(...)`. Accepted batches must use the Bytewax stream, include at least one mutation, and name a configured operation such as `definition_batch`, `execution_batch`, `task_batch`, `approval_batch`, `compensation_batch`, or `workflow_agent_batch`.

## UI Surfaces

Route contracts are exposed for: dashboard, designer, definitions, executions, tasks, approvals, agents, lifecycle, audit, analytics, settings.

`views.py` provides dependency-light models for these screens.

---

## World-Class Enhancements (v2.0)

1. **Async-First Service Layer** — All mutations run through `asyncio.Lock`-guarded async methods; eliminates thread contention in FastAPI/ASGI hosts.
2. **Persistent Storage Adapter Pattern** — `WfloRepository` protocol (`async get/put/query/delete`) with `PostgresWfloRepository` via SQLAlchemy async core; service becomes stateless and horizontally scalable.
3. **Workflow Version Diffing and Migration** — On `create_workflow_definition` for an existing name at a higher version, diffs steps, detects removed/reordered steps, and produces a migration plan; guards in-flight executions from invalidation.
4. **Real BPMN 2.0 Parser** — Replaces the regex stub with a full `xml.etree.ElementTree` namespace-aware parser that round-trips sequence flows, gateways, pools, lanes, boundary events, and data objects.
5. **Visual Designer State Serialization** — `serialize_designer_state` / `async_serialize_designer_state` emit a canonical `{nodes, edges, metadata}` JSON suitable for React Flow; makes the capability genuinely low-code.
6. **Conditional Expression Evaluator** — Inclusive gateway conditions evaluated at runtime via a safe expression evaluator (e.g. `simpleeval`); branching decisions are deterministic and unit-testable.
7. **SLA Deadline Tracking with Real Timestamps** — `sla_enforce` computes elapsed wall-clock minutes against per-step `sla_minutes`; returns `warning` / `critical` severity levels based on configurable threshold ratios.
8. **Process Mining Integration** — `process_mine` replays `WorkflowAuditEventRecord` streams and produces a Petri-net event log in XES format for PM4Py conformance checking and variant discovery.
9. **Multi-Tenant Isolation at Storage Layer** — All storage keys namespaced with tenant ID; cross-tenant lookup is structurally impossible at both in-memory dict and Postgres adapter layers.
10. **Webhook / Notification Dispatch** — `EventDispatcher` protocol (`async dispatch(event)`) with HTTP webhook, Redis Streams/Bytewax, and no-op implementations wired into `emit_event`.
11. **Parallel Gateway Join Synchronization** — `join_policy` field (`all`, `any`, `n_of_m`) on gateway records; `complete_execution` enforces join semantics by counting completed tasks per `parallel_group`.
12. **Bulk Execution Scheduling** — `schedule_bulk_executions` accepts `[{definition_id, correlation_id, payload, scheduled_at}]`, validates in a single pass, persists as `ScheduledExecutionRecord`, returns a batch receipt.
13. **Execution Replay and Idempotency** — Duplicate `correlation_id` on `async_start_execution` returns the existing record and emits a `duplicate_start_attempted` audit event; critical for at-least-once delivery.
14. **Role-Based Access Control (RBAC)** — `WfloAccessPolicy` protocol (`can(actor, operation, resource) -> bool`) checked before each mutation; ships with a role-map default implementation backed by the capability contract.
15. **Structured Error Catalog** — `WfloError` hierarchy (`WfloPermissionError`, `WfloNotFoundError`, `WfloValidationError`, `WfloPolicyError`) with machine-readable `code`, `detail`, and `context` fields; API layers map directly to HTTP status codes.

---

## New Methods

### `async_start_execution` — idempotent execution start

```python
import asyncio
from capabilities.common.wflo.service import WfloService

service = WfloService()

async def main():
    # First call creates the execution
    ex1 = await service.async_start_execution("tenant-1", definition["id"], "order-42", "user-1")
    # Second call with same correlation_id returns the existing record — no duplicate
    ex2 = await service.async_start_execution("tenant-1", definition["id"], "order-42", "user-1")
    assert ex1["id"] == ex2["id"]  # idempotent

asyncio.run(main())
```

### `async_bulk_create_tasks` — batch task creation

```python
tasks = await service.async_bulk_create_tasks(
    tenant_id="tenant-1",
    execution_id=execution["id"],
    task_specs=[
        {"step_id": steps[0]["id"], "title": "Review contract", "assignee_ref": "legal-team"},
        {"step_id": steps[1]["id"], "title": "Approve budget", "assignee_ref": "finance-lead", "due_at": "2026-06-15T09:00:00Z"},
        {"step_id": steps[2]["id"], "title": "Sign off", "assignee_ref": "cfo"},
    ],
)
# Returns list of task dicts; all created in one await
```

### `serialize_designer_state` — React Flow–compatible graph

```python
graph = service.serialize_designer_state("tenant-1", definition["id"])
# graph == {
#   "nodes": [{"id": "__start__", "type": "start", ...}, {"id": "step-id", "type": "human", ...}, ...],
#   "edges": [{"id": "e___start____step-id", "source": "__start__", "target": "step-id"}, ...],
#   "metadata": {"step_count": 3, "trigger_type": "external", "status": "published", ...},
# }
# Pass graph["nodes"] and graph["edges"] directly to React Flow's <ReactFlow> component
```

### `async_sla_enforce` — background SLA monitor

```python
# Safe to call from APScheduler or Celery beat; returns breach details
report = await service.async_sla_enforce("tenant-1", execution["id"])
if report["sla_status"] == "breach":
    for task in report["overdue_tasks"]:
        print(f"Breach: {task['step']} — {task['breach']}, SLA={task['sla_minutes']}m")
```

### `async_process_simulate` — throughput estimation

```python
sim = await service.async_process_simulate(
    tenant_id="tenant-1",
    definition_id=definition["id"],
    simulation_runs=500,
)
print(f"Estimated throughput: {sim['estimated_throughput_per_day']}/day")
print(f"SLA pass rate: {sim['sla_pass_rate']:.1%}")
```

---

## Quick Start

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

```bash
./.venv/bin/python -m py_compile capabilities/common/wflo/__init__.py capabilities/common/wflo/models.py capabilities/common/wflo/workflow_runtime.py capabilities/common/wflo/service.py capabilities/common/wflo/api.py capabilities/common/wflo/views.py capabilities/common/wflo/capability_contract.py capabilities/common/wflo/app.py capabilities/common/wflo/test_capability_contract.py capabilities/common/wflo/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/wflo/test_capability_contract.py capabilities/common/wflo/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/wflo --json
./.venv/bin/apg capabilities publish-plan capabilities/common/wflo --json
```
