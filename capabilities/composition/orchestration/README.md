# Workflow Orchestration

## Overview

Workflow Orchestration provides the runtime engine for defining, validating, releasing, and executing multi-step business processes within the APG composition layer. It supports automated tasks, human task assignments, approval workflows, cross-capability integration tasks, transactional compensation, SLA escalation, event-triggered execution, distributed tracing, cost attribution, anomaly detection, and execution snapshotting — all coordinated through Bytewax.

The business value is a governed, auditable process layer that connects APG capabilities into end-to-end business workflows. Release governance (validation evidence, dry-run, rollback plan) prevents untested workflows from reaching production. Human task coordination and SLA escalation ensure that human-in-the-loop steps do not silently stall. Compensation steps on transactional workflows provide rollback safety when multi-step operations fail partway through.

## Capability ID

`composition_orchestration`  Version: see `package_manifest.json`

## Provides

| Service | Description |
|---------|-------------|
| workflow_definition_lifecycle | Define, version, and validate workflow graphs with start events and terminal states |
| workflow_graph_validation | Cycle detection, dependency validation, handler and assignee completeness checks |
| workflow_execution_lifecycle | Start, advance, pause, resume, and complete workflow instances via Bytewax |
| human_task_coordination | Assign tasks to users, roles, and groups with SLA deadlines and escalation rules |
| workflow_release_governance | Validation evidence, dry-run, rollback plan, and approval for production release |
| workflow_rule_enforcement | Deterministic rule engine enforcing all definition and execution guardrails |
| workflow_agents | AI agent workbench for workflow architecture, BPML review, and compliance review |
| execution_snapshotting | Point-in-time state capture and recovery for long-running workflow instances |
| sla_breach_detection | Background SLA timer with proactive warning and breach escalation |
| anomaly_detection | Rolling z-score analysis flagging statistical outliers in execution durations |
| cost_attribution | Per-execution and per-tenant cost-weight ledger for FinOps reporting |
| distributed_tracing | Waterfall trace reconstruction from audit events with W3C trace context propagation |
| execution_quotas | Per-tenant concurrency and throughput rate limits with backpressure hints |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authenticate workflow operators and task assignees |
| audl | Persist immutable workflow and task audit records |
| ntfy | Send task assignment, SLA breach, and escalation notifications |
| registry | Register this capability in the global catalog |
| composition_events | Coordinate execution lifecycle events via Bytewax |
| composition_config | Read environment-specific workflow configuration values |

## Quick Start

```python
from capabilities.composition.orchestration.service import WorkflowOrchestrationService

svc = WorkflowOrchestrationService()

# Define a workflow
defn = svc.define_workflow(
    tenant_id="acme",
    name="invoice-approval",
    version="1.0.0",
    owner="finance-team",
    tasks=[
        {"id": "review", "name": "Review Invoice", "task_type": "human",
         "assigned_role": "finance", "sla": {"hours": 24},
         "escalation": [{"level": 1, "notify": "finance-manager"}]},
        {"id": "approve", "name": "Approve Payment", "task_type": "approval",
         "approval_policy": {"required_approvals": 1}, "dependencies": ["review"]},
    ],
)

# Release to production
release = svc.release_workflow(
    tenant_id="acme",
    workflow_definition_id=defn["id"],
    release_notes="Initial release",
    dry_run_result={"passed": True},
    validation_evidence={"test_coverage": 0.9},
    rollback_plan="revert to previous version",
)

# Start an execution
execution = svc.start_execution(
    tenant_id="acme",
    release_id=release["id"],
    input_data={"invoice_id": "INV-001"},
    idempotency_key="inv-001-run-1",
)
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant scope for all operations |
| workflow_definitions.owner_required | bool | true | Workflows must have an accountable owner |
| workflow_definitions.cycle_detection_enabled | bool | true | Detect cycles in the task dependency graph at definition time |
| tasks.cross_capability_contract_required | bool | true | Cross-capability tasks must reference a capability contract |
| tasks.sla_escalation_required | bool | true | SLA-bound tasks require escalation rules |
| execution.bytewax_required | bool | true | All execution lifecycle events must route through Bytewax |
| execution.idempotency_required | bool | true | Execution starts require idempotency keys |
| execution.max_parallel_branches | int | 64 | Maximum concurrent task branches per execution |
| execution.compensation_required_for_transactions | bool | true | Transactional workflows require compensation steps |
| releases.dry_run_required | bool | true | Releases require a passing dry-run result |
| releases.rollback_plan_required | bool | true | Releases require a rollback plan |
| automation_agents.max_autonomous_scope | string | "recommend_validate_and_prepare" | Ceiling on autonomous agent actions |
| observability.event_stream | string | "apg.composition.orchestration.lifecycle" | Bytewax stream name |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /composition-orchestration/dashboard | GET | composition_orchestration:view | Overview |
| definitions | /composition-orchestration/definitions | GET/POST | composition_orchestration:manage_definitions | Definitions |
| designer | /composition-orchestration/designer | GET/POST | composition_orchestration:design | Design |
| executions | /composition-orchestration/executions | GET/POST | composition_orchestration:operate | Operations |
| tasks | /composition-orchestration/tasks | GET/POST | composition_orchestration:manage_tasks | Operations |
| releases | /composition-orchestration/releases | GET/POST | composition_orchestration:release | Release |
| rules | /composition-orchestration/rules | GET | composition_orchestration:govern | Governance |
| agents | /composition-orchestration/agents | GET/POST | composition_orchestration:admin | Automation |
| settings | /composition-orchestration/settings | GET/PUT | composition_orchestration:admin | Administration |
| stream | /composition-orchestration/api/v1/stream | GET | composition_orchestration:view | Streaming |

REST API prefix: `/composition-orchestration/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| workflow_write_requires_policy | write operation without policy attached | deny |
| workflow_requires_owner | define_workflow without owner | deny |
| workflow_requires_version | define_workflow without version | deny |
| workflow_requires_start_event | define_workflow without start event | deny |
| workflow_requires_task_graph | define_workflow without executable task | deny |
| workflow_requires_terminal_state | define_workflow without terminal state | deny |
| task_requires_handler | define automated/integration task without handler | deny |
| human_task_requires_assignee | define human task without assignee, group, or role | deny |
| approval_task_requires_policy | define approval task without approval policy | deny |
| cross_capability_task_requires_contract | define cross-capability task without contract reference | deny |
| execution_requires_bytewax_stream | start_execution not via bytewax | deny |
| execution_requires_idempotency_key | start_execution without idempotency key | deny |
| high_risk_execution_requires_review | start high-risk execution without review | require_review |
| release_requires_validation | release_workflow without validation evidence | deny |
| release_requires_dry_run | release_workflow without dry-run result | deny |
| release_requires_rollback_plan | release_workflow without rollback plan | deny |
| retry_policy_requires_limit | define task with retry policy but no retry limit | deny |
| sla_task_requires_escalation | define SLA-bound task without escalation rules | require_review |
| compensation_required_for_transaction | define transactional workflow without compensation steps | deny |
| batch_schedule_requires_bytewax | batch_schedule not via bytewax | deny |
| workflow_agent_runtime_supported | register_workflow_agent with unsupported runtime | deny |
| workflow_agent_role_supported | register_workflow_agent with unsupported role | deny |
| privileged_agent_workflow_action_requires_human_approval | agent proposes privileged action without human approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| WorkflowTemplate | id, name, category, tags, complexity_level, template_data, variables, tenant_id, is_certified |
| TaskDefinition | id, name, task_type, assigned_to, assigned_role, priority, sla_hours, dependencies, conditions, max_retry_attempts, escalation_rules |
| WorkflowTrigger | id, name, trigger_type, cron_expression, event_source, event_types, event_filters, webhook_url |
| Workflow | id, name, version, tenant_id, tasks, triggers, status, max_concurrent_instances, compensation steps via tasks, required_capabilities |
| WorkflowInstance | id, workflow_id, tenant_id, status, current_tasks, completed_tasks, failed_tasks, progress_percentage, sla_deadline, is_sla_breached, audit_trail |
| TaskExecution | id, instance_id, task_id, status, assigned_to, due_date, approval_decision, escalation_level, is_sla_breached, attempt_number |
| WorkflowConnector | id, name, connector_type, connection_config, is_validated, rate_limit_per_minute, health_status |
| WorkflowAuditLog | id, tenant_id, event_type, event_category, action, resource_id, user_id, result, impact_level, compliance_tags |

Supported task types: `automated`, `human`, `approval`, `notification`, `integration`, `conditional`, `parallel`, `subprocess`, `timer`, `script`.
Supported trigger types: `manual`, `scheduled`, `event`, `api`, `webhook`, `condition`, `file`, `email`.

## New Methods

### Execution Snapshotting and Recovery

```python
# Snapshot running execution state to durable storage
snap = await svc.snapshot_execution(tenant_id="acme", execution_id="exec-001")
# snap["snapshot_id"] -> "snapshot:exec-001_2025-..."

# Recover after process restart — no reprocessing of completed steps
restored = await svc.restore_from_snapshot(tenant_id="acme", snapshot_id=snap["snapshot_id"])
```

### SLA Breach Detection

```python
# Scan all active assignments; emits sla_warning at 80% elapsed, sla_breached when overdue
report = await svc.check_sla_breaches(tenant_id="acme")
# report["breached"] -> list of task assignments past deadline
# report["at_risk"]  -> list approaching deadline

status = await svc.get_sla_status(tenant_id="acme")
# status["healthy"] / ["at_risk"] / ["breached"] buckets for ops dashboard
```

### AI-Assisted Anomaly Detection

```python
# Record duration after each completed execution (also called automatically)
await svc.record_execution_duration(tenant_id="acme", workflow_id="wf-001", duration_seconds=127.4)

# Detect statistical outliers — flags executions >3σ from rolling mean (min 10 samples)
result = await svc.detect_anomalies(tenant_id="acme")
# result["anomalies"] -> list of {workflow_id, z_score, duration_seconds, execution_id}
# Emits workflow_execution_anomaly events; ntfy integration alerts on-call operators
```

### Cost Attribution and FinOps

```python
# Per-execution accumulated cost-weight
cost = await svc.get_execution_cost(tenant_id="acme", execution_id="exec-001")
# cost["accumulated_cost"] -> 14.25, cost["pct_complete"] -> 60.0

# Tenant-wide ranked breakdown for billing period
report = await svc.get_tenant_cost_report(tenant_id="acme", period="monthly")
# report["cost_by_workflow"] -> sorted list of {workflow_definition_id, cost}
```

### Distributed Tracing

```python
# Reconstruct waterfall trace from audit events; returns spans with elapsed_ms
trace = await svc.get_execution_trace(tenant_id="acme", execution_id="exec-001")
# trace["spans"] -> [{event, elapsed_ms, payload, created_at}, ...]
# trace["trace_context"] -> W3C traceparent/tracestate propagated to cross-capability tasks
```

### Multi-Tenant Execution Quotas

```python
# Configure limits (0 = unlimited)
await svc.set_tenant_quota(tenant_id="acme", max_concurrent=50, max_starts_per_minute=20, admin_id="ops")

# Real-time usage — useful for client-side backpressure before hitting QuotaExceededError
status = await svc.get_quota_status(tenant_id="acme")
# status["current_concurrent"] -> 12, status["starts_this_minute"] -> 4
```

## World-Class Enhancements (v2.0)

| # | Enhancement | Category | Description |
|---|-------------|----------|-------------|
| 1 | Distributed Saga Coordinator | Reliability | Persist saga steps to PostgreSQL outbox before execution; background coordinator replays compensation idempotently across restarts. Exposes `begin_saga`, `commit_saga_step`, `abort_saga`. |
| 2 | Blue/Green Version Routing | Release Engineering | Bind in-flight instances to the definition version that started them; `promote_release` atomically swaps the active slot. `get_version_routing_table` returns slot-to-definition mapping and in-flight counts. |
| 3 | Idempotent Deduplication Store | Correctness | Fast-path dedup check in `start_execution` before any DAG evaluation. Returns the cached execution record on key match. `purge_idempotency_keys` manages TTL. |
| 4 | Dynamic Retry Budget with Jitter | Resilience | `TaskRetryBudget` model with exponential backoff and full jitter. `complete_task` honours retryable errors; `get_retry_status` returns current attempt count and next scheduled retry. |
| 5 | DAG Diff and Migration Validator | DevOps | `diff_workflow_versions` computes added/removed/reordered tasks. `validate_migration_safety` checks all in-flight instances against a target version before `promote_release`. |
| 6 | Real-time SSE Event Streaming | Observability | `subscribe_execution_events` returns an `asyncio.Queue` fed by every `_emit` call. Flask view at `/api/v1/stream` delivers chunked SSE frames with 15-second keepalive heartbeats. |
| 7 | Cost Attribution and Budget Guardrails | FinOps | `cost_weight` per task; per-execution and per-tenant cost ledger. `set_execution_budget` raises `BudgetExceededError` when accumulated cost exceeds the configured ceiling. |
| 8 | Parallel Fan-Out / Fan-In | Execution Semantics | `ParallelGate` task type with `all`, `any`, and `quorum` join policies. `get_gate_status` returns completed/failed branch counts and quorum result. |
| 9 | Deterministic Replay Engine | Auditability | Monotonic sequence numbers per execution. `replay_execution(up_to_sequence)` reconstructs exact state from event history for forensics and deterministic unit testing. |
| 10 | SLA Breach Detection | Operations | Background SLA timers; `check_sla_breaches` emits `sla_warning` at 80% elapsed and `sla_breached` on expiry. Integrates with `ntfy` escalation chains. |
| 11 | Workflow Template Marketplace | Developer Experience | `submit_template` validates ≥80% test coverage. `certify_template` requires operator role. `search_templates` full-text search over certified templates. `instantiate_template` substitutes parameters into a certified DAG. |
| 12 | Execution Checkpoint Snapshotting | Durability | `snapshot_execution` serialises full runtime state to a durable record. `restore_from_snapshot` re-hydrates without reprocessing. Auto-snapshot every N completed tasks (default 5). |
| 13 | Multi-Tenant Rate Limiting | Multi-tenancy | Token-bucket quotas per tenant. `start_execution` enforces concurrency and starts-per-minute; raises `QuotaExceededError` with `retry_after`. `get_quota_status` exposes real-time usage. |
| 14 | Distributed Tracing Integration | Observability | W3C `traceparent`/`tracestate` propagated through execution records and cross-capability task payloads. `get_execution_trace` reconstructs a per-execution waterfall with elapsed milliseconds. |
| 15 | AI-Assisted Anomaly Detection | AIOps | Rolling 100-sample duration window per workflow. `detect_anomalies` applies 3σ z-score; emits `workflow_execution_anomaly` events and alerts via `ntfy`. `get_anomaly_report` returns history and z-scores. |

## Streaming Events

Events emitted to the composition event stream via Bytewax (`apg.composition.orchestration.lifecycle`).

| Event | Trigger |
|-------|---------|
| workflow_defined | New workflow definition created |
| workflow_validated | Workflow passes all graph validation checks |
| workflow_released | Release approved and workflow published to execution runtime |
| workflow_execution_started | New workflow instance started with idempotency key |
| workflow_execution_advanced | Execution moves to next task(s) |
| workflow_execution_completed | All terminal states reached |
| workflow_task_assigned | Human task assigned to user, role, or group |
| workflow_agent_registered | New workflow automation agent registered |
| execution_snapshotted | Execution state persisted to snapshot store |
| execution_restored | Execution state re-hydrated from snapshot |
| sla_warning | Task assignment at 80% of SLA elapsed time |
| sla_breached | Task assignment past SLA deadline |
| workflow_execution_anomaly | Execution duration flagged as statistical outlier (>3σ) |
| tenant_quota_set | Execution quota configured for a tenant |

Stream states: `draft → validated → released → running → waiting → completed → failed → retired`

## Edge Cases Handled

- Transactional workflows must declare compensation steps at definition time (`compensation_required_for_transaction`); this cannot be added after release, ensuring rollback paths are reviewed before the workflow handles real data.
- Retry policies on tasks require explicit upper bounds (`retry_policy_requires_limit`); unbounded retries are blocked to prevent stuck tasks occupying execution resources indefinitely.
- SLA-bound tasks that lack escalation rules produce `require_review` rather than `deny`, allowing SLA monitoring without mandatory escalation in environments where silent expiry is acceptable.
- Cross-capability tasks must reference a capability contract by ID; this creates a resolvable dependency that the registry can validate, preventing workflows from referencing deprecated or retired capabilities.
- `Workflow.validate_tasks` checks that all dependency IDs reference task IDs within the same definition, catching dangling references at model construction time before any persistence occurs.
- `WorkflowInstance.audit_trail` is forwarded to `audl` via `WorkflowAuditLog` records with `security_classification` and `retention_policy` fields for compliance-level retention.
- `detect_anomalies` requires a minimum of 10 samples per workflow before flagging outliers, preventing false positives during ramp-up.
- `restore_from_snapshot` validates tenant ownership before re-hydrating, preventing cross-tenant state injection.

## Composability

- **Upstream**: `composition_events` (execution lifecycle coordination via Bytewax), `composition_config` (reads workflow timeout and SLA config), `composition_access` (policy enforcement on writes)
- **Downstream**: Domain capabilities invoked as cross-capability task handlers; `composition_gateway` routes external API triggers to execution start endpoints
- **Peer**: `audl` (receives `WorkflowAuditLog` records), `ntfy` (sends task assignments, SLA breach alerts, anomaly notifications, escalation chains), `composition_registry` (workflow templates publishable to marketplace)

## Development Notes

- `WorkflowTemplate`, `TaskDefinition`, `WorkflowTrigger`, `Workflow`, `WorkflowInstance`, `TaskExecution`, `WorkflowConnector`, and `WorkflowAuditLog` are all Pydantic v2 models, not SQLAlchemy models; persistence is handled in `service.py`.
- The `validate_cron_expression` function accepts both 5-part and 6-part (with seconds) cron expressions.
- `assert_workflow_valid`, `assert_instance_active`, and `assert_task_executable` are runtime assertion helpers for use inside service methods; they are not guard decorators.
- All new v2.0 async methods use lazy `hasattr` initialisation for their in-memory stores (`_snapshots`, `_cost_ledger`, `_execution_timings`, `_execution_quotas`) — safe to call on a freshly constructed `WorkflowOrchestrationService` without any setup.
- Key files: `capability_contract.py` (executable contract and rule engine), `models.py` (Pydantic models and enums), `service.py` (lifecycle operations), `api.py` (API helpers), `views.py` (UI model helpers).

---

© 2025 Datacraft — www.datacraft.co.ke
