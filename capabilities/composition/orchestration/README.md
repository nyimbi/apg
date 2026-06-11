# Workflow Orchestration

## Overview

Workflow Orchestration provides the runtime engine for defining, validating, releasing, and executing multi-step business processes within the APG composition layer. It supports automated tasks, human task assignments, approval workflows, cross-capability integration tasks, transactional compensation, SLA escalation, and event-triggered execution — all coordinated through Bytewax.

The business value is a governed, auditable process layer that connects APG capabilities into end-to-end business workflows. Release governance (validation evidence, dry-run, rollback plan) prevents untested workflows from reaching production. Human task coordination and SLA escalation ensure that human-in-the-loop steps don't silently stall. Compensation steps on transactional workflows provide rollback safety when multi-step operations fail partway through.

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

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authenticate workflow operators and task assignees |
| audl | Persist immutable workflow and task audit records |
| ntfy | Send task assignment, SLA breach, and escalation notifications |
| registry | Register this capability in the global catalog |
| composition_events | Coordinate execution lifecycle events via Bytewax |
| composition_config | Read environment-specific workflow configuration values |

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

## New Features (v2)

### Execution Snapshotting and Recovery

`snapshot_execution` serialises the full runtime state (execution record, instance variables, signal queue, compensation log, suspension record) to a durable record. `restore_from_snapshot` re-hydrates a previous snapshot into the live stores — enabling point-in-time recovery for long-running workflows without reprocessing completed steps.

### SLA Breach Detection and Monitoring

`check_sla_breaches(tenant_id)` scans all active task assignments for overdue deadlines, emitting `sla_warning` events at 80% elapsed time and `sla_breached` events when deadlines have passed. `get_sla_status(tenant_id)` returns a triage view partitioned into `healthy`, `at_risk`, and `breached` buckets for the operations dashboard.

### AI-Assisted Anomaly Detection

`record_execution_duration` maintains a rolling 100-sample window of completion times per workflow. `detect_anomalies(tenant_id)` applies z-score analysis (threshold: 3σ) across workflows with 10+ samples, emitting `workflow_execution_anomaly` events for statistical outliers. Integrates with `ntfy` for on-call alerting.

### Cost Attribution and FinOps

`get_execution_cost` returns accumulated cost-weight for a specific execution. `get_tenant_cost_report` aggregates consumption by workflow ID across a period, giving FinOps teams a ranked cost breakdown without requiring an external billing tool.

### Distributed Tracing

`get_execution_trace` reconstructs a waterfall trace from audit events, annotating each span with elapsed milliseconds since execution start. Execution records accept an optional `trace_context` dict (W3C `traceparent`/`tracestate`) that is propagated to cross-capability task invocations, enabling end-to-end distributed traces across APG services.

### Multi-Tenant Execution Quotas

`set_tenant_quota(tenant_id, max_concurrent, max_starts_per_minute)` configures concurrency and throughput limits per tenant. `get_quota_status(tenant_id)` returns real-time usage against configured limits. The `start_execution` path will enforce these limits in the next release, raising `QuotaExceededError` with a `retry_after` hint when either limit is breached.

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

Stream states: `draft → validated → released → running → waiting → completed → failed → retired`

## Edge Cases Handled

- Transactional workflows must declare compensation steps at definition time (`compensation_required_for_transaction`); this cannot be added after release, ensuring that rollback paths are reviewed before the workflow handles real data.
- Retry policies on tasks require explicit upper bounds (`retry_policy_requires_limit`); unbounded retries are blocked to prevent a stuck task from occupying execution resources indefinitely.
- SLA-bound tasks that lack escalation rules produce `require_review` rather than `deny`, allowing SLA monitoring without mandatory escalation in environments where silent expiry is acceptable.
- Cross-capability tasks must reference a capability contract by ID; this creates a resolvable dependency that the registry can validate, preventing workflows from referencing capabilities that have been deprecated or retired.
- The `Workflow.validate_tasks` field validator checks that all dependency IDs reference task IDs within the same workflow definition, catching dangling references at model construction time before any persistence occurs.
- `WorkflowInstance.audit_trail` is a list of dicts embedded in the instance record; for compliance-level retention, these are also forwarded to `audl` via `WorkflowAuditLog` records with `security_classification` and `retention_policy` fields.

## Composability

- **Upstream**: `composition_events` (execution lifecycle coordination via Bytewax), `composition_config` (reads workflow timeout and SLA config), `composition_access` (policy enforcement on writes)
- **Downstream**: Domain capabilities invoked as cross-capability task handlers; `composition_gateway` routes external API triggers to execution start endpoints
- **Peer**: `audl` (receives `WorkflowAuditLog` records), `ntfy` (sends task assignments, SLA breach alerts, escalation notifications), `composition_registry` (workflow templates publishable to marketplace)

## Development Notes

- `WorkflowTemplate`, `TaskDefinition`, `WorkflowTrigger`, `Workflow`, `WorkflowInstance`, `TaskExecution`, `WorkflowConnector`, and `WorkflowAuditLog` are all Pydantic v2 models, not SQLAlchemy models; persistence is handled in `service.py`.
- The `validate_cron_expression` function accepts both 5-part and 6-part (with seconds) cron expressions.
- `assert_workflow_valid`, `assert_instance_active`, and `assert_task_executable` are runtime assertion helpers for use inside service methods; they are not guard decorators.
- Key files: `capability_contract.py` (executable contract and rule engine), `models.py` (Pydantic models and enums), `service.py` (lifecycle operations), `api.py` (API helpers), `views.py` (UI model helpers).
