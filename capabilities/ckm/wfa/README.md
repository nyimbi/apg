# Workflow Automation

## Overview

The Workflow Automation capability (`ckm_wfa`) provides a BPMN 2.0-compliant workflow engine for defining, deploying, and operating business processes across the APG platform. It covers the full lifecycle from process design (drag-and-drop visual designer, BPMN XML/JSON definitions, version control) through execution (instance management, task queues, SLA tracking) to governance (approval chains with independent-reviewer requirements, exception escalation, and AI-powered optimization recommendations).

The capability is purpose-built for regulated, multi-tenant environments: every workflow activation requires approval evidence, every human task requires an assignee and SLA deadline, every approval requires an independent reviewer and a decision reason, and every state transition is captured to the audit trail. AI workflow agents participate as named, scoped actors — optimizing routing, flagging bottlenecks, and suggesting process improvements — under the same registration and disclosure controls that govern all APG agents.

## Capability ID

`ckm_wfa`  Version: 1.0.0

## Provides

| Service | Description |
|---------|-------------|
| workflow_definitions | BPMN 2.0 process definitions with versioning, approval gates, and template library |
| workflow_instances | Process instance lifecycle management with state transitions and audit trail |
| task_orchestration | Human, approval, service, decision, notification, and subprocess task types with SLA enforcement |
| approval_governance | Independent-reviewer approval chains with mandatory decision reasons and rejection rationale |
| exception_management | Exception ownership, escalation policies, and SLA breach review workflows |
| workflow_analytics | Process metrics, bottleneck detection, performance prediction, and AI recommendations |
| wfa_agents | AI agent assist for process design, approval review, exception handling, SLA monitoring, and optimization |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Identity context, RBAC, and initiator/assignee resolution |
| conf | Tenant-scoped workflow configuration and limits |
| audl | Audit log sink for all workflow state changes and decisions |
| ckm_not | Task assignment notifications, approval request routing, SLA breach alerts, exception escalation messages |
| ckm_rtc | Collaboration sessions for approval reviews, design workshops, and exception resolution calls |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant scoping for all operations |
| definitions.owner_required | bool | true | Workflow definitions require accountable owner |
| definitions.version_required | bool | true | Semantic version required on every definition |
| definitions.approval_required_for_activation | bool | true | Activation blocked without approval evidence |
| definitions.supported_triggers | list | manual, schedule, event, api, form_submission | Allowed trigger types |
| instances.initiator_required | bool | true | Instances require named initiator |
| instances.definition_must_be_active | bool | true | Only active definitions can be instantiated |
| instances.state_change_requires_audit | bool | true | Every state change must be audited |
| tasks.assignee_required_for_human_tasks | bool | true | Human tasks must have an assignee |
| tasks.sla_required | bool | true | SLA configuration required on all tasks |
| tasks.due_at_required_for_sla | bool | true | Due timestamp required when SLA tracked |
| tasks.completion_evidence_required | bool | true | Evidence required to complete a task |
| approvals.independent_reviewer_required | bool | true | Reviewer cannot be the requester |
| approvals.decision_reason_required | bool | true | Reason required for every approval decision |
| approvals.rejection_reason_required | bool | true | Specific reason required on rejections |
| exceptions.sla_breach_requires_review | bool | true | SLA breaches trigger mandatory review |
| governance.batch_event_stream | string | "bytewax" | Batch mutations must route through Bytewax |

Supported task types: `human`, `approval`, `service`, `decision`, `notification`, `subprocess`

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /ckm-wfa/dashboard | GET | ckm_wfa:view | Overview |
| designer | /ckm-wfa/designer | GET | ckm_wfa:design | Design |
| definitions | /ckm-wfa/definitions | GET | ckm_wfa:design | Design |
| instances | /ckm-wfa/instances | GET | ckm_wfa:operate | Operations |
| tasks | /ckm-wfa/tasks | GET | ckm_wfa:participate | Operations |
| approvals | /ckm-wfa/approvals | GET | ckm_wfa:approve | Governance |
| exceptions | /ckm-wfa/exceptions | GET | ckm_wfa:operate | Operations |
| agents | /ckm-wfa/agents | GET | ckm_wfa:govern | Governance |
| rules | /ckm-wfa/rules | GET | ckm_wfa:govern | Governance |
| analytics | /ckm-wfa/analytics | GET | ckm_wfa:view | Insights |
| audit | /ckm-wfa/audit | GET | ckm_wfa:view | Governance |
| settings | /ckm-wfa/settings | GET | ckm_wfa:admin | Administration |

API prefix: `/ckm-wfa/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| definition_requires_owner | create_definition without owner | deny |
| definition_requires_version | create_definition without version | deny |
| activation_requires_approval | activate_definition without approval evidence | deny |
| instance_requires_active_definition | start_instance against non-active definition | deny |
| instance_requires_initiator | start_instance without initiator | deny |
| human_task_requires_assignee | create_task type=human without assignee | deny |
| sla_task_requires_due_at | create_task SLA-tracked without due_at | deny |
| task_completion_requires_evidence | complete_task without completion evidence | deny |
| approval_requires_independent_reviewer | record_approval where reviewer == requester | deny |
| approval_requires_decision_reason | record_approval without decision reason | deny |
| rejection_requires_reason | record_approval decision=rejected without reason | deny |
| sla_breach_requires_review | escalate_task with SLA breach and no review | require_review |
| exception_requires_owner | raise_exception without exception owner | deny |
| wfa_agent_requires_registration | Agent present but not registered | deny |
| wfa_agent_runtime_supported | Agent uses unsupported runtime | deny |
| wfa_agent_role_supported | Agent uses unsupported role | deny |
| wfa_agent_requires_scope | Agent without explicit scope | deny |
| wfa_agent_requires_disclosure | Agent contribution not disclosed | deny |
| workflow_state_change_requires_audit | State change without audit event | deny |
| batch_workflow_mutation_requires_bytewax | Batch mutation not using Bytewax | deny |

Supported agent runtimes: `codex`, `claude_code`, `opencode`, `pi`

Supported agent roles: `process_designer`, `approval_reviewer`, `exception_reviewer`, `sla_reviewer`, `optimization_reviewer`

## Data Models

| Model | Key Fields |
|-------|-----------|
| WBPMProcessDefinition | id, tenant_id, process_key, process_name, process_version, process_status, bpmn_xml, bpmn_json, process_variables, is_executable, is_suspended, parent_version_id |
| WBPMProcessInstance | id, tenant_id, process_id, business_key, instance_status, process_variables, current_activities, start_time, end_time, duration_ms, initiated_by, priority, due_date, error_count |
| WBPMProcessActivity | id, tenant_id, process_id, element_id, activity_type, assignee, candidate_users, candidate_groups, form_key, gateway_direction, event_type, due_date_expression |
| WBPMProcessFlow | id, tenant_id, process_id, element_id, source_activity_id, target_activity_id, condition_expression, is_default_flow |
| WBPMTask | id, tenant_id, process_instance_id, activity_id, task_name, task_status, assignee, owner, candidate_users, candidate_groups, task_variables, due_date, priority, estimated_effort_hours, parent_task_id |
| WBPMTaskHistory | id, tenant_id, task_id, action_type, old_value, new_value, performed_by, performed_at, action_context |
| WBPMTaskComment | id, tenant_id, task_id, comment_text, comment_type, parent_comment_id, thread_level, attachments |
| WBPMProcessTemplate | id, tenant_id, template_name, template_category, bpmn_template, template_variables, is_public, usage_count, rating_average, template_version |
| WBPMCollaborationSession | id, tenant_id, session_name, session_type, target_process_id, target_instance_id, conflict_resolution_mode, session_status, session_host |
| WBPMCollaborationParticipant | id, tenant_id, session_id, user_id, collaboration_role, join_time, is_active, cursor_position, selected_elements, participant_color |
| WBPMProcessMetrics | id, tenant_id, process_id, instance_id, metric_type, metric_name, metric_value, metric_unit, measurement_timestamp |
| WBPMProcessBottleneck | id, tenant_id, process_id, bottleneck_activity, bottleneck_type, severity, impact_score, affected_instances, confidence_score, recommendation, resolution_status |
| WBPMAIRecommendation | id, tenant_id, recommendation_type, target_process_id, confidence_score, implementation_effort, recommendation_status, expires_at, reviewed_by |
| WBPMProcessRule | id, tenant_id, process_id, rule_name, rule_type, rule_condition, rule_action, rule_priority, is_active, execution_count |

## Streaming Events

Events emitted to the ckm event stream via Bytewax.

Topic: `apg.ckm_wfa.lifecycle`

| Event | Trigger |
|-------|---------|
| workflow_definition_created | New process definition persisted |
| workflow_definition_activated | Definition promoted to active status after approval |
| workflow_instance_started | Process instance initiated |
| workflow_task_created | Task generated by process engine |
| workflow_task_completed | Task completion evidence recorded |
| workflow_task_approved | Approval decision recorded (approved) |
| workflow_task_rejected | Approval decision recorded (rejected) |
| workflow_exception_raised | Exception raised and owner assigned |
| workflow_agent_registered | AI workflow agent registered |

Batch mutation guardrail: `batch_workflow_mutation_requires_bytewax`

## Edge Cases Handled

- The `activation_requires_approval` rule blocks definition activation even when the requesting user is the definition owner — approval must come from a separate actor, preventing self-approval of workflow changes.
- `approval_requires_independent_reviewer` fires when `reviewer_same_as_requester: True`. Service-layer code must compare requester identity to reviewer identity before calling `record_approval`, not rely solely on the rule engine catching it post-submission.
- `WBPMProcessInstance` validates that `end_time > start_time` via Pydantic validator and also validates that `duration_ms` is consistent (within 1-second tolerance) when all three values are provided, catching clock-skew bugs at model creation time rather than at query time.
- `WBPMProcessFlow` rejects self-referencing flows (`source_activity_id == target_activity_id`) at the Pydantic layer, preventing infinite loops from being persisted in the BPMN definition.
- `WBPMTaskComment.thread_level` is bounded 0–10, preventing unbounded comment nesting that could cause rendering and query performance issues.
- SLA breach review (`sla_breach_requires_review`) triggers as `require_review` rather than `deny`, meaning SLA-breached tasks can still be escalated but are routed through a review queue — appropriate when SLA breach is a reporting obligation rather than a hard stop.
- `WBPMProcessBottleneck.confidence_score` is typed as `ConfidenceScore` (0.0–1.0), enforced by `AfterValidator`, so AI detection services cannot emit scores outside the agreed range without a validation error at the boundary.

## Composability

- **Upstream**: `auth` resolves assignees and approvers; `conf` provides tenant limits; `audl` receives all workflow audit events; `ckm_not` handles all outbound task and approval notifications; `ckm_rtc` provides collaboration sessions for design and review.
- **Downstream**: No CKM capabilities depend on `ckm_wfa`. It is the terminal orchestration layer in the CKM stack. External APG capabilities (compliance, procurement, HR) can trigger workflow instances via the `api` and `event` trigger types and receive task callbacks via `ckm_not` notifications.
- **Peer**: `ckm_not` and `ckm_rtc` are the two peer capabilities in the CKM stack. A typical deployment pattern: `ckm_not` handles async push; `ckm_rtc` handles synchronous review sessions; `ckm_wfa` orchestrates the overall process with approval chains that can spawn both.

## Development Notes

- Models use dual representation: `WBPMProcessDefinition` carries both `bpmn_xml` (canonical) and `bpmn_json` (optional, for visual designer). Keep them in sync — divergence between the two is a common source of bugs when replaying process history.
- Pydantic v2 is used throughout with `model_config = ConfigDict(extra='forbid', validate_by_name=True)`. The `WBPMCollaborationParticipant.validate_participant_color` validator has a precedence bug: `if v and not v.startswith('#') or len(v) != 7` should read `if v and (not v.startswith('#') or len(v) != 7)`. Fix before enabling color-coded presence in the designer.
- `WBPMAIRecommendation` carries an `expires_at` field. AI recommendations should be retired automatically. Wire a scheduled job against the `schd` adapter to mark expired recommendations rather than letting stale suggestions accumulate in the UI.
- `WBPMServiceConfig` embeds `encryption_key` and `jwt_secret` as plain strings. These must be injected from the `encr` adapter or a secrets manager — never commit values to configuration files or the database.
- Bytewax stream prefix in `WBPMServiceConfig` is `wbpm`, but the streaming manifest topic is `apg.ckm_wfa.lifecycle`. Align the prefix convention before connecting producers and consumers to avoid topic mismatch.
- AI agent roles span design-time (`process_designer`) and runtime (`approval_reviewer`, `exception_reviewer`, `sla_reviewer`, `optimization_reviewer`). Design-time agents have write access to BPMN definitions; runtime agents have read-only access to instances. Enforce this distinction at the service layer — the capability contract checks only role validity, not read/write scope.

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

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/ckm/wfa/__init__.py capabilities/ckm/wfa/capability_contract.py capabilities/ckm/wfa/lifecycle.py capabilities/ckm/wfa/app.py capabilities/ckm/wfa/test_capability_contract.py
./.venv/bin/pytest -q capabilities/ckm/wfa/test_capability_contract.py
./.venv/bin/python -c "import importlib; pkg = importlib.import_module('capabilities.ckm.wfa'); service = pkg.WfaLifecycleService('tenant-proof'); print(service.dashboard_summary())"
./.venv/bin/apg capabilities implementation-audit --root capabilities/ckm/wfa --json
./.venv/bin/apg capabilities publish-plan capabilities/ckm/wfa --json
```
