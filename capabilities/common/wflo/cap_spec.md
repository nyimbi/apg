# Workflow Orchestration Capability Specification

- **Capability Name**: Workflow Orchestration
- **Capability ID**: `wflo`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package executes the APG contract for `wflo` as a deterministic workflow
definition, publication, execution, task, approval, event, and governance
runtime.

WFLO gives composed APG applications a tenant-scoped local workflow core for:

- designing workflow definitions with versioned steps, owners, trigger
  policies, retry policies, compensation references, and expected runtime;
- enforcing owner, publication approval, external trigger, AI step, and
  long-running execution review rules;
- publishing approved definitions into an executable library;
- starting executions with correlation IDs and payloads;
- creating and completing human or automated tasks;
- requesting and recording approvals;
- emitting workflow events and audit events;
- exposing dashboard, designer, definition library, execution monitor, task
  inbox, approval center, analytics, settings, rule, route, and theme surfaces
  for UI composition.

Live event buses, schedulers, notification channels, scripting engines, AI
providers, durable workflow stores, and distributed execution engines are
adapter boundaries. The checked-in package supplies deterministic local
behavior that compiler output, capacity examples, tests, publish tooling, and
APG composition can execute without those live integrations.

## Provided Services

- `workflow_definitions`
- `event_orchestration`
- `task_routing`
- `approval_flows`
- `execution_monitoring`
- `capability_rules`
- `visual_theming`

## Required Services

- `mqeb` for production event transport or Bytewax/event bridge adapters
- `auth` for actor identity, workflow permissions, and approval evidence
- `audl` for durable workflow audit trails
- `aicr` for AI-step provider and policy integration
- Optional `schd`, `ntfy`, `comp`, and `scpt` adapters for schedules,
  notifications, composition discovery, and script execution

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

Important sections:

- `definitions`: owner requirement, versioning, publication approval, and max
  workflow size.
- `execution`: event-bus expectation, maximum runtime, retry policies, and
  compensation support.
- `approvals`: human approval, delegation, audit, and timeout escalation.
- `governance`: tenant context, execution audit, AI-step policy, and external
  trigger policy requirements.
- `ui`: workflow studio, execution monitor, task inbox, and approval center.
- `theme`: default `wflo_workflow_studio` visual theme and tenant override
  policy.

## Rules

- `tenant_context_required`
- `workflow_requires_owner`
- `publish_requires_approval`
- `external_trigger_requires_policy`
- `ai_step_requires_policy`
- `long_running_execution_requires_review`

These rules are enforced in `WfloService` before state-changing operations.
Deny decisions raise `PermissionError` with the rule reason. Review decisions
create review-required workflow definitions with `required_actions` so APG
workflows or human governance queues can continue the process.

## Runtime Behavior

`service.py` exposes `WfloService`, a dependency-light runtime with:

- `create_workflow_definition()` for tenant-scoped definitions, step
  normalization, owner checks, trigger policies, AI-step policies, retry
  policies, expected runtime review, and audit events;
- `publish_workflow()` for approval-backed publication;
- `start_execution()` for published workflow execution with correlation IDs and
  payloads;
- `create_task()` and `complete_task()` for task routing and completion;
- `request_approval()` and `record_approval()` for approval gates and
  decisions;
- `complete_execution()` for closing executions only after open tasks and
  pending approvals are resolved;
- `emit_event()` plus list/dashboard helpers for definitions, executions,
  tasks, approvals, events, and audit events;
- `create_record()` and `list_records()` compatibility shims backed by
  workflow definition creation and definition listing.

`workflow_runtime.py` owns the serializable dataclasses, stable ID generation,
step-type normalization, UTC timestamps, and rule required-action extraction.

`api.py` exposes dependency-light function wrappers over the service for APG
generated runtimes and package smoke tests. `views.py` exposes route-aligned
view models for dashboard, designer, definition library, execution monitor,
task inbox, approval center, analytics, and settings.

## UI

The package exposes 8 APG Python UI route contracts through `views.py` and the
package semantic model:

- `/wflo/dashboard`
- `/wflo/designer`
- `/wflo/definitions`
- `/wflo/executions`
- `/wflo/tasks`
- `/wflo/approvals`
- `/wflo/analytics`
- `/wflo/settings`

## Theme

The package uses the `wflo_workflow_studio` APG theme contract.

Theme tokens cover workflow studios with compact density, workflow canvases,
execution timelines, task inboxes, approval queues, runtime chips, SLA chips,
and decision status styling.

## Proof Commands

Focused package proof:

```bash
./.venv/bin/python -m py_compile capabilities/common/wflo/__init__.py capabilities/common/wflo/models.py capabilities/common/wflo/workflow_runtime.py capabilities/common/wflo/service.py capabilities/common/wflo/api.py capabilities/common/wflo/views.py capabilities/common/wflo/capability_contract.py capabilities/common/wflo/app.py capabilities/common/wflo/test_capability_contract.py capabilities/common/wflo/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/wflo/test_capability_contract.py capabilities/common/wflo/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/wflo --json
./.venv/bin/apg capabilities publish-plan capabilities/common/wflo --json
```

Global package health proof:

```bash
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

## Known Non-Goals

- No live event bus, scheduler, distributed executor, notification provider,
  script runtime, AI provider, or durable workflow database is invoked in this
  package.
- No external workflow side effects are emitted by tests or local package
  methods.
- Production queue semantics, timeout escalation, and distributed compensation
  belong behind APG composition adapters.
