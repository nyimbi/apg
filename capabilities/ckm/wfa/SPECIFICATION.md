# CKM Workflow Automation Specification

## Purpose

The CKM Workflow Automation capability (`ckm_wfa`) lets generated APG
applications compose tenant-scoped workflow definitions, active process
instances, task orchestration, approval governance, exception handling,
analytics metadata, audit evidence, visual route metadata, theme metadata, and
AI-agent assistance into ERP, CRM, CKM, GRC, and operations applications.

This package owns the executable contract, deterministic guardrails,
dependency-light lifecycle service, UI route metadata, theme metadata, Bytewax
stream declaration, generated semantic evidence, and focused proof commands.
Visual designers, persistent databases, external connectors, schedulers,
provider runtimes, and stream-worker deployments remain adapter concerns.

## Users And Jobs

- Process owners define workflow definitions with accountable ownership,
  versioning, trigger type, and variable schema.
- Operators start process instances from active definitions and monitor their
  state.
- Participants receive human, approval, service, decision, notification, and
  subprocess tasks.
- Reviewers approve or reject workflow work with independent review and reason
  evidence.
- Exception owners resolve failures, SLA breaches, connector errors, and
  routing gaps.
- Platform engineers bind persistent storage, identity, notification,
  collaboration, audit, scheduler, monitoring, and Bytewax workers.
- AI agents assist with process design, approval review, exception review, SLA
  review, and optimization under explicit registration and disclosure.

## Capability Boundary

`ckm_wfa` provides:

- workflow definition management;
- workflow instance execution state;
- task orchestration and completion evidence;
- approval governance;
- exception ownership and escalation metadata;
- workflow analytics metadata;
- AI WFA-agent registration and policy enforcement;
- Bytewax stream metadata for batch workflow mutation.

`ckm_wfa` requires:

- `auth` for identity and permission context;
- `conf` for tenant configuration;
- `audl` for durable audit evidence;
- `ckm_not` for task and exception notifications;
- `ckm_rtc` for collaborative review rooms and decision capture.

## Lifecycle

Definition lifecycle:

1. A definition is created with tenant, owner, version, trigger, and variable
   schema.
2. Activation requires approval evidence.
3. Active definitions can start instances.
4. Definition state changes require audit evidence.

Instance lifecycle:

1. An instance references an active definition.
2. The instance records initiator, context, and optional correlation key.
3. Tasks, approvals, and exceptions attach to the instance.
4. Completion and cancellation remain adapter extensions until durable storage
   and orchestration engines are bound.

Task lifecycle:

1. A task references an instance and task type.
2. Human tasks require assignee or queue ownership.
3. SLA-tracked tasks require explicit due-time handling and review when
   breached.
4. Completion requires evidence and records an audit event.

Approval lifecycle:

1. Approval records reference a task, requester, reviewer, decision, and reason.
2. Reviewer and requester must be independent.
3. Rejections require a reason.
4. Approval records are audit-visible.

Exception lifecycle:

1. Exceptions reference an instance, code, severity, details, and owner.
2. Missing ownership denies the exception record.
3. Escalation, notification, and collaboration room creation are adapter
   responsibilities.

AI-agent lifecycle:

1. Agent is registered with runtime, role, scope, tenant, and disclosure.
2. Runtime must be one of `codex`, `claude_code`, `opencode`, or `pi`.
3. Role must be one of the configured WFA roles.
4. Agent contributions are audit-visible and cannot bypass policy decisions.

## Rule Engine

Rules must deny or require review for:

- missing tenant context;
- missing definition owner;
- missing definition version;
- activation without approval evidence;
- instance start from inactive definition;
- instance start without initiator;
- human task without assignee or queue;
- SLA-tracked task without due-time evidence;
- task completion without evidence;
- non-independent approval review;
- approval decision without a reason;
- rejected approval without a reason;
- SLA breach without review;
- exception without owner;
- unregistered, unsupported, unscoped, or undisclosed AI agents;
- lifecycle state changes without audit evidence;
- batch workflow mutations that do not use Bytewax.

## UI And Theme

The APG Python UI contract exposes dashboard, designer, definitions, instances,
tasks, approvals, exceptions, agents, rules, analytics, audit, and settings
routes. The theme uses compact operational density with distinct treatments for
workflow design, definition registry, instance console, task queue, approval
queue, exception queue, WFA-agent panel, stream health, and audit events.

## Streaming

Batch workflow mutation must use Bytewax. The stream topic is
`apg.ckm_wfa.lifecycle`, and state covers definitions, instances, tasks,
approvals, exceptions, WFA agents, and audit events. Live Bytewax topology
deployment is an adapter concern, but the package declares and enforces the
guardrail.

## Adapter Boundaries

Adapters must handle:

- visual process designer canvas persistence and rendering;
- durable database storage and migrations;
- external connector execution;
- notification routing through `ckm_not`;
- collaborative review through `ckm_rtc`;
- authentication and permission checks through `auth`;
- audit durability through `audl`;
- scheduler and timer execution;
- Bytewax lifecycle topology and operational monitoring.

## Acceptance Gates

- Contract validates through the APG capability registry.
- Configuration schema includes definitions, instances, tasks, approvals,
  exceptions, WFA agents, governance, observability, adapters, UI, and theme.
- Rules cover definition, activation, instance, task, approval, exception,
  agent, audit, and Bytewax guardrails.
- Lifecycle service can create and activate definitions, start instances,
  create and complete tasks, record approvals, record exceptions, register
  agents, summarize state, and validate batch mutation streams.
- Generated semantic evidence exposes provides/requires, routes, rules, theme,
  and streaming.
- README, specification, plan, progress log, focused tests, implementation
  audit, publish plan, and stale-marker scan are current.
