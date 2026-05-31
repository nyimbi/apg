# Workflow Orchestration Capability Specification

## Purpose

`wflo` is the APG common capability for governed workflow orchestration. It lets generated applications compose tenant-scoped workflow definitions, steps, triggers, tasks, approvals, executions, events, compensation, first-class provider-neutral AI workflow agents, Bytewax lifecycle batches, audit events, UI screens, visual theming, and event-stream policy.

## Scope

The capability must support:

- Tenant-local workflow definitions with owner, readable name, version, bounded step set, retry policy, trigger policy, compensation reference, expected runtime, review state, and publication state.
- Step contracts for human, automation, approval, AI, and event steps, with required policy references for AI, automation, and event steps.
- Publication and retirement through explicit approval references.
- Executions that require published definitions, correlation IDs, Bytewax event stream policy, event history, state-change audit, cancellation, failure, completion, and compensation state.
- Human task assignment, claim, completion, escalation reason, due date, and event emission.
- Approval requests with approver and reason, plus decisions with evidence, rejection, and delegation support.
- AI workflow agents as first-class records, with stable ID, readable name, supported runtime, supported workflow-governance role, owner, purpose, scope, visible contribution disclosure, and human-review treatment for privileged roles.
- Bytewax-backed event-stream configuration for batch workflow mutations, runtime events, and lifecycle batches across definitions, publications, executions, tasks, approvals, compensation, workflow agents, and audit.
- UI route contracts and dependency-light view models for generated applications.

## Dependencies

Required:

- `mqeb` for production event/message composition.
- `auth` for actor, assignee, approver, and permission composition.
- `audl` for durable workflow audit trails.
- `aicr` for AI-step provider and AI-agent policy composition.

Optional:

- `schd`, `ntfy`, `comp`, `scpt`, and `them`.

## Configuration

The authoritative configuration lives in `capability_contract.py` and includes:

- `definitions`
- `steps`
- `execution`
- `tasks`
- `approvals`
- `workflow_agents`
- `agents`
- `governance`
- `observability`
- `streaming`
- `adapters`
- `ui`
- `theme`

## Rules

The deterministic rule engine covers:

- tenant context
- workflow owner, name, steps, maximum step review, duplicate step IDs, retry policy, publication approval, and retirement approval
- external trigger policy, AI step policy, automation policy, event policy, and long-running runtime review
- published-definition and correlation-ID requirements for execution start
- Bytewax event stream enforcement
- task assignment, claim-before-completion, and escalation reason
- approval approver, reason, decision evidence, and delegation target
- completion blocking by open tasks and pending approvals
- execution cancellation/failure reason
- compensation plan requirement
- AI workflow agent stable ID, readable name, runtime, role, scope, owner, purpose, disclosure, and privileged-role human approval
- Bytewax lifecycle batch mutation count, supported operation, and lifecycle stream enforcement
- workflow state-change audit
- tenant isolation
- Bytewax batch mutation enforcement

## Runtime

`service.WfloService` is the generated-application runtime. It stores deterministic in-memory state for:

- workflow definitions
- executions
- tasks
- approvals
- events
- workflow agents
- lifecycle batches
- audit events

The runtime enforces the same guardrails exposed by the contract rule engine and keeps live providers behind adapter boundaries.

## UI

The UI contract exposes:

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

## Production Boundary

This packet does not start live event buses, schedulers, distributed executors, notification providers, script runtimes, AI providers, durable workflow databases, or live Bytewax workers. Those are production adapters behind the APG composition layer.

## Acceptance Gates

- `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe the package clearly.
- `capability_contract.py` exposes configuration, deterministic rules, UI, theme, streaming, and adapter metadata.
- Runtime/API/view tests prove positive lifecycle behavior and negative guardrail behavior.
- First-class workflow-agent composition is provider-neutral across `codex`, `claude_code`, `opencode`, and `pi`; external CLIs remain behind AICR adapter contracts.
- Lifecycle batch governance uses Bytewax metadata only and does not introduce Kafka or broker-core processing.
- `semantic_model.json`, `package_manifest.json`, and `release_report.json` match the current contract.
- Focused compile, pytest, implementation audit, publish-plan, stale-marker scan, and diff check pass.
