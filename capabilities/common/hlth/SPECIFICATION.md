# HLTH Capability Specification

## Purpose

HLTH provides APG applications with a tenant-scoped health checks and
diagnostics control plane. It defines how generated applications register
components, record health checks, maintain health baselines, request
predictions, open alerts and incidents, approve remediation actions, evaluate
deployment gates, compose health AI agents, validate lifecycle batches through
Bytewax, and expose operable UI surfaces.

The capability is split into two layers:

- **Capability control plane**: dependency-light records, deterministic rules,
  API helpers, generated-application view models, semantic-model publication,
  and audit evidence.
- **Runtime adapters**: active probes, service discovery, MONI/OpenTelemetry
  feeds, ML engines, metrics stores, ticketing systems, notification systems,
  remediation runners, deployment systems, and APG audit/notification
  integrations.

HLTH also makes health and reliability AI agents first-class APG citizens.
Agents can be implemented by fast-moving runtimes such as Codex, Claude Code,
opencode, Pi, or later adapters, while HLTH owns tenant-scoped registration,
scope, purpose, human approval, contribution disclosure, and audit rules.

## Capability Outcomes

HLTH must let a generated application:

1. Register tenant-scoped system components with owner, type, environment,
   criticality, dependencies, status, and lifecycle timestamps.
2. Record health checks only after deterministic guardrail evaluation.
3. Deny health checks without tenant context, component ID, or registered
   component evidence.
4. Produce score, dimension, status, decision, and matched-rule evidence for
   each check.
5. Create critical alerts and incidents when health scores breach critical
   thresholds.
6. Require owner and notification route evidence for critical health alerts.
7. Create baseline records with sample count and freshness metadata.
8. Require review before stale baselines can be used for predictions.
9. Require review for predictions below the configured confidence threshold.
10. Require approved runbooks, production approval, independent reviewers, and
    review notes before production remediation can execute.
11. Block deployments while unresolved critical incidents exist unless a waiver
    is recorded.
12. Record lifecycle and rule decisions as audit evidence.
13. Provide generated-application view models for dashboard, components,
    checks, baselines, predictions, alerts, incidents, remediation, deployment
    gates, reports, audit, adapters, and settings.
14. Publish semantic-model and release evidence from the live capability
    contract rather than stale embedded JSON.
15. Register first-class health agents with supported runtime, role, owner,
    scope, purpose, contribution-disclosure, and privileged-role approval.
16. Validate health lifecycle mutation batches through a Bytewax-first stream
    contract.

## Functional Scope

### Component Lifecycle

Component records define which services, jobs, databases, queues, functions, or
infrastructure resources can emit health checks. Each component stores tenant,
component ID, name, component type, environment, owner, criticality,
dependencies, status, and timestamps.

Valid component statuses are `active`, `maintenance`, `degraded`, `retiring`,
and `disabled`.

### Health Check Lifecycle

Health check records represent governed health state. They include tenant,
component, dimension, score, summary, status, decision, matched rules, alert and
incident references, and timestamps.

Valid health check statuses are `healthy`, `degraded`, `critical`,
`pending_review`, and `denied`.

### Baseline and Prediction Lifecycle

Baseline records capture expected score and sample evidence for a component and
dimension. Prediction records capture predicted score, confidence, risk,
decision, and baseline evidence.

Predictions using stale baselines or low confidence require review.

### Alert and Incident Lifecycle

Critical health checks must create alert evidence. Critical alerts require an
owner and notification route. Critical incidents require an owner and route and
remain unresolved until an adapter or generated application resolves them.

### Remediation Lifecycle

Remediation requests must reference an active incident, requester, environment,
runbook, proposed action, reason, production approval state, reviewer, and
decision state. Production remediation requires attached runbook evidence,
production approval, independent reviewer, and review notes.

### Deployment Gate Lifecycle

Deployment gates evaluate unresolved critical incidents for a tenant and return
`allow`, `deny`, or `require_review`. Waivers must be explicit and auditable.

### Health Agent Lifecycle

Health agent records define AI-assisted reliability contributors that can be
composed into generated applications. Each agent stores tenant, agent ID, name,
runtime, role, operating scope, accountable owner, purpose, contribution
disclosure, human-approval requirement, status, and timestamp.

Supported runtimes in this packet are `codex`, `claude_code`, `opencode`, and
`pi`. Supported roles are `component_health_reviewer`, `baseline_reviewer`,
`prediction_reviewer`, `incident_reviewer`, `remediation_reviewer`,
`deployment_gate_reviewer`, and `dependency_map_reviewer`.

Privileged roles are `prediction_reviewer`, `incident_reviewer`,
`remediation_reviewer`, and `deployment_gate_reviewer`. Without explicit human
approval they are persisted as `pending_review` records with policy decision,
matched rules, review reasons, and required reviewer evidence so operators can
approve or reject them instead of losing the attempted registration.

### Bytewax Lifecycle Stream

HLTH lifecycle batches represent bulk mutations that affect components, health
checks, baselines, predictions, incidents, or health-agent records. The
executable contract requires Bytewax as the lifecycle processor, uses
`hlth.lifecycle` as the lifecycle stream name, and covers `hlth.components`,
`hlth.checks`, `hlth.baselines`, `hlth.predictions`, `hlth.incidents`, and
`hlth.agents` topics.

Denied lifecycle batches are persisted with `denied` status before
`PermissionError` is raised. This gives generated applications durable evidence
for routing violations, remediation work queues, and audit timelines.

### Review Evidence Lifecycle

HLTH records must preserve policy decisions on review-required and denied
records. The durable fields are `policy_decision`, `matched_rules`,
`review_reasons`, and `review_evidence`. Pending review queues cover health
checks, predictions, alerts, incidents, remediation requests, deployment gates,
health agents, and lifecycle batches.

## Rules

The rule engine is deterministic. It returns `allow`, `deny`, or
`require_review` with matched rule names and effects.

Baseline rules:

- tenant context is required
- component health updates require a component ID
- component health updates require registered active components
- disabled components block health checks
- health scores must be within valid range
- critical health scores require alert evidence
- critical alerts require owner evidence
- critical alerts require notification route evidence
- critical incidents require owner evidence
- critical incidents require notification route evidence
- stale baselines require review before prediction use
- low-confidence predictions require review
- remediation requests require runbook evidence
- production remediation requires approval evidence
- remediation review requires an independent reviewer
- remediation review notes are required
- unresolved critical incidents block deployment
- deployment waivers require review evidence
- health-agent runtimes and roles must be supported
- health agents require scope, owner, purpose, and contribution disclosure
- privileged health-agent roles require human approval evidence or review
- health lifecycle batches must declare Bytewax as the processor

## UI and Theming

HLTH must expose compact operations-oriented UI metadata. The UI routes are
metadata only in this packet; generated APG applications render them in their
selected shell.

Required screens:

- dashboard
- components
- checks
- baselines
- predictions
- alerts
- incidents
- remediation
- deployment gates
- reports
- audit
- adapters
- agents
- lifecycle
- settings

## Integration Boundaries

HLTH depends conceptually on:

- `conf` for tenant defaults and health policy
- `auth` for user and permission context
- `audl` for immutable audit evidence
- `moni` for observability feeds and health signals
- `mqeb` for health event fanout
- `ntfy` for alert and remediation notifications
- `cach` for health summary caching

The dependency-light packet must not require those capabilities at import time.
Adapters bind them at runtime.

## Non-Goals

- HLTH does not train production ML models inside the control-plane packet.
- HLTH does not execute production remediation without adapter evidence and
  explicit approval.
- HLTH does not require Kubernetes, cloud APIs, OpenTelemetry, MONI, ticketing,
  notification, or deployment systems merely to publish its APG capability
  contract.
- HLTH does not make benchmark or accuracy claims without named runtime
  backends and measured evidence.
- HLTH does not embed Codex, Claude Code, opencode, Pi, or any other AI runtime;
  it defines the first-class APG composition contract those adapters must honor.
- HLTH does not use a broker as its core lifecycle stream dependency; Bytewax is
  the required processor for this packet.

## Acceptance Criteria

The HLTH packet is serviceable when:

- `SPECIFICATION.md`, `PLAN.md`, and `README.md` explain the capability, usage,
  extension points, and adapter boundaries.
- `capability_contract.py` exposes configuration, rules, UI routes, and theme
  components that cover the lifecycle above.
- `service.py` includes a dependency-light `HlthService` that can register
  components, record health checks, create baselines, request predictions,
  create alerts/incidents, request and decide remediation, evaluate deployment
  gates, register health agents, validate Bytewax lifecycle batches, preserve
  review evidence, and list audit evidence.
- `api.py` exposes callable helpers over `HlthService`.
- `view_models.py` exposes generated-application view models.
- `app.py`, `semantic_model.json`, and `release_report.json` are derived from
  current contract evidence.
- Focused package tests prove the rule engine, lifecycle service, health-agent
  guardrails, Bytewax lifecycle guardrail, view models, semantic model, and
  publish-plan path.
