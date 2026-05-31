# MONI Capability Specification

## Purpose

MONI provides APG applications with a tenant-scoped monitoring and observability
control plane. It defines how generated applications register telemetry sources,
ingest metrics/logs/traces, govern alert routing, track service-level objectives
and incidents, approve remediation actions, and expose operable UI surfaces.

The capability is split into two layers:

- **Capability control plane**: dependency-light records, deterministic rules,
  API helpers, generated-application view models, semantic-model publication,
  and audit evidence.
- **Runtime adapters**: OpenTelemetry collectors, metrics stores, log stores,
  trace stores, alert managers, notification systems, runbook executors,
  incident tools, SIEM/SOAR systems, and APG audit/notification integrations.

This packet also makes observability AI agents first-class APG citizens. Agents
can be implemented by fast-moving runtimes such as Codex, Claude Code, opencode,
Pi, or later adapters, but MONI owns the tenant-scoped registration,
permissions, scope, purpose, human-approval, contribution-disclosure, and audit
rules that make those agents composable inside generated applications.

## Capability Outcomes

MONI must let a generated application:

1. Register tenant-scoped telemetry sources with owner, service, environment,
   allowed signal types, route evidence, and lifecycle status.
2. Ingest metric, log, and trace signal metadata only after deterministic
   guardrail evaluation.
3. Deny telemetry without tenant context or without a registered source.
4. Block unredacted PII in logs.
5. Require review for high-cardinality metrics.
6. Create SLO definitions with threshold, window, owner, route, and status.
7. Open alert records and correlate alerts into incident records.
8. Require notification routes for critical alerts.
9. Require incident ownership for critical incidents.
10. Require approved runbooks and independent approval before production
    remediation can execute.
11. Record every lifecycle and rule decision as audit evidence.
12. Provide generated-application view models for dashboards, source inventory,
    signal explorer, SLOs, alert center, incidents, remediation queue, analytics,
    rule manager, audit timeline, adapters, and settings.
13. Publish semantic-model and release evidence from the live capability
    contract rather than stale embedded JSON.
14. Register first-class monitoring agents with supported runtime, role, owner,
    scope, purpose, contribution-disclosure, and privileged-role approval.
15. Validate monitoring lifecycle mutation batches through a Bytewax-first
    stream contract and explicitly reject non-Bytewax core stream declarations.

## Functional Scope

### Source Lifecycle

Source records define which applications or infrastructure components can emit
signals into MONI. Each source stores tenant, source ID, service name,
environment, owner, allowed signal types, route evidence, status, and timestamps.

Valid source statuses are `active`, `disabled`, and `retiring`.

### Signal Lifecycle

Signal records represent governed telemetry metadata. They include tenant,
source, signal type, name, value/summary, labels, severity, trace ID, service,
classification, decision, matched rules, and lifecycle status.

Valid signal statuses are `accepted`, `pending_review`, `denied`, and
`dropped`.

### Alert and Incident Lifecycle

Alerts are generated from rule or SLO breaches and must capture severity,
source, route, owner, deduplication key, acknowledgment, and resolution state.
Critical alerts require notification route evidence. Critical incidents require
an owner and escalation route.

### Remediation Lifecycle

Remediation requests must reference an incident, environment, requester,
runbook, evidence, reviewer, and decision state. Production remediation requires
approved runbook evidence and independent reviewer notes.

### Monitoring Agent Lifecycle

Monitoring agent records define AI-assisted observability contributors that can
be composed into generated applications. Each agent stores tenant, agent ID,
name, runtime, role, operating scope, accountable owner, purpose, contribution
disclosure, human-approval requirement, status, and timestamp.

Supported runtimes in this packet are `codex`, `claude_code`, `opencode`, and
`pi`. Supported roles are `slo_reviewer`, `alert_reviewer`,
`incident_reviewer`, `anomaly_triage`, `metric_quality_reviewer`,
`trace_correlation_reviewer`, and `dashboard_reviewer`.

Privileged roles are `slo_reviewer`, `alert_reviewer`, `incident_reviewer`, and
`anomaly_triage`. They require explicit human approval before registration.

### Bytewax Lifecycle Stream

MONI lifecycle batches represent bulk mutations that affect metrics, alerts,
incidents, SLOs, or monitoring-agent records. The executable contract requires
Bytewax as the lifecycle processor, uses `moni.lifecycle` as the lifecycle
stream name, and covers the `moni.metrics`, `moni.alerts`, `moni.incidents`,
`moni.slos`, and `moni.agents` topics. Non-Bytewax broker declarations are not
accepted as core lifecycle processors for this packet.

### Rules

The rule engine is deterministic. It returns `allow`, `deny`, or
`require_review` with matched rule names and effects.

Baseline rules:

- tenant context is required
- source registration is required
- disabled sources block signal ingestion
- metric ingestion requires a source identifier
- trace ingestion requires a trace ID and service name
- logs containing PII must be redacted
- high-cardinality metrics require review
- critical alerts require notification routes
- critical incidents require owners
- SLOs require service, objective, threshold, and alert route evidence
- production remediation requires approved runbooks
- remediation review requires an independent reviewer
- review notes are required for remediation and exception decisions
- telemetry retention above tenant limits requires review
- monitoring-agent runtimes and roles must be supported
- monitoring agents require scope, owner, purpose, and contribution disclosure
- privileged monitoring-agent roles require human approval
- monitoring lifecycle batches must declare Bytewax as the processor

## UI and Theming

MONI must expose compact operations-oriented UI metadata. The UI routes are
metadata only in this packet; generated APG applications render them in their
selected shell.

Required screens:

- dashboard
- sources
- metrics
- logs
- traces
- SLOs
- alerts
- incidents
- remediation
- analytics
- rules
- audit
- adapters
- agents
- lifecycle
- settings

## Integration Boundaries

MONI depends conceptually on:

- `conf` for tenant defaults and environment configuration
- `auth` for user and permission context
- `audl` for immutable audit evidence
- `mqeb` for event fanout and alert workflows
- `ntfy` for notification routes
- `cach` for query/result caching

The dependency-light packet must not require those capabilities at import time.
Adapters bind them at runtime.

## Non-Goals

- MONI does not implement a full production time-series database in the control
  plane packet.
- MONI does not make benchmark claims without named runtime backends.
- MONI does not execute production remediation without adapter evidence and
  explicit approval.
- MONI does not require OpenTelemetry, Prometheus, Grafana, SIEM/SOAR, or
  notification systems merely to publish its APG capability contract.
- MONI does not embed Codex, Claude Code, opencode, Pi, or any other AI runtime;
  it defines the first-class APG composition contract those adapters must honor.
- MONI does not use a broker as its core lifecycle stream dependency; Bytewax
  is the required processor for this packet.

## Acceptance Criteria

The MONI packet is serviceable when:

- `SPECIFICATION.md`, `PLAN.md`, and `README.md` explain the capability, usage,
  extension points, and adapter boundaries.
- `capability_contract.py` exposes configuration, rules, UI routes, and theme
  components that cover the lifecycle above.
- `service.py` includes a dependency-light `MoniService` that can register
  sources, ingest signals, create SLOs, create alerts/incidents, request and
  decide remediation, register monitoring agents, validate Bytewax lifecycle
  batches, and list audit evidence.
- `api.py` exposes callable helpers over `MoniService`.
- `view_models.py` exposes generated-application view models.
- `app.py`, `semantic_model.json`, and `release_report.json` are derived from
  current contract evidence.
- Focused package tests prove the rule engine, lifecycle service, monitoring
  agent guardrails, Bytewax lifecycle guardrail, view models, semantic model,
  and publish-plan path.
