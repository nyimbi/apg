# APG Monitoring and Observability Capability

MONI is APG's tenant-scoped monitoring and observability control plane. It
provides source registration, governed signal ingestion, SLO definitions, alert
routing, incident records, remediation review, deterministic guardrails, UI
metadata, monitoring-agent composition, Bytewax lifecycle batch validation,
theming, durable review evidence, and release evidence.

## Current Executable Runtime

The package contains:

- a deterministic capability contract in `capability_contract.py`
- a dependency-light lifecycle service in `service.py`
- direct generated-application helpers in `api.py`
- generated-application view models in `view_models.py`
- a live contract-derived semantic model in `app.py`
- first-class monitoring-agent and Bytewax streaming manifests
- focused package tests under `tests/`

The larger async monitoring runtime remains available for backend execution. The
control plane is importable without OpenTelemetry, Prometheus, log stores, trace
stores, notification systems, incident tools, or APG runtime adapters.

## Lifecycle Records

MONI defines first-class records for:

- telemetry source registration
- metric, log, and trace signal metadata
- SLO definitions
- alerts and incident correlation
- remediation requests and reviews
- monitoring-agent registrations
- Bytewax lifecycle-batch validation evidence
- observability lifecycle audit events
- pending-review queues across signals, remediation requests, monitoring
  agents, lifecycle batches, alerts, and incidents

These records let generated APG applications compose observability controls
without hand-writing monitoring governance for each application.

## Guardrails

MONI evaluates observability actions through deterministic rules. Baseline
guardrails cover tenant context, source registration, disabled source denial,
trace metadata, PII log redaction, high-cardinality metrics, SLO alert routes,
critical alert routes, incident ownership, retention exceptions, approved
runbooks, independent remediation review, review notes, monitoring-agent
runtime/role/scope/owner/purpose/disclosure, privileged agent approval, and
Bytewax lifecycle stream routing.

## AI Agent Composition

Monitoring agents are first-class APG records. MONI supports the `codex`,
`claude_code`, `opencode`, and `pi` runtime identifiers and governs SLO,
alert, incident, anomaly, metric-quality, trace-correlation, and dashboard
review roles. Privileged roles without human approval are preserved as
`pending_review` records for operator review, while unsupported runtimes,
unsupported roles, missing ownership, missing purpose, missing scope, and
missing contribution disclosure remain blocking denials.

## Bytewax Lifecycle

MONI requires `bytewax` as the lifecycle processor for bulk metric, alert,
incident, SLO, and monitoring-agent mutations. The control plane records
accepted and denied lifecycle batches and rejects non-Bytewax core stream
declarations.

## Adapter Boundary

MONI is backend-neutral. Production adapters may bind to OpenTelemetry,
Prometheus-compatible collectors, metrics databases, log stores, trace stores,
Grafana-like dashboards, notification systems, incident tools, or SIEM/SOAR
systems. Adapters must honor MONI rule decisions, tenant isolation, alert route
requirements, incident ownership, remediation decisions, and audit evidence.
AI runtime adapters must honor MONI agent scope and contribution-disclosure
requirements rather than bypassing the control plane.

## Verification Scope

The capability packet is verified with focused compile checks, package contract
tests, lifecycle service tests, publish-plan evidence, stale-marker scans, and
diff checks. Full repository tests, live telemetry backends, APG auth/audit/
notification adapters, rendered dashboards, and performance benchmarks remain
separate validation tasks.
