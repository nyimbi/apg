# APG Monitoring and Observability Capability

MONI is APG's tenant-scoped monitoring and observability control plane. It
provides source registration, governed signal ingestion, SLO definitions, alert
routing, incident records, remediation review, deterministic guardrails, UI
metadata, theming, and release evidence.

## Current Executable Runtime

The package contains:

- a deterministic capability contract in `capability_contract.py`
- a dependency-light lifecycle service in `service.py`
- direct generated-application helpers in `api.py`
- generated-application view models in `view_models.py`
- a live contract-derived semantic model in `app.py`
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
- observability lifecycle audit events

These records let generated APG applications compose observability controls
without hand-writing monitoring governance for each application.

## Guardrails

MONI evaluates observability actions through deterministic rules. Baseline
guardrails cover tenant context, source registration, disabled source denial,
trace metadata, PII log redaction, high-cardinality metrics, SLO alert routes,
critical alert routes, incident ownership, retention exceptions, approved
runbooks, independent remediation review, and review notes.

## Adapter Boundary

MONI is backend-neutral. Production adapters may bind to OpenTelemetry,
Prometheus-compatible collectors, metrics databases, log stores, trace stores,
Grafana-like dashboards, notification systems, incident tools, or SIEM/SOAR
systems. Adapters must honor MONI rule decisions, tenant isolation, alert route
requirements, incident ownership, remediation decisions, and audit evidence.

## Verification Scope

The capability packet is verified with focused compile checks, package contract
tests, lifecycle service tests, publish-plan evidence, stale-marker scans, and
diff checks. Full repository tests, live telemetry backends, APG auth/audit/
notification adapters, rendered dashboards, and performance benchmarks remain
separate validation tasks.
