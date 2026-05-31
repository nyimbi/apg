# MONI Capability Development Plan

## Build Strategy

MONI already contains a large async monitoring runtime. The immediate objective
is to make MONI a coherent APG capability packet that generated applications can
compose without starting optional telemetry backends.

The packet will:

1. Document MONI's source, signal, SLO, alert, incident, remediation, rule, UI,
   agent, Bytewax lifecycle, and adapter boundaries.
2. Expand the executable capability contract with lifecycle-oriented
   configuration, rules, routes, and theme components.
3. Add a dependency-light `MoniService` control plane beside the existing async
   `MonitoringService`.
4. Add generated-application API helpers and view models.
5. Replace stale embedded semantic evidence with live contract-derived evidence.
6. Add first-class monitoring-agent registration and Bytewax lifecycle-batch
   validation so AI observability agents are composable under explicit
   guardrails.
7. Prove the packet with focused tests, implementation audit, publish-plan, and
   stale-marker scans.

## Implementation Tasks

### 1. Specification and Documentation

- Add `SPECIFICATION.md`.
- Add `PLAN.md`.
- Add a practical `README.md`.
- Replace overclaiming package-summary text in `cap_spec.md`.
- Replace stale task planning text in `todo.md` with current follow-up work.

### 2. Capability Contract

- Add configuration for source registration, signal governance, SLOs,
  incidents, remediation reviews, monitoring agents, Bytewax streaming,
  adapters, and audit.
- Add rules for source registration, disabled source denial, trace evidence,
  SLO route evidence, incident ownership, remediation review, review notes, and
  retention review.
- Add rules for monitoring-agent runtime/role/scope/owner/purpose/disclosure,
  privileged-role approval, and Bytewax lifecycle-batch routing.
- Add routes for sources, logs, SLOs, incidents, audit, adapters, agents, and
  lifecycle batches.
- Add theme components for source health, SLO burn rate, incident timeline,
  remediation approval, adapter status, monitoring-agent roster, Bytewax
  lifecycle monitor, and audit events.

### 3. Lifecycle Service

- Preserve the existing async `MonitoringService`.
- Add dataclass records for source, signal, SLO, alert, incident, remediation,
  monitoring-agent, lifecycle-batch, and audit state.
- Add deterministic guardrail evaluation using `capability_contract.py`.
- Add summaries for dashboards and release evidence.

### 4. API and View Models

- Extend `api.py` with helpers for source registration, signal ingestion, SLOs,
  alerts, incidents, remediation requests, remediation decisions, record
  listing, monitoring-agent registration, lifecycle-batch validation, and
  capability status.
- Add `view_models.py` for generated-application UI models.

### 5. Packaging Evidence

- Replace stale `app.py` embedded JSON with live contract-derived semantic
  evidence.
- Refresh `semantic_model.json` and `release_report.json`.
- Update `package_manifest.json`.
- Rename the legacy package contract test to `tests/test_package_contract.py`.
- Include agent and streaming manifests in semantic, release, and component
  evidence.

### 6. Verification and Review

- Run focused compile and MONI package tests only.
- Run `apg capabilities implementation-audit --root capabilities/common/moni`.
- Run `apg capabilities publish-plan capabilities/common/moni --json`.
- Search primary MONI package files for stale overclaiming or obsolete package
  markers.
- Run `git diff --check`.
- Perform focused code review and resolve emergent issues before commit.

## Deferred Work

- Live OpenTelemetry collector ingestion.
- Prometheus, ClickHouse, Elasticsearch, Grafana, SIEM/SOAR, PagerDuty, and
  notification adapters.
- Production persistence and retention enforcement.
- Rendered dashboard/browser verification.
- Full repository test suite.
- Benchmark claims and runtime SLO validation.
- Durable Bytewax topology deployment.
- Live AI-runtime adapters for Codex, Claude Code, opencode, Pi, and future
  agent providers.
