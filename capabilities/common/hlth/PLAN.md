# HLTH Capability Development Plan

## Build Strategy

HLTH already contains a large async health runtime. The immediate objective is
to make HLTH a coherent APG capability packet that generated applications can
compose without starting optional probes, ML backends, notification systems, or
deployment systems.

The packet will:

1. Document HLTH's component, check, baseline, prediction, alert, incident,
   remediation, deployment-gate, health-agent, Bytewax lifecycle, rule, UI, and
   adapter boundaries.
2. Expand the executable capability contract with lifecycle-oriented
   configuration, rules, routes, and theme components.
3. Add a dependency-light `HlthService` control plane beside the existing async
   `SystemHealthService`.
4. Add generated-application API helpers and view models.
5. Replace stale embedded semantic evidence with live contract-derived
   evidence.
6. Add first-class health-agent registration and Bytewax lifecycle-batch
   validation so AI reliability agents are composable under explicit
   guardrails.
7. Prove the packet with focused tests, implementation audit, publish-plan, and
   stale-marker scans.

## Implementation Tasks

### 1. Specification and Documentation

- Add `SPECIFICATION.md`.
- Add `PLAN.md`.
- Replace overclaiming `README.md`.
- Replace overclaiming package-summary text in `cap_spec.md`.
- Replace stale task planning text in `todo.md` with current follow-up work.

### 2. Capability Contract

- Add configuration for component registry, health checks, baselines,
  predictions, incidents, remediation reviews, deployment gates, adapters, and
  audit.
- Add configuration for health agents and Bytewax lifecycle streams.
- Add rules for component registration, disabled components, score ranges,
  critical alert evidence, incident ownership, stale baselines, low-confidence
  predictions, remediation approval, review notes, deployment blocking, and
  waiver review.
- Add rules for health-agent runtime/role/scope/owner/purpose/disclosure,
  privileged-role approval, and Bytewax lifecycle-batch routing.
- Add routes for checks, baselines, deployment gates, audit, adapters, agents,
  and lifecycle batches.
- Add theme components for component inventory, check timeline, baseline
  freshness, incident impact, deployment gates, adapter health, and audit
  decisions.
- Add theme components for health-agent roster and Bytewax lifecycle monitor.

### 3. Lifecycle Service

- Preserve the existing async `SystemHealthService`.
- Add dataclass records for component, check, baseline, prediction, alert,
  incident, remediation, deployment gate, health-agent, lifecycle-batch, and
  audit state.
- Add deterministic guardrail evaluation using `capability_contract.py`.
- Add summaries for dashboards and release evidence.

### 4. API and View Models

- Extend `api.py` with helpers for component registration, health checks,
  baselines, predictions, alerts, incidents, remediation requests, remediation
  decisions, deployment gates, health-agent registration, lifecycle-batch
  validation, record listing, and capability status.
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

- Run focused compile and HLTH package tests only.
- Run `apg capabilities implementation-audit --root capabilities/common/hlth`.
- Run `apg capabilities publish-plan capabilities/common/hlth --json`.
- Search primary HLTH package files for stale overclaiming or obsolete package
  markers.
- Run `git diff --check`.
- Perform focused code review and resolve emergent issues before commit.

## Deferred Work

- Live active probes, service discovery, and cloud/Kubernetes adapters.
- MONI/OpenTelemetry/Prometheus ingestion adapters.
- ML model training and measured prediction accuracy.
- Notification, ticketing, incident, remediation, and deployment adapters.
- Production persistence and retention enforcement.
- Durable Bytewax topology deployment.
- Live AI-runtime adapters for Codex, Claude Code, opencode, Pi, and future
  agent providers.
- Rendered dashboard/browser verification.
- Full repository test suite.
- Benchmark claims and runtime SLO validation.
