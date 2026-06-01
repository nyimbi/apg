# PRED Specification

## Purpose

PRED provides APG with first-class predictive analytics for generated
applications. It supports governed model registration, feature lineage,
forecasting, scoring, scenario simulation, drift monitoring, UI composition, and
auditable policy enforcement.

## Scope

This packet establishes the executable baseline for PRED:

- Contract-driven configuration, schema, adapters, deterministic rules, UI
  routes, and visual theme tokens.
- A dependency-light runtime service for generated applications.
- UI view models that can be composed into APG screens.
- First-class AI prediction-agent composition metadata, runtime records,
  guardrails, and UI models.
- Bytewax-only lifecycle batch validation for generated-application mutation
  streams.
- Package evidence that can be published and self-tested from the current
  executable contract.
- Focused tests for the contract, lifecycle, guardrails, view models, and
  package evidence.

## Actors

- Planner or analyst: creates forecasts, scores entities, and compares
  scenarios.
- Analytics owner: registers models and feature sets.
- Model governance team: approves predictive models and reviews drift.
- AI prediction agent: assists with forecast review, score governance,
  explainability review, drift triage, and lifecycle governance through a
  provider-neutral APG contract.
- Platform operator: configures adapters, Bytewax event streaming, audit, auth,
  metrics, cache, and generated app deployment.

## Functional Requirements

### Model Lifecycle

- Register models with tenant, owner, algorithm, target, environment, training
  history, feature names, approval state, explainability state, and metadata.
- Approve models with an accountable approver.
- Require owner, algorithm, target, and tenant context before model
  registration.
- Preserve review-required short training history, missing feature metadata, and
  missing explainability approval outcomes as `pending_review` model evidence
  with matched rules and review reasons.

### Feature Lifecycle

- Register feature sets with owner, feature names, lineage refs, and source
  system.
- Deny missing owner, feature names, and source system.
- Preserve missing-lineage review outcomes as `pending_review` feature-set
  evidence with matched rules and review reasons.

### Forecast Lifecycle

- Create forecasts from registered models, named series, history values, and a
  positive horizon.
- Deny missing model, series, insufficient history, and invalid horizons.
- Preserve long-horizon review outcomes as `pending_review` forecast evidence
  with matched rules and review reasons.

### Scoring Lifecycle

- Score tenant-scoped entities with registered models, feature sets, feature
  values, environment, impact, and optional explanation references.
- Deny production scoring without approval.
- Deny scoring without feature lineage.
- Deny high-impact scoring without explainability.
- Deny missing entity id or feature values.

### Scenario Lifecycle

- Simulate scenarios from a model, baseline score, feature adjustments, and
  assumptions.
- Deny missing assumptions, adjustments, or baseline evidence.

### Drift Lifecycle

- Record drift reports with metric, score, threshold, review evidence, status,
  and audit events.
- Preserve above-threshold drift without review evidence as `pending_review`
  drift evidence with matched rules and review reasons.

### AI Agent Lifecycle

- Register provider-neutral AI prediction agents as first-class capability
  records.
- Support the runtimes `codex`, `claude_code`, `opencode`, and `pi`.
- Support forecast, score, feature-lineage, scenario, drift, model-release,
  explainability, batch-scoring, and prediction-steward roles.
- Require tenant context, supported runtime, supported role, bounded scope,
  accountable owner, documented purpose, and machine-contribution disclosure.
- Mark privileged score, drift, model-release, explainability, and
  batch-scoring roles as `pending_review` when human approval evidence is
  absent.
- Keep runtime invocation behind the AICR adapter boundary so APG can integrate
  rapidly changing agent runtimes without changing analytics contracts.

### Bytewax Lifecycle Batch

- Validate lifecycle mutation batches before accepting generated-application
  state changes.
- Require the `bytewax` processor for model, feature-set, forecast, score,
  scenario, drift, explainability, and prediction-agent batch operations.
- Persist accepted and denied batch evidence for dashboard, governance, batch,
  lifecycle, and audit views.

### UI and Theming

- Expose routes for dashboard, forecasts, scores, features, scenarios, models,
  drift, batch scoring, explainability, agents, lifecycle batches, governance,
  audit, and settings.
- Provide route-specific view models.
- Publish forecast-console theme tokens and component hints.

### Adapters

- Use Bytewax for batch scoring/event streams.
- Expose adapter keys for generated runtime, helper runtime, HTTP API, AICR,
  MLCM, ETLP, CONF, AUTH, AUDL, MONI, and CACH.

## Non-Goals

- Live model-provider inference.
- Live Bytewax stream execution.
- Persistent database migrations.
- Browser-rendered UI validation.
- Load, latency, drift, accuracy, and throughput benchmarking.

These are later integration and hardening tasks once the executable baseline is
stable.

## Acceptance Criteria

- `get_capability_contract()` exposes at least 39 deterministic rules, at least
  14 UI routes, Bytewax adapter evidence, runtime adapter evidence,
  first-class agent metadata, lifecycle-stream metadata, and theme component
  metadata.
- `PredService` executes model, feature set, forecast, score, scenario, drift,
  agent, lifecycle-batch, list, dashboard, and APG record compatibility flows.
- Guardrail tests prove denied cases fail before state is accepted, while
  review-required model, feature, forecast, and drift outcomes are retained as
  `pending_review` records with deterministic policy evidence.
- `app.self_test()` passes and fails if route, rule, Bytewax, or runtime
  evidence becomes stale.
- Package JSON evidence can be regenerated from `app.semantic_model()` and
  `app.component_manifest()`.
