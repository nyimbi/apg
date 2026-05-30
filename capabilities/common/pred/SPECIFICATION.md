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
- Package evidence that can be published and self-tested from the current
  executable contract.
- Focused tests for the contract, lifecycle, guardrails, view models, and
  package evidence.

## Actors

- Planner or analyst: creates forecasts, scores entities, and compares
  scenarios.
- Analytics owner: registers models and feature sets.
- Model governance team: approves predictive models and reviews drift.
- Platform operator: configures adapters, Bytewax event streaming, audit, auth,
  metrics, cache, and generated app deployment.

## Functional Requirements

### Model Lifecycle

- Register models with tenant, owner, algorithm, target, environment, training
  history, feature names, approval state, explainability state, and metadata.
- Approve models with an accountable approver.
- Require owner, algorithm, target, and tenant context before model
  registration.
- Require review for short training history and missing model feature metadata.

### Feature Lifecycle

- Register feature sets with owner, feature names, lineage refs, and source
  system.
- Deny missing owner, feature names, and source system.
- Require review when lineage is missing at registration time.

### Forecast Lifecycle

- Create forecasts from registered models, named series, history values, and a
  positive horizon.
- Deny missing model, series, insufficient history, and invalid horizons.
- Require review for horizons beyond the configured limit.

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
- Deny over-threshold drift persistence when no review evidence exists.

### UI and Theming

- Expose routes for dashboard, forecasts, scores, features, scenarios, models,
  drift, batch scoring, explainability, governance, audit, and settings.
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

- `get_capability_contract()` exposes at least 30 deterministic rules, at least
  12 UI routes, Bytewax adapter evidence, runtime adapter evidence, and theme
  component metadata.
- `PredService` executes model, feature set, forecast, score, scenario, drift,
  list, dashboard, and APG record compatibility flows.
- Guardrail tests prove denied cases fail before state is accepted, and
  review-required cases are surfaced through explicit review state or
  PermissionError depending on the lifecycle stage.
- `app.self_test()` passes and fails if route, rule, Bytewax, or runtime
  evidence becomes stale.
- Package JSON evidence can be regenerated from `app.semantic_model()` and
  `app.component_manifest()`.
