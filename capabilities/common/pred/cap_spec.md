# Predictive Analytics Capability Specification

- **Capability Name**: Predictive Analytics
- **Capability ID**: `pred`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`pred` provides executable, tenant-scoped predictive analytics behavior for APG
applications. It supports model registration and approval, feature-lineage
registration, governed forecasting, production scoring, scenario simulation,
drift reporting, audit evidence, route metadata, and theme-aware view models.

The package is intentionally dependency-light. Live model lifecycle, feature
store, ETLP, monitoring, cache, and AI runtime systems are adapter boundaries;
this package provides deterministic local behavior that APG applications,
examples, tests, and publish tooling can compose now.

## Provided Services

- `pred_operations`
- `forecasting`
- `predictive_scoring`
- `scenario_simulation`
- `feature_lineage`
- `prediction_monitoring`
- `model_explainability`

## Required Services

- `tenant_context`

## Optional Adapter Partners

- `aicr` for model-provider and inference adapters
- `mlcm` for managed model lifecycle promotion
- `etlp` for feature and training-data lineage
- `moni` for production drift telemetry
- `audl` for durable audit export
- `cach` for low-latency score caching
- `nlpc` for natural-language forecast explanation surfaces

## Runtime Behavior

`PredService` is the executable package surface. It stores local in-memory
records for:

- predictive models with owner, target, algorithm, approval, explainability,
  environment, history, and feature metadata;
- feature sets with source-system and lineage references;
- forecast runs with history counts, horizon, confidence-interval settings,
  deterministic forecast values, and long-horizon review state;
- score runs with model, feature set, entity, environment, impact,
  deterministic score, and explanation reference;
- scenario simulations with baseline, adjusted score, delta, and assumptions;
- drift reports with metric, score, threshold, and review status;
- audit events for registered models, feature sets, forecasts, scores,
  scenarios, drift reports, and approvals.

`predictive_runtime.py` owns deterministic helpers for stable IDs, environment
validation, impact validation, feature-name normalization, score calculation,
forecast projection, scenario projection, and drift status classification.

Compatibility helpers remain available:

- `create_record()` registers a deterministic predictive model for older
  generated package tests.
- `list_records()` returns registered predictive models.
- `PredRecord` aliases `PredictiveModel`.

## Rules And Guardrails

The capability contract exposes deterministic rules:

- `tenant_context_required`
- `forecast_requires_history`
- `production_score_requires_approved_model`
- `scoring_requires_feature_lineage`
- `high_impact_prediction_requires_explainability`
- `long_horizon_requires_review`

The service enforces those rules at the local execution boundary:

- all operations require tenant context;
- models require an owner and algorithm;
- forecasts require at least 24 historical observations;
- long horizons require recorded review;
- production scoring requires an approved model;
- scoring requires feature lineage;
- high-impact scoring requires model explainability and an explanation
  reference;
- scenarios require explicit assumptions;
- missing or cross-tenant models and feature sets are rejected.

## API Helpers

`api.py` exposes dependency-light helpers over the shared service instance:

- `capability_status()`
- `register_model()`
- `approve_model()`
- `register_feature_set()`
- `create_forecast()`
- `score_entity()`
- `simulate_scenario()`
- `record_drift()`
- `create_record()`
- `list_records()`
- `dashboard_summary()`

These helpers are framework-neutral. Web, worker, CLI, or generated-app
adapters should call them instead of coupling the package to a specific
framework.

## UI Routes And View Models

The APG Python UI route contract remains defined in `capability_contract.py`.
`views.py` materializes route-oriented view models:

- `dashboard_model()`
- `forecast_console_model()`
- `score_monitor_model()`
- `scenario_lab_model()`
- `model_board_model()`
- `governance_model()`

The view models expose package state, rule metadata, route metadata, and the
`pred_forecast_console` theme contract without requiring a web framework.

## Theme

The package uses the `pred_forecast_console` APG theme contract. It includes
tokens and component styling for forecast charts, score cards, scenario
matrices, and feature-lineage panels.

## Adapter Boundaries

Keep these integrations behind explicit adapters unless a future slice verifies
them directly:

- live model training, registry, deployment, and rollback;
- feature store and ETLP lineage ingestion;
- production scoring infrastructure;
- OpenTelemetry or external monitoring systems;
- cache invalidation and score materialization;
- natural-language explanation generation;
- durable audit export.

## Focused Verification

Use these commands for a PRED package slice:

```bash
./.venv/bin/python -m py_compile capabilities/common/pred/__init__.py capabilities/common/pred/models.py capabilities/common/pred/predictive_runtime.py capabilities/common/pred/service.py capabilities/common/pred/api.py capabilities/common/pred/views.py capabilities/common/pred/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/pred/test_capability_contract.py capabilities/common/pred/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/pred --json
./.venv/bin/apg capabilities publish-plan capabilities/common/pred --json
```
