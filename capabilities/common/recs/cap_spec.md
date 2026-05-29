# Recommender Systems Capability Specification

- **Capability Name**: Recommender Systems
- **Capability ID**: `recs`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`recs` gives APG applications a deterministic recommender runtime for
catalog registration, profile feature capture, ranking policies, model
training, recommendation generation, recommendation experiments, drift
inspection, and governance evidence.

The package is intentionally adapter-oriented. It executes useful local
recommendation behavior without a live feature store, model provider, vector
database, event pipeline, or experimentation platform, while preserving clear
boundaries for `pred`, `aicr`, `nlpc`, `mdm`, `etlp`, `audl`, and production
ranking services.

## Provided Services

- `personalized_recommendations`
- `ranking_policies`
- `catalog_matching`
- `experiment_optimization`
- `profile_features`

## Required Services

- `pred`
- `aicr`
- `nlpc`

Optional composition partners include `mdm`, `etlp`, `audl`, and `comp`.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`.

The executable configuration covers:

- enabled recommendation algorithms, model ownership, minimum training events,
  and drift monitoring;
- ranking policy requirements, diversity constraints, sensitive-attribute
  filtering, and minimum recommendation confidence;
- experiment approval, holdout, business metrics, and maximum unreviewed
  experiment allocation;
- tenant context, profile consent, recommendation audit, high-impact
  explainability, UI route enablement, and theme selection.

## Rules

- `tenant_context_required`
- `profile_consent_required`
- `ranking_policy_required`
- `model_training_requires_events`
- `high_impact_requires_explainability`
- `large_experiment_requires_review`

`RecsService.evaluate()` delegates to the deterministic rule engine and the
service layer enforces additional local package guardrails for catalog
readiness, profile consent, ranking policy thresholds, model owners, drift
monitoring, holdout allocation, business metrics, and experiment approval.

## Runtime Surfaces

- `models.py` defines tenant-scoped catalog items, profiles, ranking policies,
  recommendation models, training runs, recommendation sets, experiments, and
  audit events.
- `recommendation_runtime.py` provides deterministic helpers for stable IDs,
  algorithm validation, impact validation, feature normalization, label
  normalization, feature-affinity scoring, confidence calculation, explanation
  text, and drift status classification.
- `service.py` owns the in-process lifecycle for catalog registration, profile
  recording, ranking policy attachment, model training, recommendation
  generation, experiment creation, drift recording, compatibility records,
  list/query helpers, dashboard summaries, and audit events.
- `api.py` exposes dependency-light helpers over the service for generated APG
  applications and package tests.
- `views.py` returns route-aware models for the dashboard, recommendation
  console, model registry, catalog manager, profile feature view, experiment
  studio, ranking policy view, and governance surfaces.

## Executable Lifecycle

1. Register catalog items with features, tags, categories, and sensitive
   attributes.
2. Record tenant-scoped recommendation profiles with consent and feature
   vectors.
3. Attach a ranking policy with confidence, diversity, and sensitive filtering
   guardrails.
4. Train an owned model with sufficient events and drift monitoring.
5. Generate governed recommendations for consented profiles.
6. Create experiments with approval, holdout, business metrics, and review for
   large allocations.
7. Record model drift and inspect dashboard, model, recommendation, experiment,
   policy, and governance view models.

Negative paths block missing tenant context, insufficient training events,
missing model owners, missing drift monitoring, personalized recommendations
without consent, recommendations without ranking policy, high-impact
recommendations without explanations, invalid ranking thresholds, experiments
without approval, missing holdout, missing business metrics, and large
experiments without review.

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `recs_recommendation_console` APG theme contract.

## Adapter Boundaries

The current package does not call live feature stores, vector databases,
streaming event systems, hosted model providers, experimentation platforms,
audit exporters, or personalization services. Those systems should be
connected through explicit adapters after the deterministic local lifecycle
remains green.

## Focused Verification

```bash
./.venv/bin/pytest -q capabilities/common/recs/test_capability_contract.py capabilities/common/recs/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/recs --json
./.venv/bin/apg capabilities publish-plan capabilities/common/recs --json
```
