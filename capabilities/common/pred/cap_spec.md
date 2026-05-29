# Predictive Analytics Capability Specification

- **Capability Name**: Predictive Analytics
- **Capability ID**: `pred`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `pred`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `pred_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `forecast_requires_history`
- `production_score_requires_approved_model`
- `scoring_requires_feature_lineage`
- `high_impact_prediction_requires_explainability`
- `long_horizon_requires_review`

## UI

The package exposes 7 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `pred_forecast_console` APG theme contract.
