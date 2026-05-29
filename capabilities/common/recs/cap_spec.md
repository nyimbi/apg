# Recommender Systems Capability Specification

- **Capability Name**: Recommender Systems
- **Capability ID**: `recs`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `recs`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `recs_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `profile_consent_required`
- `ranking_policy_required`
- `model_training_requires_events`
- `high_impact_requires_explainability`
- `large_experiment_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `recs_recommendation_console` APG theme contract.
