# Digital Twin Framework Capability Specification

- **Capability Name**: Digital Twin Framework
- **Capability ID**: `dtwn`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `dtwn`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `dtwn_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `twin_requires_owner`
- `simulation_requires_model`
- `telemetry_requires_authenticated_source`
- `simulation_requires_approval`
- `high_risk_prediction_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `dtwn_digital_twin_ops` APG theme contract.
