# Anomaly Detection Capability Specification

- **Capability Name**: Anomaly Detection
- **Capability ID**: `anom`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `anom`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `anom_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `detection_requires_monitoring_source`
- `baseline_requires_history`
- `critical_anomaly_requires_owner`
- `baseline_reset_requires_approval`
- `high_false_positive_rate_requires_tuning`

## UI

The package exposes 7 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `anom_signal_console` APG theme contract.
