# Security Operations Capability Specification

- **Capability Name**: Security Operations
- **Capability ID**: `seop`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `seop`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `seop_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `detection_requires_alert_source`
- `incident_requires_owner`
- `critical_incident_requires_escalation`
- `response_requires_playbook_approval`
- `high_confidence_anomaly_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `seop_security_ops` APG theme contract.
