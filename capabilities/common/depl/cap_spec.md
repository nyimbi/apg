# Deployment Management Capability Specification

- **Capability Name**: Deployment Management
- **Capability ID**: `depl`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `depl`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `depl_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `release_requires_owner`
- `deployment_requires_health_gate`
- `production_requires_approval`
- `rollback_requires_plan`
- `large_canary_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `depl_release_ops` APG theme contract.
