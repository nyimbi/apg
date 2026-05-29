# Environment Management Capability Specification

- **Capability Name**: Environment Management
- **Capability ID**: `envm`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `envm`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `envm_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `environment_requires_owner`
- `production_change_requires_approval`
- `promotion_requires_path`
- `secret_scope_requires_policy`
- `high_drift_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `envm_environment_ops` APG theme contract.
