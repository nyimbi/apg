# Continuous Integration and Delivery Capability Specification

- **Capability Name**: Continuous Integration and Delivery
- **Capability ID**: `cicd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `cicd`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `cicd_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `pipeline_requires_owner`
- `build_requires_secret_scope`
- `artifact_requires_signature`
- `promotion_requires_quality_gate`
- `high_parallelism_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `cicd_pipeline_ops` APG theme contract.
