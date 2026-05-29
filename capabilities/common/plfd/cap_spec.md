# Platform Foundation Capability Specification

- **Capability Name**: Platform Foundation
- **Capability ID**: `plfd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `plfd`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `plfd_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `foundation_service_requires_owner`
- `dependency_health_required`
- `configuration_baseline_required`
- `platform_change_requires_approval`
- `broad_platform_change_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `plfd_platform_foundation` APG theme contract.
