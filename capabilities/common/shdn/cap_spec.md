# Shutdown and Lifecycle Control Capability Specification

- **Capability Name**: Shutdown and Lifecycle Control
- **Capability ID**: `shdn`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `shdn`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `shdn_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `service_requires_owner`
- `shutdown_requires_health_gate`
- `shutdown_requires_backup_snapshot`
- `production_shutdown_requires_approval`
- `force_shutdown_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `shdn_lifecycle_control` APG theme contract.
