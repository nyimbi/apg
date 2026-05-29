# Tenants Legacy Capability Specification

- **Capability Name**: Tenants Legacy
- **Capability ID**: `tens`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `tens`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `tens_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `legacy_tenant_requires_owner`
- `mapping_requires_validation`
- `migration_requires_approval`
- `access_boundary_required`
- `stale_legacy_tenant_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `tens_legacy_tenant_migration` APG theme contract.
