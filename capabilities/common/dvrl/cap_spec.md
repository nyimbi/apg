# Data Virtualization Capability Specification

- **Capability Name**: Data Virtualization
- **Capability ID**: `dvrl`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `dvrl`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `dvrl_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `source_registration_requires_credentials`
- `restricted_query_requires_rbac`
- `sensitive_results_block_cache`
- `query_requires_lineage_capture`
- `high_cost_query_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `dvrl_federation_console` APG theme contract.
