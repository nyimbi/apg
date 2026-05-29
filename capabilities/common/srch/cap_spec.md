# Search Engine Capability Specification

- **Capability Name**: Search Engine
- **Capability ID**: `srch`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `srch`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `srch_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `indexing_requires_owner`
- `restricted_query_requires_rbac_filter`
- `semantic_query_requires_embeddings`
- `large_result_window_requires_review`
- `bulk_index_requires_lineage`

## UI

The package exposes 7 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `srch_discovery_console` APG theme contract.
