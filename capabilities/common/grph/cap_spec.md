# Graph Data Management Capability Specification

- **Capability Name**: Graph Data Management
- **Capability ID**: `grph`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `grph`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `grph_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `node_write_requires_owner`
- `edge_write_requires_type`
- `restricted_relationship_requires_review`
- `deep_traversal_requires_review`
- `lineage_graph_requires_source_asset`

## UI

The package exposes 6 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `grph_relationship_console` APG theme contract.
