# Knowledge Graph Capability Specification

- **Capability Name**: Knowledge Graph
- **Capability ID**: `kngr`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `kngr`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `kngr_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `entity_resolution_requires_source`
- `semantic_enrichment_requires_confidence`
- `reasoning_requires_evidence`
- `deep_reasoning_requires_review`
- `uncurated_public_graph_blocked`

## UI

The package exposes 6 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `kngr_semantic_console` APG theme contract.
