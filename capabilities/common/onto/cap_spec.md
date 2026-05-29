# Ontology Management Capability Specification

- **Capability Name**: Ontology Management
- **Capability ID**: `onto`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `onto`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `onto_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `term_requires_owner`
- `publication_requires_approval`
- `breaking_change_requires_review`
- `low_confidence_mapping_requires_review`
- `duplicate_term_blocks_publication`

## UI

The package exposes 7 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `onto_vocabulary_workbench` APG theme contract.
