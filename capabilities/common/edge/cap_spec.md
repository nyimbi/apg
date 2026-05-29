# Edge Computing Capability Specification

- **Capability Name**: Edge Computing
- **Capability ID**: `edge`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `edge`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `edge_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `node_requires_attestation`
- `workload_requires_signed_artifact`
- `sync_requires_conflict_policy`
- `edge_transport_requires_security`
- `long_offline_window_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `edge_operations_console` APG theme contract.
