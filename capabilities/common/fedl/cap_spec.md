# Federated Learning Capability Specification

- **Capability Name**: Federated Learning
- **Capability ID**: `fedl`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `fedl`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `fedl_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `participant_requires_attestation`
- `round_requires_minimum_participants`
- `secure_aggregation_required`
- `privacy_budget_requires_review`
- `poisoning_signal_blocks_round`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `fedl_privacy_mesh` APG theme contract.
