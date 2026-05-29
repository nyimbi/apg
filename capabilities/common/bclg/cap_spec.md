# Blockchain Ledger Services Capability Specification

- **Capability Name**: Blockchain Ledger Services
- **Capability ID**: `bclg`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `bclg`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `bclg_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `ledger_requires_owner`
- `transaction_requires_signature`
- `key_custody_required`
- `contract_requires_review`
- `high_value_transaction_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `bclg_ledger_ops` APG theme contract.
