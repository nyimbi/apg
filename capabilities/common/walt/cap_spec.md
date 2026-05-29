# Wallet and Payment Core Capability Specification

- **Capability Name**: Wallet and Payment Core
- **Capability ID**: `walt`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `walt`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `walt_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `wallet_requires_owner`
- `instrument_requires_encryption`
- `high_value_requires_mfa`
- `settlement_requires_reconciliation`
- `high_risk_transaction_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `walt_wallet_ops` APG theme contract.
