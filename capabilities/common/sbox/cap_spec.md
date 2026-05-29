# Sandbox/Testing Environment Capability Specification

- **Capability Name**: Sandbox/Testing Environment
- **Capability ID**: `sbox`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `sbox`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `sbox_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `sandbox_requires_owner`
- `sandbox_requires_isolation_profile`
- `secrets_require_redaction`
- `outbound_network_requires_approval`
- `long_lived_sandbox_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `sbox_safe_testing` APG theme contract.
