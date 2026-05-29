# Zero Trust Network Access Capability Specification

- **Capability Name**: Zero Trust Network Access
- **Capability ID**: `ztna`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `ztna`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `ztna_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `identity_must_be_verified`
- `device_posture_required`
- `resource_policy_required`
- `privileged_access_requires_mfa`
- `high_risk_access_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `ztna_zero_trust_ops` APG theme contract.
