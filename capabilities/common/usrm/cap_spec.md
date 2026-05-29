# User Management Capability Specification

- **Capability Name**: User Management
- **Capability ID**: `usrm`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `usrm`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `usrm_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `user_requires_identity`
- `invite_requires_consent_notice`
- `privileged_user_requires_mfa`
- `deprovision_requires_access_revocation`
- `bulk_user_action_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `usrm_user_lifecycle` APG theme contract.
