# Identity Federation Capability Specification

- **Capability Name**: Identity Federation
- **Capability ID**: `idfd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `idfd`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `idfd_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `provider_requires_signing_key`
- `saml_assertion_requires_encryption`
- `oidc_client_requires_redirect_allowlist`
- `privileged_federation_requires_mfa`
- `stale_metadata_requires_refresh`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `idfd_federation_console` APG theme contract.
