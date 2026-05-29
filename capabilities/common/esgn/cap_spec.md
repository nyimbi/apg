# Digital Forms and eSign Capability Specification

- **Capability Name**: Digital Forms and eSign
- **Capability ID**: `esgn`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `esgn`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `esgn_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `form_template_requires_owner`
- `form_publication_requires_approval`
- `signing_requires_identity_verification`
- `evidence_requires_encryption`
- `regulated_form_requires_compliance_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `esgn_forms_signing` APG theme contract.
