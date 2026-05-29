# Website Builder Capability Specification

- **Capability Name**: Website Builder
- **Capability ID**: `wsbl`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `wsbl`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `wsbl_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `site_requires_owner`
- `publish_requires_approval`
- `custom_component_requires_review`
- `public_site_requires_accessibility_pass`
- `privacy_banner_requires_consent_policy`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `wsbl_site_builder` APG theme contract.
