# UI/UX Theming and Branding Capability Specification

- **Capability Name**: UI/UX Theming and Branding
- **Capability ID**: `them`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `them`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `them_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `theme_requires_owner`
- `publish_requires_approval`
- `brand_asset_requires_license`
- `accessible_contrast_required`
- `large_rollout_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `them_brand_system` APG theme contract.
