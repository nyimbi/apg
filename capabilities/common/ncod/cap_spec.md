# No-Code/Low-Code Builder Capability Specification

- **Capability Name**: No-Code/Low-Code Builder
- **Capability ID**: `ncod`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `ncod`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `ncod_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `app_requires_owner`
- `publish_requires_approval`
- `script_extension_requires_policy`
- `external_connector_requires_policy`
- `production_change_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `ncod_app_builder` APG theme contract.
