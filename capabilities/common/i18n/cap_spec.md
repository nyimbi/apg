# Internationalization Capability Specification

- **Capability Name**: Internationalization
- **Capability ID**: `i18n`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `i18n`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `i18n_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `locale_requires_owner`
- `machine_translation_requires_review`
- `publish_requires_approval`
- `restricted_content_requires_filtering`
- `low_coverage_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `i18n_localization_workbench` APG theme contract.
