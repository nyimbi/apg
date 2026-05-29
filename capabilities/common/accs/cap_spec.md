# Accessibility Services Capability Specification

- **Capability Name**: Accessibility Services
- **Capability ID**: `accs`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `accs`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `accs_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `audit_requires_standard`
- `violation_requires_remediation_owner`
- `published_ui_requires_contrast`
- `media_requires_captions`
- `critical_issue_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `accs_accessibility_ops` APG theme contract.
