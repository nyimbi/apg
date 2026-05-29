# Data Loss Prevention Capability Specification

- **Capability Name**: Data Loss Prevention
- **Capability ID**: `dlpd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `dlpd`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `dlpd_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `inspection_source_requires_policy`
- `sensitive_content_requires_classification`
- `high_severity_exfiltration_requires_block`
- `quarantine_requires_encryption`
- `large_export_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `dlpd_data_protection_ops` APG theme contract.
