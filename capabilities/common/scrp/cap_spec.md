# Scraper/Data Harvesting Capability Specification

- **Capability Name**: Scraper/Data Harvesting
- **Capability ID**: `scrp`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `scrp`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `scrp_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `source_requires_owner`
- `source_terms_required`
- `pii_requires_handling_policy`
- `harvest_requires_schedule_policy`
- `sensitive_source_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `scrp_harvest_ops` APG theme contract.
