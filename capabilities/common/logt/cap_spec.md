# Logging and Tracing Capability Specification

- **Capability Name**: Logging and Tracing
- **Capability ID**: `logt`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `logt`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `logt_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `pipeline_requires_owner`
- `trace_context_required`
- `sensitive_log_requires_redaction`
- `log_export_requires_approval`
- `large_query_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `logt_observability_console` APG theme contract.
