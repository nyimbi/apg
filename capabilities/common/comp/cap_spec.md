# Compliance Management Capability Specification

- **Capability Name**: Compliance Management
- **Capability ID**: `comp`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `comp`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `comp_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `control_requires_owner`
- `stale_evidence_requires_refresh`
- `regulated_data_requires_dlp`
- `report_requires_approval`
- `overdue_finding_requires_escalation`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `comp_compliance_command_center` APG theme contract.
