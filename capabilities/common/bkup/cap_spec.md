# Backup and Restore Capability Specification

- **Capability Name**: Backup and Restore
- **Capability ID**: `bkup`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `bkup`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `bkup_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `backup_plan_requires_owner`
- `snapshot_requires_encryption`
- `restore_requires_integrity_check`
- `production_restore_requires_approval`
- `stale_restore_test_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `bkup_continuity_ops` APG theme contract.
