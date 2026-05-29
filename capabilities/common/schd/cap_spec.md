# Scheduling and Job Orchestration Capability Specification

- **Capability Name**: Scheduling and Job Orchestration
- **Capability ID**: `schd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `schd`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `schd_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `schedule_requires_owner`
- `timezone_required`
- `critical_job_requires_monitoring`
- `external_job_requires_approval`
- `long_running_job_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `schd_scheduler_ops` APG theme contract.
