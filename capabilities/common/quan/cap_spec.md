# Quantum Computing Capability Specification

- **Capability Name**: Quantum Computing
- **Capability ID**: `quan`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `quan`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `quan_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `backend_requires_approval`
- `circuit_requires_owner`
- `sensitive_input_requires_encryption`
- `job_requires_quota`
- `large_job_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `quan_quantum_lab` APG theme contract.
