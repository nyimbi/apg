# IoT Device Integration Capability Specification

- **Capability Name**: IoT Device Integration
- **Capability ID**: `iotd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `iotd`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `iotd_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `device_requires_identity`
- `telemetry_requires_encryption`
- `dangerous_command_requires_approval`
- `firmware_requires_signature`
- `stale_device_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `iotd_device_ops` APG theme contract.
