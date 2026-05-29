# Custom Scripting Engine Capability Specification

- **Capability Name**: Custom Scripting Engine
- **Capability ID**: `scpt`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `scpt`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `scpt_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `script_requires_owner`
- `sandbox_required`
- `dangerous_permission_requires_approval`
- `external_network_requires_policy`
- `high_resource_script_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `scpt_script_workbench` APG theme contract.
