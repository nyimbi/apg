# Plugin/Extension Framework Capability Specification

- **Capability Name**: Plugin/Extension Framework
- **Capability ID**: `plgn`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `plgn`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `plgn_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `plugin_requires_owner`
- `plugin_requires_signature`
- `permissions_require_review`
- `plugin_requires_sandbox`
- `external_plugin_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `plgn_extension_marketplace` APG theme contract.
