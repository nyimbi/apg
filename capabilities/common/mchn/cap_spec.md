# Multi-Channel Output Capability Specification

- **Capability Name**: Multi-Channel Output
- **Capability ID**: `mchn`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `mchn`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `mchn_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `channel_requires_owner`
- `template_requires_approval`
- `sensitive_output_requires_encryption`
- `unhealthy_channel_blocks_delivery`
- `large_delivery_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `mchn_omnichannel_output` APG theme contract.
