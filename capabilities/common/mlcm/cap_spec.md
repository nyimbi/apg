# AI Model Lifecycle Management Capability Specification

- **Capability Name**: AI Model Lifecycle Management
- **Capability ID**: `mlcm`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `mlcm`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `mlcm_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `model_registration_requires_owner`
- `production_promotion_requires_approval`
- `deployment_requires_model_card`
- `low_eval_score_blocks_promotion`
- `drifted_model_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `mlcm_model_ops_console` APG theme contract.
