# Workflow Orchestration Capability Specification

- **Capability Name**: Workflow Orchestration
- **Capability ID**: `wflo`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `wflo`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `wflo_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `workflow_requires_owner`
- `publish_requires_approval`
- `external_trigger_requires_policy`
- `ai_step_requires_policy`
- `long_running_execution_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `wflo_workflow_studio` APG theme contract.
