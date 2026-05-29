# AI Agent Composition Capability Specification

- **Capability Name**: AI Agent Composition
- **Capability ID**: `agnt`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `agnt`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `agnt_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `agent_requires_model`
- `agent_runtime_must_be_registered`
- `team_requires_agent`
- `handoff_endpoint_must_resolve`
- `workspace_runtime_requires_sandbox`
- `external_runtime_requires_approval`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `agnt_agent_ops` APG theme contract.
