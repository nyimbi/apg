# Chat and Messaging Capability Specification

- **Capability Name**: Chat and Messaging
- **Capability ID**: `chat`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `chat`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `chat_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `room_requires_owner`
- `retention_policy_required`
- `external_guest_requires_policy`
- `restricted_content_requires_moderation`
- `large_room_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `chat_team_messaging` APG theme contract.
