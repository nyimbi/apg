# Video Conferencing Capability Specification

- **Capability Name**: Video Conferencing
- **Capability ID**: `vidc`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `vidc`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `vidc_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `meeting_requires_host`
- `external_guest_requires_policy`
- `recording_requires_consent`
- `recording_requires_encryption`
- `large_meeting_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `vidc_meeting_room` APG theme contract.
