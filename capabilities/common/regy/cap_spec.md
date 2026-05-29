# API/Service Registry Capability Specification

- **Capability Name**: API/Service Registry
- **Capability ID**: `regy`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `regy`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `regy_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `service_registration_requires_owner`
- `service_registration_requires_health_endpoint`
- `duplicate_service_name_blocked`
- `breaking_change_requires_review`
- `cross_tenant_discovery_denied`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `regy_service_catalog` APG theme contract.
