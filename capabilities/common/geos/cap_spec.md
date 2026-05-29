# Geo-Spatial Services Capability Specification

- **Capability Name**: Geo-Spatial Services
- **Capability ID**: `geos`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package materializes the executable APG contract for `geos`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `geos_operations`

## Required Services

- `tenant_context`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `location_consent_required`
- `geofence_requires_owner`
- `event_source_must_be_registered`
- `sensitive_location_requires_review`
- `large_polygon_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `geos_location_intelligence` APG theme contract.
