# ESG/Carbon Tracking Capability Specification

- **Capability Name**: ESG/Carbon Tracking
- **Capability ID**: `esgc`
- **Category**: common
- **Version**: 1.0.0

## Purpose

ESGC is APG's package-backed ESG and carbon-tracking capability. It provides
tenant-scoped emissions inventories, approved emission-factor libraries,
activity-data capture, carbon dioxide equivalent calculations, anomaly review,
compliance-mapped ESG reports, reduction target tracking, audit evidence, UI
route metadata, semantic-model publication, and publish-plan evidence.

The package now carries executable runtime behavior instead of generic record
storage: `service.py`, `models.py`, `carbon_engine.py`, `api.py`, and `views.py`
manage emissions inventories, factors, activities, sustainability reports,
targets, deterministic carbon calculations, dashboard summaries, compatibility
helpers, and APG rule enforcement.

## Provided Services

- `esgc_operations`

## Required Services

- `tenant_context`
- `pred` for forecasting and target trajectory integration
- `geos` for reporting-boundary and location evidence
- `comp` for ESG report compliance mapping and approvals
- `audl` for durable emissions and report audit trails
- `iotd` or external metering adapters for measured activity data when enabled

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `inventory_requires_owner`
- `factor_requires_approved_source`
- `emission_requires_boundary`
- `report_requires_approval`
- `emission_anomaly_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

The UI contract covers emissions dashboard, emissions inventory, factor
library, data sources, report builder, target tracker, audit evidence, and
settings surfaces.

## Theme

The package uses the `esgc_sustainability_ops` APG theme contract.

## External Runtime Boundary

ESGC keeps live IoT meters, utility data feeds, supplier portals, factor
databases, geospatial boundary services, forecasting services, compliance
systems, audit stores, and regulator submission endpoints behind APG integration
boundaries. Capability tests and publish evidence exercise deterministic package
behavior without requiring external ESG datasets or credentials. Production
deployments can bind ESGC to metering adapters, Scope 3 supplier data, official
emission-factor sources, GHG Protocol/ISSB/CSRD mappings, prediction services,
and audit systems through APG configuration and credential-vault services.
