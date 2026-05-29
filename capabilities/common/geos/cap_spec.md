# Geo-Spatial Services Capability Specification

- **Capability Name**: Geo-Spatial Services
- **Capability ID**: `geos`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`geos` provides executable location intelligence for APG applications. It
manages tenant-scoped event sources, geofences, location events, territories,
spatial analytics, privacy controls, data-residency evidence, and map-oriented
UI models. The capability turns raw coordinates into governable business
events that other APG capabilities can compose into routing, field service,
asset tracking, compliance, workforce, delivery, and planning workflows.

The package includes a comprehensive asynchronous geospatial service layer for
geocoding, geofencing, territory management, route optimization, compliance,
trajectory analysis, hotspots, prediction, anomaly detection, visualization,
and streaming. It also exposes a dependency-light APG facade (`GeosService`)
that proves the core executable contract without requiring live map providers,
PostGIS, traffic APIs, weather APIs, or streaming infrastructure.

## Provided Services

- `geofencing`: create and evaluate circular, rectangular, and polygonal
  geofences with owners, active rules, triggers, and geometry review.
- `location_events`: register event sources and process consented location
  updates against tenant-scoped geofences.
- `spatial_analytics`: run privacy-preserving spatial summaries over
  geofences, territories, and processed events.
- `territory_management`: manage service, sales, delivery, and operational
  territories with overlap review.
- `location_prediction`: expose the integration boundary for predictive
  location intelligence through APG `pred` and `aicr`.
- `geos_operations`: provide compatibility operations for package inspection,
  generated APG tooling, and publish-plan evidence.

## Required Services

- `tenant_context`: all executable operations require tenant context.
- `pred`: predictive location and risk models.
- `aicr`: AI-assisted analytics and anomaly detection integration.
- `mdm`: master-data alignment for addresses, assets, customers, territories,
  and entities.

## Optional Services

- `ntfy`: geofence, dwell, route deviation, and compliance notifications.
- `edge`: edge ingestion for GPS, IoT, mobile, and vehicle events.
- `audl`: durable audit trail persistence.
- `wflo`: location-triggered workflow orchestration.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. The contract includes:

- `geofencing`: owner, vertex-count, accuracy, and active-rule requirements.
- `events`: source registration, location consent, retention, and edge
  ingestion controls.
- `analytics`: spatial-index, predictive-location, territory-overlap, and
  aggregation-privacy requirements.
- `governance`: tenant context, audit, residency, and sensitive-location review
  controls.
- `ui`: map console, geofence editor, event monitor, and spatial analytics
  toggles.
- `theme`: tenant-overridable location-intelligence visual tokens.

## Rules

- `tenant_context_required`: deny operations without tenant context.
- `location_consent_required`: deny location event processing without recorded
  location consent.
- `geofence_requires_owner`: deny geofence creation without an accountable
  owner.
- `event_source_must_be_registered`: deny location events from unregistered
  sources.
- `sensitive_location_requires_review`: deny sensitive-location processing
  without privacy review.
- `large_polygon_requires_review`: require spatial review for oversized
  geofence polygons.

## UI

The package exposes APG Python view models for:

- Dashboard: routes, rule metadata, theme metadata, summaries, and recent
  operational state.
- Map console: geofences, territories, events, and analytics layers.
- Geofence editor: geometry state, triggers, owners, and active-rule status.
- Event monitor: registered sources, processed events, consent status, and
  matched geofences.
- Territory manager: regions, overlaps, owners, and territory status.
- Spatial analytics: privacy-preserving aggregates and spatial index controls.
- Privacy console: residency, consent, sensitive-location review, and audit
  controls.

## Theme

The package uses the `geos_location_intelligence` APG theme contract. It
defines compact operational tokens and component-level visual contracts for map
consoles, geofence editing, event monitoring, and territory management.

## External Runtime Boundary

The in-repository APG facade uses deterministic geometry and in-memory state so
the capability remains executable in tests and generated package workflows.
Production deployments should wire PostGIS or equivalent spatial storage,
external geocoding/map providers, traffic and weather feeds, IoT/mobile event
ingestion, edge processing, notification delivery, audit vaults, and workflow
engines through APG adapters without changing the GEOS capability contract.
