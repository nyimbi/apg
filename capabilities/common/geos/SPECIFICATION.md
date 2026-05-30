# GEOS Capability Specification

## Identity

- Capability ID: `geos`
- Display name: Geo-Spatial Services
- Category: `common`
- Owner: APG Platform Team
- Runtime shell: `apg_python`
- Theme: `geos_location_intelligence`

## Purpose

GEOS is the tenant-scoped location-intelligence capability for APG
applications. It governs event-source registration, geofencing, location-event
processing, territory planning, spatial analytics, sensitive-location review,
data residency, AI location-agent participation, audit evidence, and Bytewax
lifecycle events.

The package must remain usable without live map providers, native H3/GEOS
extensions, routing engines, warehouses, stream processors, or web servers.
Those systems remain adapter boundaries. Local package proof focuses on
deterministic location governance, lifecycle state, tenant isolation, and
composition behavior.

## Domain Model

GEOS owns these package-level records:

- event sources with consent model, source type, data residency policy, and
  sensitive-location posture;
- geofences with owner, boundary, trigger events, active rule, and event count;
- location events with entity identity, coordinate, accuracy, matched
  geofences, and processing status;
- territories with owner, type, boundary, overlap state, and status;
- spatial analytics runs with event/geofence/territory counts and hotspot
  counts;
- location agents with runtime, role, scope, disclosure, policy reference, and
  status;
- audit events for lifecycle decisions.

All mutable package-level state must be tenant-qualified so duplicate IDs in
different tenants cannot collide.

## Lifecycle

1. Register a location event source with tenant context, consent model, and data
   residency policy.
2. Record privacy review before enabling sensitive-location sources or events.
3. Create a geofence with owner, valid boundary, active rule, and spatial review
   when geometry is large.
4. Process a location event only when the source is registered, consent exists,
   accuracy is acceptable, and sensitive-location review is present when needed.
5. Match location events to tenant-local geofences.
6. Create territories with overlap review.
7. Run spatial analytics only when spatial index and aggregation privacy
   controls are present.
8. Register AI location agents with supported runtime, role, scope, disclosure,
   and policy evidence.
9. Change geofence state only with reason and audit evidence.
10. Emit tenant-scoped audit and Bytewax lifecycle events.

## Rules And Guardrails

- `tenant_context_required`: operations require tenant context.
- `location_consent_required`: location-event processing requires consent.
- `geofence_requires_owner`: geofences require accountable ownership.
- `event_source_must_be_registered`: location events require registered
  sources.
- `sensitive_location_requires_review`: sensitive-location processing requires
  privacy review.
- `large_polygon_requires_review`: large geofence geometry requires spatial
  review.
- `data_residency_policy_required`: event sources require residency policy.
- `active_geofence_rule_required`: geofences require active rules.
- `minimum_location_accuracy_required`: events must meet accuracy policy.
- `spatial_index_required`: analytics require spatial index evidence.
- `aggregation_privacy_required`: analytics require privacy-preserving
  aggregation.
- `location_agent_*`: AI location agents require registration, supported
  runtime, explicit scope, and contribution disclosure.
- `geos_state_change_*`: state changes require reason and audit evidence.
- `cross_tenant_location_access_denied`: tenant boundaries must not be crossed.
- `batch_location_mutation_requires_bytewax`: batch mutations require Bytewax
  event streams.

## UI And Theme

GEOS exposes route and view-model surfaces for dashboard, map console,
geofence editor, event monitor, territory manager, spatial analytics, privacy,
AI location agents, audit trail, and settings.

The `geos_location_intelligence` theme provides semantic tokens and component
metadata for maps, geometry editing, event streams, territories, agents, and
audit timelines.

## Adapter Boundaries

These integrations remain replaceable:

- map, geocoding, routing, and reverse-geocoding providers;
- H3, R-tree, geohash, and warehouse spatial-index providers;
- prediction, anomaly, and territory-optimization engines;
- edge location ingestion and device telemetry providers;
- audit, notification, workflow, master-data, and model-lifecycle services;
- Bytewax stream processors and low-latency event topology.

Local package tests must not require those systems.

## Acceptance Gates

```bash
./.venv/bin/pytest -q capabilities/common/geos/test_capability_contract.py capabilities/common/geos/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/geos --json
./.venv/bin/apg capabilities publish-plan capabilities/common/geos --json
git diff --check -- capabilities/common/geos
```
