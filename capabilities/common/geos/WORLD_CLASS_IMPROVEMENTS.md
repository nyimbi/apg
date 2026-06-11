# GEOS - World Class Improvement Plan

**Capability:** Geospatial (geos)
**Version:** 3.0.0 target
**Author:** Nyimbi Odero <nyimbi@gmail.com>
**Company:** Datacraft
**Date:** 2026-06-11

---

## Overview

The following 15 improvements transform the GEOS capability from a solid spatial
processing engine into a world-class geospatial intelligence platform. Each
improvement is grounded in production needs observed in fleet management,
logistics, smart-city, and field-operations deployments.

---

## 1. Spatial Index Acceleration via R-Tree / S2

**Problem:** Point-in-polygon and proximity lookups iterate all geofences
linearly — O(n * m) on every event ingestion. At scale (>10k geofences,
>100k events/min) this becomes the dominant bottleneck.

**Improvement:** Replace the brute-force scan in `_point_in_boundary` and
`_point_in_polygon` with an in-memory R-tree (via `rtree` or `shapely.strtree`)
per tenant. Insert geofence bounding boxes on creation; remove on retirement.
Proximity and boundary checks become O(log n).

**Impact:** 100-1000x throughput gain on geofence matching. Enables sub-10ms
event processing at 5k geofences/tenant.

---

## 2. Adaptive Geofence Dwell-Time Enforcement

**Problem:** The current model generates enter/exit events but has no notion
of dwell time. Operators routinely need "alert only if entity is inside for
more than N minutes" to suppress drive-by false positives.

**Improvement:** Add a `min_dwell_seconds` field to the geofence schema.
`process_location_event` tracks per-entity, per-geofence entry timestamps in
a TTL map. The ENTER event is only emitted after `min_dwell_seconds` elapses
without an exit. An IMMEDIATE_EXIT event cancels a pending dwell.

**Impact:** Eliminates the most common category of false-positive alerts in
fleet and field-ops deployments.

---

## 3. Multi-Tenant Quota and Rate-Limit Engine

**Problem:** There are no per-tenant limits on geofences, events/minute, or
data-retention windows. A single tenant can exhaust shared resources.

**Improvement:** Add a `GLSTenantQuota` model and a `QuotaEnforcer` service
that checks event-ingestion rate, active geofence count, territory count, and
data-export size against per-tenant caps before accepting operations. Caps are
configurable via the capability contract and overridable per-tenant via policy.

**Impact:** Production-grade multi-tenancy. Prevents noisy-neighbour effects.
Provides billing surface for usage-based pricing.

---

## 4. Snap-to-Road and Map Matching

**Problem:** Raw GPS coordinates for vehicles deviate 5-50 m from roads due
to multi-path error. Route analysis and territory-crossing detection is
inaccurate without road-snapping.

**Improvement:** Implement a `snap_to_road` async method using the
Hidden Markov Model map-matching algorithm (Viterbi over candidate road
segments). Attach to a PostGIS/OpenStreetMap road network via adapter.
Return both the snapped trajectory and the confidence of each snap decision.

**Impact:** Fleet route analysis accuracy improves from ~60% to >95%.
Required for driver-behaviour scoring and compliance audit.

---

## 5. Convoy and Formation Detection

**Problem:** Logistics and security use-cases need to know when multiple
entities are moving together as a group (convoy, formation, platoon). No
current support.

**Improvement:** Add a `detect_convoy` async method that ingests concurrent
location streams for a set of entity IDs and applies a spatiotemporal
correlation algorithm (sliding window cross-correlation of positions). Returns
convoy membership, formation centroid, coherence score, and dissolution events.

**Impact:** Enables escorted-cargo monitoring, platoon driving optimisation,
and coordinated patrol detection.

---

## 6. Offline-First Sync Protocol with Conflict Resolution

**Problem:** Mobile and IoT clients in low-connectivity environments queue
location events locally. When connectivity returns, events arrive
out-of-order with stale timestamps. The current event model has no
deduplication or conflict-resolution logic.

**Improvement:** Add a `sync_offline_events` async method that accepts a
batch of events with client-side timestamps and sequence numbers. Apply
last-writer-wins by server-ingestion timestamp for coordinate updates, but
union-semantics for geofence trigger records. Return a per-event
accept/reject/deduplicate verdict.

**Impact:** Reliable data ingestion for field-operations (mining, agriculture,
construction) where connectivity is intermittent.

---

## 7. Geospatial Explainability Layer

**Problem:** Machine-learning outputs (anomaly scores, trajectory patterns,
hotspot significance) are black boxes. Compliance-sensitive operators
(healthcare, finance, government) cannot audit or contest automated decisions.

**Improvement:** Add an `explain_spatial_decision` async method that takes
any scored geospatial result (anomaly detection, hotspot, prediction) and
returns a LIME/SHAP-inspired attribution: which input features (distance
from baseline, time-of-day deviation, speed, geofence membership) drove the
score and by how much. Output is human-readable JSON.

**Impact:** Compliance with EU AI Act transparency requirements. Reduces
alert fatigue by giving operators context to triage results.

---

## 8. GeoJSON / TopoJSON / FlatGeobuf Export Pipeline

**Problem:** The `export_geofences` method returns raw internal dicts.
Downstream GIS tools (QGIS, ArcGIS, Mapbox, deck.gl) require standard
geospatial formats.

**Improvement:** Implement a `export_spatial_layer` async method that
serialises geofences, territories, and event clusters to GeoJSON
Feature Collections, TopoJSON (topology-preserving for territories), and
FlatGeobuf (binary, indexed, streamable). Add a streaming variant for
large exports.

**Impact:** Zero-friction integration with the entire GIS tool ecosystem.
Enables spatial data to flow directly into BI platforms and reporting
pipelines.

---

## 9. Probabilistic Geofence Matching

**Problem:** GPS accuracy varies from 3 m (clear sky) to 50+ m (urban
canyon). A hard point-in-polygon test misclassifies events near boundaries
at a rate proportional to GPS error.

**Improvement:** Replace the binary `_point_in_boundary` decision with a
probabilistic one. Given a coordinate and its `accuracy_meters` (modelled as
a 2D Gaussian), compute the probability that the true position lies inside the
geofence. Emit PROBABLE_ENTER / PROBABLE_EXIT events with a confidence value;
only emit definitive ENTER/EXIT when confidence exceeds a configurable
threshold.

**Impact:** Reduces boundary-zone false positives by 60-80%. Essential for
compliance monitoring where incorrect exit events trigger contractual or
regulatory consequences.

---

## 10. Temporal Geofences with Recurrence Rules

**Problem:** Geofences currently have no notion of time. Operators need
geofences that are active only during working hours, on weekdays, or during
a specific recurring shift window.

**Improvement:** Add `schedule` to the geofence schema, supporting an iCal
RRULE expression (e.g. `FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=8,9,...,17`).
The event-processing pipeline evaluates the recurrence rule against the event
timestamp before applying geofence matching.

**Impact:** Eliminates entire classes of out-of-hours false alerts. Enables
"school zone" style geofences that activate only during school hours.

---

## 11. Privacy-Preserving Location Aggregation (k-Anonymity)

**Problem:** The heatmap and clustering outputs can de-anonymise individuals
when a cell contains fewer than k entities. This violates GDPR and local
data-protection requirements for consumer-facing applications.

**Improvement:** Implement k-anonymity suppression in `heatmap_generate` and
`clustering_spatial`. Cells with fewer than `k` unique entity IDs are merged
with neighbours until the cell reaches k-anonymity threshold. Expose k as a
configurable parameter per tenant, defaulting to 5. Log all suppression
decisions to the audit trail.

**Impact:** GDPR-compliant analytics output by design. Enables sharing of
aggregated location insights with third parties without legal review for each
export.

---

## 12. Real-Time Geofence Pressure / Capacity Monitoring

**Problem:** Venues, warehouses, and restricted zones need to know current
occupancy and trigger alerts when capacity thresholds are reached. Current
geofences count total entries but not concurrent occupancy.

**Improvement:** Add `max_occupancy` and `occupancy_alert_threshold` fields
to the geofence model. Track concurrent occupancy via ENTER minus EXIT
per-entity bookkeeping. Emit CAPACITY_WARNING and CAPACITY_EXCEEDED events
when thresholds are breached. Expose `get_geofence_occupancy` as a real-time
query.

**Impact:** Critical for venue management, warehouse safety compliance, and
pandemic-era crowd control applications.

---

## 13. Multi-Modal Route Cost Functions

**Problem:** Route optimisation uses distance or assumed constant speed. Real
logistics requires cost functions that incorporate tolls, vehicle load,
time-of-day traffic, driver hours-of-service regulations, and CO2 emissions.

**Improvement:** Refactor `route_calculate` to accept a pluggable `CostModel`
protocol. Ship three built-in implementations: `DistanceCost`,
`TimeCost` (with time-varying speed profiles), and `EmissionCost` (using
vehicle type and load factor). Allow composing cost models as weighted sums.

**Impact:** 15-25% improvement in optimised route quality for logistics
operators. Enables carbon-offset tracking and regulatory compliance for
fleet emissions reporting.

---

## 14. Cross-Tenant Spatial Federation

**Problem:** Multi-organisation deployments (e.g. city-wide logistics
consortiums, shared industrial parks) need controlled spatial queries across
tenant boundaries — e.g. "are any of tenant B's vehicles in my restricted
zone?" — without exposing full data.

**Improvement:** Implement a `federated_boundary_check` async method that
accepts a cross-tenant permission token. The requesting tenant specifies a
boundary; the system checks the target tenant's entities against it and
returns only presence/count (not identities or precise coordinates) unless
the cross-tenant agreement includes identity disclosure. All cross-tenant
queries are logged to both tenants' audit trails.

**Impact:** Enables B2B geospatial collaboration (port authorities and
shipping companies, municipalities and utilities) without the security and
compliance risk of full data sharing.

---

## 15. Geospatial Change Detection and Diff

**Problem:** Territory and geofence boundaries change over time (re-zoning,
fleet coverage updates, construction). There is no way to query what changed
between two snapshots or compute the affected entities.

**Improvement:** Implement `boundary_diff` and `compute_affected_entities`
async methods. `boundary_diff` computes the symmetric difference of two
boundary versions (added / removed / unchanged areas as GeoJSON). 
`compute_affected_entities` re-evaluates all location events from a time
window against the new boundary and returns entities whose geofence membership
changed due to the boundary modification, enabling targeted notifications.

**Impact:** Eliminates the need for full re-processing on boundary updates.
Provides a clear audit trail of territorial changes and their operational
impact — essential for SLA and compliance tracking.

---

## Implementation Priority

| # | Improvement | Effort | Impact | Priority |
|---|-------------|--------|--------|----------|
| 1 | Spatial Index Acceleration | M | Critical | P0 |
| 9 | Probabilistic Geofence Matching | M | High | P0 |
| 2 | Adaptive Dwell-Time Enforcement | S | High | P1 |
| 10 | Temporal Geofences (RRULE) | M | High | P1 |
| 12 | Capacity / Occupancy Monitoring | S | High | P1 |
| 3 | Multi-Tenant Quota Engine | L | Medium | P2 |
| 8 | GeoJSON Export Pipeline | S | High | P2 |
| 11 | k-Anonymity Suppression | M | High | P2 |
| 13 | Multi-Modal Route Cost Functions | M | Medium | P2 |
| 15 | Geospatial Change Detection | M | Medium | P2 |
| 4 | Snap-to-Road Map Matching | L | High | P3 |
| 5 | Convoy Detection | L | Medium | P3 |
| 6 | Offline Sync Protocol | L | Medium | P3 |
| 7 | Explainability Layer | L | Medium | P3 |
| 14 | Cross-Tenant Federation | XL | Medium | P4 |

Effort: S = 1-3 days, M = 1-2 weeks, L = 3-4 weeks, XL = 6+ weeks.

---

*© 2026 Datacraft. All rights reserved.*
