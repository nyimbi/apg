# GEOS Geo-Spatial Services Capability

GEOS is APG's governed location-intelligence capability. It provides generated
applications with a dependency-light way to compose event-source registration,
geofencing, location-event processing, territory planning, spatial analytics,
privacy enforcement, AI location-agent coordination, tenant isolation, and
Bytewax lifecycle event integration.

The local package runs without live map providers, H3/GEOS native extensions,
routing engines, data warehouses, web servers, or stream processors. Production
geo engines attach through adapters declared by the capability contract.

## What GEOS Provides

- Tenant-scoped event-source registration with consent model and data residency policy.
- Tenant-scoped geofences: owner assignment, active rule, geometry validation, large-polygon review.
- Location-event processing: source registration, consent, accuracy, sensitive-location review, geofence matching.
- Territory management with overlap detection and review enforcement.
- Spatial analytics: heatmaps, DBSCAN clustering, proximity search, distance matrices, isochrones.
- Geocoding and reverse geocoding with pluggable provider adapters.
- Route calculation with multi-point waypoints and mode selection.
- Coordinate reference system (CRS) transforms (WGS84 → UTM and others).
- Elevation queries, timezone lookup, and address validation.
- Bulk ingestion helpers for events, geofences, and territories.
- Geofence compliance audit (consent coverage, stale geofence detection).
- First-class AI location agents for Codex, Claude Code, OpenCode, Pi, and future runtimes.
- Geofence state lifecycle management (active / paused / retired) with audit evidence.
- Framework-neutral API helpers and UI view models.
- Visual theme metadata and Bytewax stream lifecycle metadata.

## Main Files

| File | Purpose |
|------|---------|
| `SPECIFICATION.md` | Functional requirements, lifecycle, rules, UI, and adapter boundaries |
| `PLAN.md` | Implementation sequencing and review checklist |
| `capability_contract.py` | Executable configuration, rules, UI routes, theme, and Bytewax metadata |
| `service.py` | `GeosService` facade and all sub-service classes |
| `api.py` | Package API helpers and legacy FastAPI integration surfaces |
| `views.py` | Dashboard, map, geofence, event, analytics, agent, audit, and settings view models |
| `models.py` | Pydantic v2 models for all geospatial domain types |

## Quick Start

```python
from capabilities.common.geos.service import GeosService

svc = GeosService()

# Register an event source
svc.register_event_source(
    "src-mobile", "tenant-a", "Mobile GPS", "mobile",
    consent_model="explicit",
    data_residency_policy="ke-residency",
)

# Create a circular geofence
svc.create_geofence(
    "yard", "tenant-a", "Operations Yard", "fleet-ops",
    boundary={
        "type": "circle",
        "center": {"latitude": -1.286389, "longitude": 36.817223},
        "radius_meters": 500,
    },
)

# Ingest a location event
event = svc.process_location_event(
    "evt-1", "tenant-a", "src-mobile",
    "vehicle-1", "vehicle",
    -1.2865, 36.8171,
    location_consent_recorded=True,
)
# event["matched_geofences"] lists geofences the coordinate falls inside
```

The event is accepted only when tenant context, source registration, consent,
accuracy, privacy review, and data-residency guardrails are satisfied.

## API Reference

### Core Operations (synchronous)

| Method | Description |
|--------|-------------|
| `register_event_source(source_id, tenant_id, name, source_type, consent_model, data_residency_policy, ...)` | Register a GPS/IoT source with consent and residency policy |
| `create_geofence(geofence_id, tenant_id, name, owner, boundary, ...)` | Create circle, polygon, or rectangle geofence |
| `process_location_event(event_id, tenant_id, source_id, entity_id, entity_type, lat, lon, consent_recorded, ...)` | Ingest one location event; returns matched geofences |
| `create_territory(territory_id, tenant_id, name, owner, boundary, ...)` | Define a territory with optional overlap review |
| `run_spatial_analysis(analysis_id, tenant_id, spatial_index_available, aggregation_privacy_applied)` | Summarise events, geofences, and hotspot density |
| `register_location_agent(agent_id, tenant_id, name, runtime, role, scope, ...)` | Register an AI location agent |
| `change_geofence_state(tenant_id, geofence_id, status, reason, ...)` | Transition geofence to active / paused / retired |
| `dashboard_summary(tenant_id)` | KPI snapshot: counts of all entities |
| `list_*(tenant_id)` | List event sources, geofences, events, territories, agents, audits, analytics |

### Async Geospatial Methods

| Method | Description |
|--------|-------------|
| `geocode(request_id, tenant_id, address_text, provider)` | Free-text address → coordinate |
| `reverse_geocode(request_id, tenant_id, lat, lon, provider)` | Coordinate → address |
| `route_calculate(route_id, tenant_id, origin, destination, waypoints, mode)` | Distance + ETA between points |
| `isochrone(isochrone_id, tenant_id, center, travel_time_minutes, mode)` | Reachable area polygon |
| `proximity_search(search_id, tenant_id, center, radius_km, entity_types)` | Events within radius, sorted by distance |
| `polygon_intersect(operation_id, tenant_id, polygon_a, polygon_b)` | Boolean intersection test |
| `heatmap_generate(heatmap_id, tenant_id, resolution)` | Grid-density heatmap from location events |
| `clustering_spatial(cluster_id, tenant_id, eps_km, min_samples)` | Grid-cell DBSCAN clustering |
| `address_validate(validation_id, tenant_id, address_components)` | Component completeness check |
| `coordinate_transform(operation_id, tenant_id, lat, lon, from_crs, to_crs)` | WGS84 ↔ UTM |
| `boundary_check(check_id, tenant_id, lat, lon)` | Which geofences/territories contain a point |
| `distance_matrix(matrix_id, tenant_id, origins, destinations)` | O×D great-circle distance matrix |
| `time_zone_lookup(lookup_id, tenant_id, lat, lon)` | UTC offset + IANA timezone |
| `elevation_query(query_id, tenant_id, points)` | Elevation (m) for a list of coordinates |
| `geospatial_analytics(tenant_id)` | Aggregate KPIs across all geospatial entities |
| `bulk_process_events(tenant_id, events)` | Ingest a list of location events |
| `bulk_create_geofences(tenant_id, geofences, owner)` | Create multiple geofences in one call |
| `bulk_create_territories(tenant_id, territories, owner)` | Create multiple territories in one call |
| `export_geofences(tenant_id, fmt)` | Snapshot all geofences as structured payload |
| `geofence_compliance_audit(audit_id, tenant_id)` | Consent and residency compliance findings |
| `health_check(tenant_id)` | Service health and entity counts |

## New Methods — Usage Examples

### `proximity_search` — nearest events to a point

```python
result = await svc.proximity_search(
    search_id="srch-1",
    tenant_id="tenant-a",
    center={"latitude": -1.286389, "longitude": 36.817223},
    radius_km=2.0,
    entity_types=["vehicle", "driver"],
)
# result["matches"] sorted by ascending _distance_km
```

### `isochrone` — 10-minute driving reach from depot

```python
iso = await svc.isochrone(
    isochrone_id="iso-morning",
    tenant_id="tenant-a",
    center={"latitude": -1.286389, "longitude": 36.817223},
    travel_time_minutes=10,
    mode="driving",
)
# iso["polygon"] — 8-vertex approximate reach boundary
# iso["approx_radius_km"] — numeric radius
```

### `distance_matrix` — 2×2 origin/destination table

```python
matrix = await svc.distance_matrix(
    matrix_id="dm-001",
    tenant_id="tenant-a",
    origins=[
        {"latitude": -1.286389, "longitude": 36.817223},
        {"latitude": -1.300000, "longitude": 36.820000},
    ],
    destinations=[
        {"latitude": -1.295000, "longitude": 36.825000},
        {"latitude": -1.310000, "longitude": 36.810000},
    ],
)
# matrix["matrix"][i][j] — great-circle km from origin i to destination j
```

### `geofence_compliance_audit` — detect consent gaps

```python
audit = await svc.geofence_compliance_audit(
    audit_id="audit-q2",
    tenant_id="tenant-a",
)
# audit["findings"] lists high/low severity issues
# audit["risk_level"] — "high" if any critical finding present
```

### `coordinate_transform` — WGS84 to UTM

```python
transformed = await svc.coordinate_transform(
    operation_id="xfm-001",
    tenant_id="tenant-a",
    latitude=-1.286389,
    longitude=36.817223,
    from_crs="WGS84",
    to_crs="UTM",
)
# transformed["output"]["easting"], ["northing"], ["zone"]
```

## World-Class Enhancements (v2.0)

The following 15 improvements are planned to transform GEOS from a solid spatial
processing engine into a production-grade geospatial intelligence platform.
Implementation priority (P0–P4) is listed; see `WORLD_CLASS_IMPROVEMENTS.md`
for full technical detail.

| # | Enhancement | Priority | Impact |
|---|-------------|----------|--------|
| 1 | **Spatial Index Acceleration (R-Tree / S2)** — Replace O(n·m) geofence scan with in-memory R-tree; O(log n) matching. 100–1000× throughput gain at scale. | P0 | Critical |
| 9 | **Probabilistic Geofence Matching** — Model GPS accuracy as 2D Gaussian; emit PROBABLE_ENTER/EXIT with confidence scores. Reduces boundary-zone false positives 60–80%. | P0 | High |
| 2 | **Adaptive Dwell-Time Enforcement** — `min_dwell_seconds` on geofences suppresses drive-by false positives. ENTER event held until dwell elapsed. | P1 | High |
| 10 | **Temporal Geofences with Recurrence Rules** — iCal RRULE `schedule` field; geofences active only during specified hours/weekdays. Eliminates out-of-hours false alerts. | P1 | High |
| 12 | **Real-Time Geofence Occupancy Monitoring** — `max_occupancy` field + concurrent entity bookkeeping. CAPACITY_WARNING and CAPACITY_EXCEEDED events. | P1 | High |
| 3 | **Multi-Tenant Quota and Rate-Limit Engine** — Per-tenant caps on geofence count, event ingestion rate, and data-export size. Prevents noisy-neighbour effects. | P2 | Medium |
| 8 | **GeoJSON / TopoJSON / FlatGeobuf Export** — `export_spatial_layer` streaming to standard GIS formats; zero-friction integration with QGIS, Mapbox, deck.gl. | P2 | High |
| 11 | **Privacy-Preserving k-Anonymity Aggregation** — Heatmap and cluster cells with fewer than k entities merged until threshold met. GDPR-compliant analytics by design. | P2 | High |
| 13 | **Multi-Modal Route Cost Functions** — Pluggable `CostModel` protocol; built-in `DistanceCost`, `TimeCost`, `EmissionCost`; composable weighted sums. | P2 | Medium |
| 15 | **Geospatial Change Detection and Diff** — `boundary_diff` computes symmetric difference between boundary versions; `compute_affected_entities` re-evaluates impacted events. | P2 | Medium |
| 4 | **Snap-to-Road / HMM Map Matching** — Viterbi-based map matching against OSM road network; fleet route accuracy from ~60% to >95%. | P3 | High |
| 5 | **Convoy and Formation Detection** — `detect_convoy` applies spatiotemporal cross-correlation to concurrent location streams; returns membership, centroid, and coherence score. | P3 | Medium |
| 6 | **Offline-First Sync Protocol** — `sync_offline_events` accepts batches with client timestamps and sequence numbers; last-writer-wins + union semantics; per-event verdicts. | P3 | Medium |
| 7 | **Geospatial Explainability Layer** — `explain_spatial_decision` returns LIME/SHAP-inspired attribution for anomaly, hotspot, and prediction outputs. EU AI Act ready. | P3 | Medium |
| 14 | **Cross-Tenant Spatial Federation** — `federated_boundary_check` with cross-tenant permission tokens; returns presence/count only unless identity disclosure agreed; dual audit trail. | P4 | Medium |

## Verification

```bash
# Syntax check
./.venv/bin/python -m py_compile \
    capabilities/common/geos/__init__.py \
    capabilities/common/geos/capability_contract.py \
    capabilities/common/geos/service.py \
    capabilities/common/geos/api.py \
    capabilities/common/geos/views.py \
    capabilities/common/geos/app.py

# Unit tests
./.venv/bin/pytest -q \
    capabilities/common/geos/test_capability_contract.py \
    capabilities/common/geos/tests/test_package_contract.py

# Self-test
./.venv/bin/python -c \
    "from capabilities.common.geos import app; r=app.self_test(); print(r); assert r['passed']"

# Implementation audit
./.venv/bin/apg capabilities implementation-audit \
    --root capabilities/common/geos --json

# Publish plan
./.venv/bin/apg capabilities publish-plan capabilities/common/geos --json
```

---

*© 2025 Datacraft — www.datacraft.co.ke*
