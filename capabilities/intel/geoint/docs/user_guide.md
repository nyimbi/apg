# Geospatial Intelligence — User Guide

**Capability ID**: `intel_geoint` | **Domain**: `intel` | **Version**: `2.0.0`

---

## Overview

`intel_geoint` is the APG package-backed capability for governed geospatial intelligence applications. The service layer enforces lawful authority at every stage, from source registration through dissemination, while providing a rich async analysis surface covering imagery analysis, movement tracking, terrain modelling, and spatial analytics.

---

## Installation

```bash
pip install apg-intel-geoint
```

---

## Service Initialisation

```python
from capabilities.intel.geoint import GeospatialIntelligenceService

svc = GeospatialIntelligenceService(
    tenant_id="tenant-a",
    actor_id="analyst-1",
    db_url="postgresql://localhost/geoint",  # optional; in-memory if omitted
)
```

All state mutations are tenant-scoped. Passing `tenant_id` at construction sets the default for async methods; CRUD methods accept an explicit `tenant_id` argument.

---

## Core Workflow

### 1. Lawful Authority

Every area, source, and collection plan must reference a valid authority record.

```python
authority = svc.record_authority(
    "auth-1", "tenant-a",
    authority_type="mission_order",
    scope_reference="scope://mission/port-corridor",
    classification="secret",
    approver_id="approver-1",
    expires_at="2027-01-01",
    evidence_reference="evidence://auth-1",
)
```

### 2. Area of Interest

```python
area = svc.record_area(
    "area-1", "tenant-a",
    name="Port corridor",
    geometry_reference="-1.0,36.8,-0.8,37.2",  # minlat,minlon,maxlat,maxlon
    classification="secret",
    owner_id="owner-1",
    authority_id=authority["id"],
    evidence_reference="evidence://area-1",
)
```

### 3. Imagery Source

```python
source = svc.register_source(
    "src-1", "tenant-a",
    source_type="satellite",
    sensor_type="optical",
    resolution_class="high",
    owner_id="owner-1",
    authority_id=authority["id"],
    evidence_reference="evidence://src-1",
)
```

### 4. Collection Plan

```python
plan = svc.record_collection_plan(
    "plan-1", "tenant-a",
    authority_id=authority["id"],
    area_id=area["id"],
    source_id=source["id"],
    collection_mode="continuous",
    retention_days=90,
    approval_reference="approval://plan-1",
    evidence_reference="evidence://plan-1",
)
```

### 5. Observation → Feature → Change → Assessment → Dissemination

```python
obs = svc.record_observation(
    "obs-1", "tenant-a", plan["id"],
    observation_reference="s3://imagery/obs-1.tif",
    captured_at="2026-06-01T06:00:00Z",
    geospatial_accuracy_score=0.92,
    evidence_reference="evidence://obs-1",
)

feature = svc.record_feature(
    "feat-1", "tenant-a", obs["id"],
    feature_type="facility",
    geometry_reference="-0.9,37.0",
    confidence_score=0.88,
    analyst_id="analyst-1",
    evidence_reference="evidence://feat-1",
)

change = svc.record_change(
    "chg-1", "tenant-a", feature["id"],
    change_type="construction",
    severity="high",
    confidence_score=0.85,
    analyst_id="analyst-1",
    evidence_reference="evidence://chg-1",
)

assessment = svc.record_assessment(
    "asmt-1", "tenant-a", change["id"],
    assessment_type="threat",
    classification="secret",
    analyst_id="analyst-1",
    evidence_reference="evidence://asmt-1",
)

dissemination = svc.record_dissemination(
    "diss-1", "tenant-a", assessment["id"],
    audience="j2-staff",
    release_marking="SECRET//REL",
    approval_reference="approval://diss-1",
    evidence_reference="evidence://diss-1",
)
```

---

## Async Analysis Methods

All analysis methods are `async`. Run them inside an `asyncio` event loop or
from an `async` function.

### Satellite Imagery Analysis

```python
result = await svc.satellite_imagery_analysis("obs-1", "change_detection")
# {analysis_id, accuracy_score, detected_feature_count, change_probability, resolution_class}
```

### Multi-Spectral Band Analysis

Computes NDVI, NDWI, normalised burn ratio (NBR), and thermal anomaly score.

```python
result = await svc.multispectral_band_analysis("obs-1", ["visible", "nir", "swir", "thermal"])
# {band_indices: {ndvi, ndwi, nbr, thermal_anomaly_score}, band_classes: {vegetation_health, water_presence}}
```

### Change Detection

```python
result = await svc.change_detection(
    location={"lat": -0.9, "lon": 37.0},
    date1="2026-01-01",
    date2="2026-06-01",
)
# {total_changes_detected, significant_change_count, detection_confidence}
```

### Change Velocity

Rate-of-change accumulation with trend classification and next-window forecast.

```python
result = await svc.change_velocity("area-1", window_days=30)
# {rate_per_day, trend: accelerating|stable|decelerating, forecasted_next_window}
```

### Facility Identification

```python
result = await svc.facility_identification(
    coordinates={"lat": -0.9, "lon": 37.0},
    radius=25.0,
)
# {facility_count, facilities: [{feature_id, distance_km, confidence}]}
```

### Terrain Analysis

```python
coords = [{"lat": -1.0, "lon": 36.8}, {"lat": -1.1, "lon": 37.0}, {"lat": -0.8, "lon": 37.2}]
result = await svc.terrain_analysis(coords)
# {centroid, bounding_box, estimated_area_km2, perimeter_km}
```

### Terrain Model

Extends terrain analysis with ruggedness index and slope classification.

```python
model = await svc.terrain_model(coords)
# {model_id, ruggedness_index, slope_class: flat|undulating|mountainous}
```

### Shadow Analysis

Estimates solar-driven shadow and occlusion coverage at a given timestamp.

```python
shadow = await svc.shadow_analysis(coords, "2026-06-11T08:00:00Z")
# {solar_elevation_deg, occlusion_fraction, shadow_class: minimal|moderate|severe, no_coverage_warning}
```

### Infrastructure Mapping

```python
result = await svc.infrastructure_mapping(coords, "road")
# {feature_count, features}
```

### Population Density Analysis

Feature-density proxy per km² with density class.

```python
result = await svc.population_density_analysis(coords)
# {area_km2, feature_density_per_km2, density_class: low|medium|high}
```

### Movement Tracking

```python
result = await svc.movement_tracking("target-1", {"start": "2026-01-01", "end": "2026-06-01"})
# {observation_count, positions, last_seen}
```

### Trajectory Reconstruction

Kinematic trajectory with heading vectors and predicted next position.

```python
track = await svc.trajectory_reconstruction("target-1", {"start": "2026-01-01", "end": "2026-06-01"})
# {track_segments: [{heading_deg, velocity_proxy}], predicted_next_position}
```

### Movement Pattern

Regularity classification over a tracking window.

```python
pattern = await svc.movement_pattern("target-1", {"start": "2026-01-01", "end": "2026-06-01"})
# {regularity: regular|irregular|insufficient_data}
```

### Pattern of Life

Baseline model from historical observations for anomaly scoring.

```python
pol = await svc.pattern_of_life("target-1", build_days=90)
# {baseline: {observation_frequency_per_day, mean_accuracy, dwell_score, activity_centre}}
```

### Activity Zone

Identifies primary and secondary activity clusters for a tracked target.

```python
zones = await svc.activity_zone("target-1", radius_km=10.0)
# {zones: [{zone_id, centre, radius_km, activity_level}]}
```

### Hot Zone Detection

Clusters geospatial features into spatial hot-zones ranked by threat score.

```python
hot = await svc.detect_hot_zones(radius_km=5.0, min_samples=3)
# {hot_zones: [{zone_id, centre, member_count, mean_confidence, threat_score}]}
```

### Confidence Chain

Full provenance chain from change through feature, observation, and source.

```python
chain = await svc.confidence_chain("chg-1")
# {chain: [{stage, id, confidence}], composite_confidence}
```

### Source Fusion

Multi-observation fusion weighted by resolution class.

```python
fusion = await svc.source_fusion("area-1", window_seconds=3600)
# {contributing_observations, fused_confidence, provenance}
```

### Tile Coverage Map

Grid-based coverage tracking with gap detection per area.

```python
tiles = await svc.tile_coverage_map("area-1", tile_size_km=10.0)
# {total_tiles, covered_tiles, uncovered_tiles, tiles: [{tile_id, centre, covered, last_collection_at}]}
```

### Geofence Operations

```python
# Register a zone
await svc.register_geofence("zone-1", lat=-0.9, lon=37.0, radius_km=5.0)

# Update a target's position
await svc.update_target_position("target-1", lat=-0.91, lon=37.01)

# Check if target is inside zone
alert = await svc.geofence_alert("target-1", "zone-1")
# {inside_zone: True|False, distance_km, alert}
```

### Route Analysis

```python
route = await svc.route_analysis(
    origin={"lat": -0.9, "lon": 37.0},
    destination={"lat": -1.5, "lon": 37.8},
    avoidance_zones=["zone-1"],
)
# {direct_distance_km, zone_conflicts, route_clear}
```

### Satellite Schedule

```python
schedule = await svc.satellite_schedule("area-1", frequency="daily")
# {schedule_id, revisit_days, status}
```

### GEOINT Report

```python
report = await svc.geoint_report_generate("secret", "area-1")
# {assessment_count, high_confidence_feature_count, critical_change_count}
```

### Analytics Dashboard

```python
analytics = await svc.geoint_analytics()
# {area_count, source_count, observation_count, geofence_count, tracked_targets}
```

---

## Governance Rules

Operations are blocked by `PermissionError` with descriptive reason codes when:

| Rule | Reason Code |
|------|-------------|
| No tenant context | `tenant_context_required` |
| Missing lawful authority | `lawful_authority_required` |
| Authority/area mismatch | `area_authority_mismatch` |
| Authority/source mismatch | `source_authority_mismatch` |
| Harmful scope without approval | `targeting_or_harmful_scope_denied` |
| No event stream declared | `bytewax_event_stream_required` |

---

## Agent Registration

```python
agent = svc.register_geoint_agent(
    "agent-1", "tenant-a",
    name="imagery-analyst",
    runtime="claude_code",
    role="analyst",
    scope="imagery_analysis",
)

# Validate an agent action before executing it
svc.validate_agent_action(
    tenant_id="tenant-a",
    privileged_scope=True,
    human_approval_recorded=True,
    targeting_or_harmful_scope=False,
)
```

---

## Dashboard Summary

```python
summary = svc.dashboard_summary("tenant-a")
# {authority_count, area_count, source_count, collection_plan_count,
#  observation_count, feature_count, change_count, assessment_count,
#  dissemination_count, review_count, agent_count, audit_event_count}
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-geoint/dashboard` | `intel_geoint:view` | Overview |
| `/intel-geoint/authorities` | `intel_geoint:authorities` | Governance |
| `/intel-geoint/areas` | `intel_geoint:areas` | Planning |
| `/intel-geoint/sources` | `intel_geoint:sources` | Collection |
| `/intel-geoint/collection-plans` | `intel_geoint:collection` | Collection |
| `/intel-geoint/observations` | `intel_geoint:observations` | Processing |
| `/intel-geoint/features` | `intel_geoint:features` | Analysis |
| `/intel-geoint/changes` | `intel_geoint:changes` | Analysis |

---

## Production Boundaries

The following stay behind adapters and are not implemented in this service:

- Live satellite/aerial tasking and sensor control
- Weapon targeting or any harmful operational planning
- Map tile rendering and GIS engine execution
- Large imagery storage and computer vision extraction
- Geocoding and routing engines
- Dissemination delivery pipelines
- GraphRAG projection
- Durable Bytewax topology execution

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Rule definitions and contract introspection
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised improvements roadmap
- `README.md` — Quick reference
