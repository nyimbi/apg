# APG Geospatial Intelligence

`intel_geoint` is the APG package-backed capability for governed geospatial
intelligence applications. It composes authorities, areas of interest,
imagery/geospatial sources, collection plans, observations, features, change
detections, assessments, dissemination, reviews, Bytewax lifecycle metadata,
UI/view models, visual theming, and provider-neutral AI-agent automation.

## What It Provides

- Lawful authority workflow with scope, approver, expiry, classification, and
  evidence.
- Area-of-interest registry with geometry references, owner, authority, and
  evidence.
- Imagery/geospatial source registry with source type, sensor type, resolution
  class, owner, authority, and evidence.
- Collection plan workflow with authority/source/area matching, retention,
  approval, and evidence.
- Observation, feature, change, assessment, dissemination, and review workflows.
- Deterministic rule guardrails enforced before service state changes.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.intel.geoint.lifecycle`.
- Multi-spectral band analysis (NDVI, NDWI, NBR, thermal anomaly).
- Persistent trajectory reconstruction with kinematic modelling.
- Spatial hot-zone detection via proximity clustering.
- Change velocity and trend forecasting per area of interest.
- Confidence provenance chain across the full workflow.
- Multi-source observation fusion weighted by sensor resolution class.
- Pattern-of-life baseline construction per tracked target.
- Tile-based coverage map with gap detection.
- Shadow and occlusion modelling from solar elevation angle.

## Use The Service

```python
from capabilities.intel.geoint import GeospatialIntelligenceService

service = GeospatialIntelligenceService()
authority = service.record_authority(
    "auth-1",
    "tenant-a",
    "mission_order",
    "scope://mission",
    "secret",
    "approver-1",
    "2026-12-31",
    "evidence://authority",
)
area = service.record_area(
    "area-1",
    "tenant-a",
    "Port corridor",
    "geojson://area-1",
    "secret",
    "owner-1",
    authority["id"],
    "evidence://area",
)
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `lawful_authority_required`,
`area_authority_mismatch`, `source_authority_mismatch`,
`targeting_or_harmful_scope_denied`, or `bytewax_event_stream_required`.

## Async Analysis Methods

```python
import asyncio

async def run():
    svc = GeospatialIntelligenceService(tenant_id="tenant-a")

    # Multi-spectral analysis
    bands = await svc.multispectral_band_analysis("obs-1", ["visible", "nir", "swir"])
    print(bands["band_indices"])  # {"ndvi": 0.23, "ndwi": -0.11, "nbr": 0.08}

    # Trajectory reconstruction
    track = await svc.trajectory_reconstruction(
        "target-1",
        {"start": "2026-01-01T00:00:00Z", "end": "2026-06-01T00:00:00Z"},
    )
    print(track["predicted_next_position"])

    # Hot-zone detection
    zones = await svc.detect_hot_zones(radius_km=5.0, min_samples=3)
    print(zones["hot_zones"])

    # Change velocity
    vel = await svc.change_velocity("area-1", window_days=30)
    print(vel["trend"])  # accelerating / stable / decelerating

    # Confidence chain
    chain = await svc.confidence_chain("change-1")
    print(chain["composite_confidence"])

    # Source fusion
    fusion = await svc.source_fusion("area-1", window_seconds=3600)
    print(fusion["fused_confidence"])

    # Pattern of life
    pol = await svc.pattern_of_life("target-1", build_days=90)
    print(pol["baseline"])

    # Tile coverage map
    tiles = await svc.tile_coverage_map("area-1", tile_size_km=10.0)
    print(tiles["uncovered_tiles"])

    # Shadow analysis
    shadow = await svc.shadow_analysis(
        [{"lat": -1.0, "lon": 36.8}, {"lat": -1.1, "lon": 36.9}, {"lat": -0.9, "lon": 37.0}],
        "2026-06-11T08:00:00Z",
    )
    print(shadow["shadow_class"])

asyncio.run(run())
```

## Compose In Generated Apps

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Verify Locally

```bash
./.venv/bin/pytest -q capabilities/intel/geoint/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/geoint/app.py
./.venv/bin/apg capabilities inspect intel_geoint --json
./.venv/bin/apg capabilities publish-plan capabilities/intel/geoint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/geoint --json
```

## Production Boundaries

Live satellite/aerial tasking, sensor control, weapon targeting, harmful
operational planning, map tile rendering, GIS engines, large imagery storage,
computer vision extraction, geocoding, routing, dissemination delivery, GraphRAG
projection, and durable Bytewax topology execution stay behind adapters.

---

## World-Class Enhancements (v2.0)

- **I1.** GEOINT Capability - World Class Improvements
- **I2.** Multi-Spectral Band Analysis Pipeline
- **I3.** Persistent Trajectory Reconstruction
- **I4.** Spatial Clustering and Hot-Zone Detection
- **I5.** Event-Driven Geofence Subscription Model
- **I6.** Federated Source Fusion
- **I7.** Temporal Change Velocity
- **I8.** Shadow and Occlusion Modelling
- **I9.** Coordinate Reference System (CRS) Normalisation
- **I10.** Pattern-of-Life Baseline Construction
- **I11.** Tile-Based Coverage Tracking
- **I12.** Confidence Propagation Through the Workflow Chain
- **I13.** Cross-Tenant Federation and Compartmented Sharing
- **I14.** Satellite Revisit Optimisation
- **I15.** Immutable Audit Log with Tamper-Evidence

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
