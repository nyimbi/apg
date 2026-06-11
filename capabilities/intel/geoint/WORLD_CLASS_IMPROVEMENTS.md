# GEOINT Capability - World Class Improvements

**Capability**: `intel_geoint` (Geospatial Intelligence)
**Version Target**: 2.0.0
**Date**: 2026-06-11

---

## 1. Multi-Spectral Band Analysis Pipeline

Current imagery analysis treats all observations as monolithic blobs. A world-class system decomposes imagery by spectral band (visible, NIR, SWIR, thermal) and runs per-band change detection. Implementing `multispectral_band_analysis(image_id, bands)` as an async method that returns per-band statistics — NDVI, NDWI, burn ratio — with confidence intervals would unlock vegetation health, water body mapping, and fire detection from the same collection.

## 2. Persistent Trajectory Reconstruction

`movement_tracking` returns raw observations with no velocity or heading computation. Trajectory reconstruction fits observed positions to a kinematic model (constant velocity, constant turn rate), interpolates gaps, extracts bearing sequences, and emits velocity vectors. This turns point-in-time sightings into a continuous track with predicted future positions and anomaly detection on deviation from baseline trajectory.

## 3. Spatial Clustering and Hot-Zone Detection

There is no spatial aggregation over features or changes. Adding a DBSCAN-based clustering method `detect_hot_zones(radius_m, min_samples)` that groups features and changes into spatial clusters, scores each cluster by severity and recency, and returns ranked hot-zones enables rapid operational prioritisation without manual area filtering.

## 4. Event-Driven Geofence Subscription Model

Geofence alerts are currently pull-based (caller must invoke `geofence_alert`). A push model where targets are subscribed to zones and an async background task emits domain events on entry/exit removes the polling burden. Integrating with the APG event bus (`apg.intel.geoint.geofence`) means downstream capabilities (alerts, reporting) react automatically.

## 5. Federated Source Fusion

Single-source observations carry inherent accuracy limits. Source fusion weights multiple simultaneous observations of the same AOI by sensor quality (resolution_class, sensor_type) and temporal proximity, producing a fused confidence score higher than any individual source. The `fuse_observations(area_id, window_seconds)` method should return a fused observation record with provenance chain.

## 6. Temporal Change Velocity

Change detections have types and severities but no rate-of-change metric. Computing change velocity — the rate at which new changes accumulate per AOI per time unit — surfaces accelerating threat environments before they reach critical severity. `change_velocity(area_id, window_days)` should return changes/day with trend (accelerating/stable/decelerating) and forecasted count.

## 7. Shadow and Occlusion Modelling

Terrain model currently computes ruggedness proxy from perimeter/area ratio. True shadow modelling requires solar elevation angle (from timestamp and geographic centroid) to compute shadows cast by terrain and structures, identifying occlusion zones where imagery cannot confirm ground truth. `shadow_analysis(area_coords, observation_timestamp)` returns no-coverage polygons.

## 8. Coordinate Reference System (CRS) Normalisation

All geometry references are opaque strings. A CRS normalisation layer that parses WKT, GeoJSON, or `lat,lon` strings and projects them into a canonical WGS84 GeoJSON representation would allow internal spatial operations (intersection, containment, proximity) to run correctly against heterogeneous source geometries. Without this, `facility_identification` can only parse naive `lat,lon` formats.

## 9. Pattern-of-Life Baseline Construction

`movement_pattern` detects regularity but has no concept of a learned baseline. Building a per-target pattern-of-life model from historical observations — typical arrival/departure times, dwell locations, recurrence periods — enables anomaly scoring: how far does today's observed behaviour deviate from the learned baseline? `build_pattern_of_life(target_id, days)` computes the baseline and `score_against_pattern(target_id)` returns deviation score.

## 10. Tile-Based Coverage Tracking

Collection plan coverage is tracked at area level only. Production systems divide AOIs into fixed-size tiles and track per-tile collection recency, cloud cover, and sensor availability. `tile_coverage_map(area_id, tile_size_km)` returns a grid of tiles annotated with last_collection_at, cloud_percentage, and coverage_gap_days, enabling gap-fill scheduling.

## 11. Confidence Propagation Through the Workflow Chain

Confidence scores exist on features and changes but are not propagated through assessments and disseminations. Each workflow stage should inherit and attenuate the upstream confidence: if a feature has 0.9 confidence, the change derived from it should be weighted by that, the assessment further weighted by analyst reliability score. `confidence_chain(change_id)` returns the full provenance tree with per-stage confidence factors.

## 12. Cross-Tenant Federation and Compartmented Sharing

All service state is keyed by `(tenant_id, item_id)`. There is no mechanism to share a classified assessment with a partner tenant under a release marking. A federated sharing protocol with release marking enforcement — `federate_assessment(assessment_id, partner_tenant_id, release_marking)` — would implement the GEOINT dissemination workflow properly, enforcing that the partner tenant's classification ceiling meets the release marking requirement before materialising the shared record.

## 13. Satellite Revisit Optimisation

`satellite_schedule` accepts a frequency string but does not model actual orbital mechanics or tasking conflicts. A revisit optimiser takes a set of AOIs ranked by collection priority, available satellite passes (as time windows), and cloud cover forecasts, then produces an optimal tasking schedule minimising collection gaps while respecting priority ordering. Even a simplified version using greedy scheduling against time windows has significant operational value.

## 14. Immutable Audit Log with Tamper-Evidence

`_audit` appends to an in-memory list with no integrity protection. For GEOINT operations subject to legal authority, the audit log must be tamper-evident. Implementing a hash-chain audit log where each entry includes `prev_hash = SHA256(prev_record)` means any post-hoc modification to the log is detectable. Persisting this chain to a write-once store (append-only PostgreSQL table with statement-level triggers) completes the integrity guarantee.

## 15. Semantic Tile Description for AI-Assisted Analysis

Current feature extraction requires human analysts to label geometry references. An LLM-assisted semantic description pipeline uses a locally hosted vision-language model (LLaVA via Ollama) to generate natural language descriptions of imagery tiles, which are then embedded and stored in a vector index. `semantic_tile_search(query, area_id)` retrieves tiles whose visual content matches the query, enabling analytic discovery without predefined feature type taxonomies.
