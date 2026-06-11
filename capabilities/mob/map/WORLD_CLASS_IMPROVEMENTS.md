# Mobile Maps (mob_map) — World-Class Improvement Roadmap

**Capability ID**: `mob_map` | **Domain**: `mob` | **Author**: Nyimbi Odero | **© 2025 Datacraft**

This document catalogues 15 targeted improvements that would elevate the Mobile Maps capability
from a solid lifecycle-management layer to a world-class offline-maps and navigation platform.

---

## 1. Vector Tile Caching with Differential Sync

**Current gap**: Maps tile downloads are all-or-nothing bulk operations.

**Improvement**: Implement a differential tile-pack mechanism — compute a Merkle tree over cached
tile regions, transmit only the changed leaves, and version each tile-pack with a content hash.
Reduces bandwidth by 60–90 % for incremental map updates. Pairs with the existing `offline_sync`
infrastructure to share compression and encryption pipelines.

**Key types**: `TileRegionSpec`, `TilePackManifest`, `TileDeltaResult`

---

## 2. Turn-by-Turn Navigation Engine (OSRM / Valhalla adapter)

**Current gap**: No routing layer; the capability stores POI data but cannot path-find between them.

**Improvement**: Add an async `RouteService` wrapping OSRM/Valhalla HTTP APIs for route
calculation, with a local fallback that runs the Contraction Hierarchies algorithm on a
pre-downloaded graph segment. Returns `RouteResponse` with geometry (GeoJSON `LineString`),
manoeuvre list, distance, duration, and traffic-aware ETA.

**Key types**: `RouteRequest`, `RouteResponse`, `ManoeuvreStep`, `TrafficSegment`

---

## 3. Geofence Lifecycle Management

**Current gap**: No spatial trigger mechanism for location-based notifications or workflow automation.

**Improvement**: Define circular and polygon geofences stored in PostGIS. Evaluate entry/exit
events server-side via a streaming pipeline (Bytewax) or on-device with a lightweight point-in-polygon
check. Emit `geofence_entered` / `geofence_exited` CloudEvents, composable with `ntfy` for
location-triggered push notifications.

**Key types**: `GeofenceCreate`, `GeofenceResponse`, `GeofenceEvent`

---

## 4. POI Semantic Search with Embedding Index

**Current gap**: POI lookup is exact-match by category; no fuzzy or semantic search.

**Improvement**: Embed POI name + description using a locally-hosted Ollama embedding model
(e.g. `nomic-embed-text`). Persist vectors in pgvector. Expose `search_pois_semantic` returning
ranked results with cosine similarity scores. Degrades gracefully to trigram similarity when the
vector index is cold.

**Key types**: `POISearchRequest`, `POISearchResult`, `EmbeddingVector`

---

## 5. Offline Route Pre-computation & Packaging

**Current gap**: Routes require connectivity; no offline package format for saved trips.

**Improvement**: Introduce a `TripPackage` that bundles the route geometry, required map tiles
(referenced by tile-pack ID), turn-by-turn instructions, and relevant POI stops into a single
compressed archive. Devices download one package before a trip and navigate fully offline.
Uses zstd compression; AES-256-GCM envelope if the tenant has encryption required.

**Key types**: `TripPackageCreate`, `TripPackageResponse`, `TripPackageStatus`

---

## 6. Real-Time Location Sharing (Presence Layer)

**Current gap**: No multi-user location awareness; critical for field-force and logistics apps.

**Improvement**: Add a presence channel (WebSocket or SSE endpoint) where device agents publish
their current location at a configurable interval (1 – 60 s). Positions are stored in a
time-series table with configurable TTL. Expose a `list_nearby_devices` query parameterised by
bounding box or radius. Rate-limit writes via the existing token-bucket enforcement.

**Key types**: `LocationUpdate`, `PresenceSession`, `NearbyDeviceResult`

---

## 7. Elevation-Aware Route Profiles

**Current gap**: Route profiles (walking, cycling, driving) ignore terrain elevation.

**Improvement**: Integrate SRTM or Copernicus DEM data as an offline elevation dataset.
Annotate route segments with grade (%), add elevation profile charts to `RouteResponse`, and
offer a `min_elevation_gain` or `scenic_route` optimisation objective. Particularly valuable for
hiking and cycling verticals.

**Key types**: `ElevationProfile`, `RouteSegmentElevation`, `ElevationTileCache`

---

## 8. Map Style Engine (Tenant-Brandable Cartography)

**Current gap**: Map appearance is fixed; tenants cannot customise for white-label deployments.

**Improvement**: Store MapLibre GL style JSON per tenant in the `conf` capability. Allow tenants
to override colour palettes, font stacks, layer visibility, and icon sprites via a structured
`MapStyleConfig` model. Render previews server-side with headless Mapbox Static Images API or a
local MapLibre Node renderer, returning a thumbnail URL for the dashboard.

**Key types**: `MapStyleConfig`, `MapStylePreview`, `MapTheme`

---

## 9. Fleet Tracking & ETA Broadcasting

**Current gap**: No aggregated vehicle/asset tracking or ETA computation pipeline.

**Improvement**: Model a `FleetVehicle` entity linked to `MobileApp` + `device_id`. Ingest
location telemetry, compute live ETA to a set of `WaypointStop` entities using the routing
engine, and broadcast updates via the `mqeb` event bus. Expose a `get_fleet_eta` query returning
ordered stops with predicted arrival windows and delay confidence intervals.

**Key types**: `FleetVehicle`, `WaypointStop`, `ETABroadcast`

---

## 10. Spatial Analytics Aggregations

**Current gap**: Analytics are event-count tables; no geospatial aggregation (heatmaps, corridors).

**Improvement**: Add PostGIS-backed aggregation queries: heatmap density grid (`ST_SquareGrid`),
popular corridors (`ST_ClusterDBSCAN` on route traces), and dwell-time polygons. Expose as
GeoJSON `FeatureCollection` responses consumable directly by MapLibre or Leaflet frontends.

**Key types**: `HeatmapRequest`, `HeatmapResponse`, `DwellCluster`

---

## 11. Map Data Freshness & Auto-Update Scheduling

**Current gap**: Offline tile packs go stale; no automated refresh pipeline.

**Improvement**: Track `tile_pack_age_days` per region. Register a scheduled job (via CronCreate)
that compares the server tile generation timestamp against the device's cached manifest hash.
Trigger background downloads for stale regions when the device is on Wi-Fi and charging.
Configurable staleness threshold per tenant (default 7 days).

**Key types**: `TilePackFreshnessScan`, `TileUpdateSchedule`, `RefreshPolicy`

---

## 12. Multi-Modal Transit Integration

**Current gap**: Navigation supports only single-mode routing (driving/walking/cycling).

**Improvement**: Parse GTFS (General Transit Feed Specification) feeds for local transit agencies.
Persist `TransitRoute`, `TransitStop`, and `ServiceCalendar` entities. Extend the routing engine
to include transit legs in intermodal `RouteResponse` objects, specifying boarding times,
headways, and fare estimates alongside walking and driving segments.

**Key types**: `GTFSIngestion`, `TransitRoute`, `IntermodalLeg`

---

## 13. Privacy-Preserving Location Anonymisation

**Current gap**: Raw GPS tracks are stored verbatim; GDPR/LGPD exposure for PII-sensitive fleets.

**Improvement**: Apply k-anonymity and spatial generalisation before persistence: snap positions
to the nearest road segment (map matching via OSRM nearest API), quantise timestamps to configurable
granularity (e.g. 30 s buckets), and redact home/work zones defined in user privacy profiles.
Configurable per tenant; non-destructive (raw data optionally held in an encrypted cold store).

**Key types**: `LocationAnonymisationConfig`, `AnonymisedTrack`, `PrivacyZone`

---

## 14. Indoor Mapping & Venue Navigation

**Current gap**: Maps are outdoor-only; no floor-plan support for malls, airports, hospitals.

**Improvement**: Ingest IMDF (Indoor Mapping Data Format) venue packages. Persist `Venue`,
`Floor`, `Unit`, and `Opening` entities in PostGIS. Generate indoor routing graphs between
`Opening` entities on each floor and across elevators/stairs. Expose `navigate_indoor` returning
step-by-step floor-aware directions with floor transitions.

**Key types**: `VenuePackage`, `IndoorRoute`, `FloorTransition`

---

## 15. Predictive Caching via ML Trip Prediction

**Current gap**: Tile pre-fetching is demand-driven; no anticipatory caching.

**Improvement**: Train a lightweight trip-prediction model (Markov chain or small LSTM via Ollama
served endpoint) on historical navigation sessions per user. Pre-fetch tile packs for the top-3
predicted next destinations when the device connects to Wi-Fi. Use a UCB1 bandit policy to
balance exploration (new routes) against exploitation (confirmed commute patterns).

**Key types**: `TripPrediction`, `PrefetchJob`, `BanditState`

---

*All improvements follow the APG code standards: async-first, Pydantic v2 models, UUID7 IDs,
tenant-scoped enforcement, PostGIS spatial storage, and local Ollama models for any ML components.*
