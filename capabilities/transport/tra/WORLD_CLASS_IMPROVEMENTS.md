# Asset Tracking — World Class Improvements

**Capability**: `transport_tra` | **Version**: 1.0.0 → 2.0.0

The following 15 improvements would elevate this capability from functional to world-class.

---

## 1. Streaming Telemetry Ingest Pipeline

**Problem**: Location updates are ingested one-at-a-time via synchronous calls.
**Improvement**: Add a `ingest_telemetry_stream()` coroutine that accepts an async iterator of raw GPS frames, applies deduplication (same position within 10 m within 30 s), batches them through the Bytewax pipeline, and emits cloud events. This reduces per-message overhead by ~60× and enables sub-second map refresh.

---

## 2. Predictive Route Deviation Detection

**Problem**: The system only knows whether an asset left a geofence; it cannot warn *before* deviation occurs.
**Improvement**: Add `predict_route_deviation()` which fits a Kalman filter over the last N position pings to project the next heading. If the projected trajectory diverges from the planned route corridor by > threshold_km within the next T minutes, emit a pre-emptive `route_deviation_imminent` alert. Saves ~8 minutes of reaction time on average.

---

## 3. Multi-Leg Journey Analytics

**Problem**: `tracking_report()` produces flat counts; it cannot represent a journey with waypoints, stops, and legs.
**Improvement**: Add `journey_analytics()` which segments a location history into discrete legs separated by stops (speed < 2 km/h for > idle_threshold_minutes). Returns per-leg distance, duration, average speed, and stop dwell time. Enables SLA breach detection per delivery leg.

---

## 4. Dwell Time & Detention Tracking

**Problem**: Container detention fees are tracked externally; no automated alert when dwell time exceeds free time.
**Improvement**: Add `container_dwell_alert()` that maintains entry timestamps for geofenced depots/ports and fires a `detention_imminent` alert N hours before the free-time window closes. Integrates with `transport_car` cargo custody chain for automated detention invoice generation.

---

## 5. Harsh Event Detection (Acceleration / Braking)

**Problem**: `speeding` is a supported alert type but there is no method that actually computes it from GPS deltas.
**Improvement**: Add `detect_harsh_events()` which computes speed delta between consecutive location pings and classifies events: `harsh_braking` (−0.3 g), `harsh_acceleration` (+0.3 g), `speeding` (> posted limit). Required for fleet safety scoring and insurance telematics.

---

## 6. Multi-Standard Cold Chain Certificate Generation

**Problem**: `record_cold_chain()` writes a record but never produces the compliance certificate required by regulators.
**Improvement**: Add `generate_cold_chain_certificate()` that aggregates all readings for an asset over a shipment period, validates against the selected standard, and produces a signed PDF-equivalent metadata blob with HACCP/GDP/ATP deviation summary. Feeds `comp` capability directly.

---

## 7. Real-Time Geofence Dwell Analytics

**Problem**: Entry/exit events are counted but dwell duration is never computed.
**Improvement**: Augment `GeofenceDwellRecord` (new model) to store entry timestamp, exit timestamp, and computed dwell minutes per asset per geofence. Add `geofence_dwell_report()` returning average, max, and percentile dwell times. Enables slot optimisation at depots and loading bays.

---

## 8. Asset Clustering for Map Density Control

**Problem**: `fleet_map_view()` returns every asset position; rendering 10 000 points crashes browser clients.
**Improvement**: Add `fleet_map_clusters()` which groups nearby assets into geohash-6 cells (~1.2 km²), returns centroid + count per cell, and expands to individual features only when zoom ≥ threshold. Reduces payload by 95% at national-scale views.

---

## 9. Offline Telemetry Buffer & Replay

**Problem**: Assets in tunnels, mines, or rural areas lose connectivity for hours; buffered pings are silently dropped.
**Improvement**: Add `replay_buffered_telemetry()` which accepts an ordered list of timestamped pings, validates temporal ordering, deduplicates against already-stored updates, and applies them to historical state. Emits a `telemetry_replay_complete` event with gap statistics.

---

## 10. Fleet-Wide Utilisation Benchmarking

**Problem**: `asset_utilisation()` reports a single asset; there is no cross-fleet comparison.
**Improvement**: Add `fleet_utilisation_benchmark()` returning utilisation percentiles (p25/p50/p75/p95) across all active assets, identifies the bottom quartile for redeployment, and computes idle cost at a configurable cost-per-hour rate. Produces a prioritised action list.

---

## 11. Alert Suppression & Deduplication

**Problem**: Aggressive GPS polling causes repeated entry/exit events for assets near geofence boundaries (the "boundary oscillation" problem), flooding alert channels.
**Improvement**: Add configurable hysteresis: an asset must be `inside` for > `min_dwell_minutes` before an entry alert fires, and `outside` for > `min_dwell_minutes` before an exit alert fires. This requires a `GeofenceAssetState` store and the `update_location()` method to carry state across invocations.

---

## 12. Anomaly-Based Tamper Detection

**Problem**: Tamper is only detected via hardware flag (`tamper_detected=True`); sophisticated attacks that simply power-cycle the device are missed.
**Improvement**: Add `detect_location_anomaly()` which uses a rolling speed/distance consistency check: if the new position implies the asset moved > max_plausible_speed_kmh since the last ping, flag it as `position_jump_anomaly` and fire a medium-severity alert. Catches GPS spoofing and cloned trackers.

---

## 13. Audit Log Streaming to Immutable Store

**Problem**: `self.audit_events` is an in-memory list; it is lost on restart and can be mutated in-process.
**Improvement**: Refactor `_audit()` to publish to an append-only event store (PostgreSQL `INSERT`-only table with a trigger blocking `UPDATE`/`DELETE`) via the `audl` capability. Add `audit_log_query()` supporting time-range, event-type, and actor filters. Satisfies SOC 2 audit trail requirements.

---

## 14. Multi-Tenant Isolation via Row-Level Security

**Problem**: The `_key(tenant_id, item_id)` tuple pattern works in memory but the PostgreSQL schema has no RLS policies, making a SQL injection on any endpoint a cross-tenant data leak.
**Improvement**: Add Alembic migration enabling `ALTER TABLE ... ENABLE ROW LEVEL SECURITY` with `USING (tenant_id = current_setting('app.tenant_id'))` on every tracking table. Set `app.tenant_id` at the start of each request context. Eliminates the entire class of cross-tenant leakage bugs.

---

## 15. Composable Tracking Pipeline DSL

**Problem**: Consumers must call individual service methods and wire events manually; composing a "track + geofence + cold chain + alert" pipeline requires ~50 lines of boilerplate per use case.
**Improvement**: Introduce a declarative pipeline builder:
```python
pipeline = (
    TrackingPipeline(svc, tenant_id="acme")
    .for_asset("VEH-001")
    .with_geofence("nairobi_depot", alert_on="both")
    .with_cold_chain(standard="haccp")
    .on_breach(notify="ops-team@acme.co")
    .build()
)
await pipeline.run(telemetry_stream)
```
This removes the O(N) integration surface and makes the capability usable by domain operators without Python expertise.
