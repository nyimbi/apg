# Route Optimisation — World-Class Improvement Roadmap

**Capability**: `transport_rou` | **Domain**: transport | **Author**: Nyimbi Odero
**Date**: 2026-06-11 | **Copyright**: © 2025 Datacraft

---

## 1. OR-Tools VRP Solver Integration

Replace the nearest-neighbour heuristic with Google OR-Tools `RoutingModel` to solve
Vehicle Routing Problems (VRP) optimally.  The current NN heuristic produces tours
typically 20–30% longer than optimal on random instances.  OR-Tools supports
heterogeneous fleet capacities, time windows, and penalty-based soft constraints
natively.  Expose `solver_time_limit_seconds` per tenant so latency-sensitive
workloads can trade optimality for speed.

**Impact**: 15–30% reduction in total fleet kilometres.  Directly improves fuel cost
and CO2 KPIs reported to `transport_rep`.

---

## 2. Real-Time Traffic Adapter Pool

Current traffic integration records events but does not poll live APIs.  Build an
`AsyncTrafficAdapter` ABC with concrete implementations for HERE Maps, Google Maps
Traffic, TomTom Flow, and OpenStreetMap/Overpass.  Each adapter exposes a
`async def poll_incidents(bbox) -> list[TrafficIncident]` contract.  A background
asyncio task refreshes the pool every 60 s and publishes to the `apg.transport.traffic`
event stream.

**Impact**: Dynamic rerouting becomes event-driven rather than manually triggered,
reducing on-road delay by the difference between incident detection and driver
notification — typically 5–15 min.

---

## 3. Multi-Depot VRP Support

Most enterprise fleets operate from several depots.  Introduce `Depot` as a
first-class model and extend `optimise_route` to assign stops to the nearest feasible
depot before solving each sub-fleet independently.  Savings function selects depot
assignments using Clarke-Wright.

**Impact**: Eliminates artificial single-depot constraints that inflate route lengths
by 8–40% for geographically dispersed fleets.

---

## 4. Stochastic ETA with Confidence Intervals

Replace deterministic ETA (distance ÷ average speed) with a stochastic model that
draws speed distributions from historical GPS trace data.  Use a Gaussian Process or
simple quantile regression over (hour_of_day, day_of_week, road_class) features.
Return `eta_p50_minutes`, `eta_p90_minutes`, and `eta_p95_minutes` so downstream
SLA commitments can be set at the correct percentile.

**Impact**: Eliminates systematic ETA under-estimation that leads to SLA breaches and
customer dissatisfaction in last-mile delivery.

---

## 5. Carbon Budget Constraint Engine

Expose `max_co2_kg` as a hard route constraint.  The solver iterates over mode mixes
until the carbon budget is satisfied or flags infeasibility.  Integrate with the
`intel_esg` capability to record actuals vs. budget and feed scope-3 reporting.

**Impact**: Enables enterprises operating under Science Based Targets (SBTi) to
enforce fleet decarbonisation without manual spreadsheet audits.

---

## 6. Driver Fatigue & HOS Compliance

Integrate Hours-of-Service (HOS) rules (EU Regulation 561/2006, US FMCSA) as a
constraint layer.  Insert mandatory rest breaks (45-min break after 4.5 h driving)
and daily/weekly duty limits automatically.  Expose `DriverHOSProfile` with
`accumulated_driving_minutes` and `last_break_at` fields so the solver can position
rest stops optimally.

**Impact**: Eliminates manual HOS planning, reduces legal risk, and can reduce
overall route time by 5–10% by positioning breaks at depots rather than roadside.

---

## 7. Predictive Load Balancing Across Fleet

Use historical delivery density heat-maps (stored per-tenant) to pre-cluster stops
into balanced work zones before optimisation runs.  A k-means clustering step (k =
vehicle count) partitions stops geographically; each cluster is then solved
independently.  Rebalance zones when cumulative loads deviate more than 15% from
the fleet average.

**Impact**: Reduces overtime for overloaded drivers and idle time for underloaded
ones, cutting labour cost by an estimated 8–12%.

---

## 8. Turn-Restriction & Road-Attribute Graph

The current haversine distance assumes straight-line travel.  Integrate an
OpenStreetMap road network graph (via `osmnx` + `networkx`) so that routing honours
turn restrictions, one-way streets, weight limits, and height clearances.  Cache the
graph per bounding-box tile in Redis with a 24-hour TTL.

**Impact**: Produces routes that vehicles can actually follow, removing infeasible
paths that cause driver detours and GPS recalculation events.

---

## 9. Priority Delivery Escalation

Introduce `delivery_priority: urgent | high | standard | low` on `RouteStop`.
Re-sequence stops dynamically when an urgent stop is added post-dispatch: insert it
at the minimum-detour position within the remaining tour without violating other time
windows.  Emit a `priority_stop_inserted` event so the driver app receives a push
notification.

**Impact**: Enables same-day upgrade requests without full re-optimisation, preserving
most of the original tour structure.

---

## 10. Persistent Route History & Replay

Implement an event-sourced route history store in PostgreSQL.  Every mutation
(plan, reroute, stop-complete) is appended as an immutable event row.  A
`replay_route_events(route_id, until_timestamp)` function reconstructs any
historical state.  Supports incident post-mortems, SLA dispute resolution, and
training data generation for ML models.

**Impact**: Provides the audit trail required by ISO 9001 logistics quality standards
and removes dependence on in-memory state that is lost on service restart.

---

## 11. Fleet Utilisation Optimisation

Track vehicle utilisation (actual load ÷ capacity, actual time driven ÷ shift hours)
per route.  Surface `underutilised_vehicles` and `overloaded_vehicles` lists in
`fleet_route_summary`.  Recommend consolidation when average utilisation falls below
a configurable threshold (default 70%).  Feed recommendations to `transport_flt`.

**Impact**: Consolidating under-utilised routes by 20% can reduce fleet operating
cost by a comparable margin, typically the largest single efficiency lever in
last-mile logistics.

---

## 12. Live Map Streaming via Server-Sent Events

Expose a `/transport-route/routes/<route_id>/stream` SSE endpoint that pushes
incremental stop-completion and ETA-update events to the browser dashboard.  Use
asyncio queues per route to fan out to multiple connected clients without polling.

**Impact**: Eliminates the 30-second polling loop in the current dashboard view,
reducing server load and improving dispatcher situational awareness latency from
~30 s to sub-second.

---

## 13. LLM-Powered Natural Language Route Briefing

Integrate a locally hosted Ollama model (e.g., `llama3.2`) to generate a concise
plain-English route briefing from the structured route guide.  Include traffic
conditions, special instructions, and ETA to each stop.  The briefing is returned by
`driver_route_guide` when `format="nl"` is requested.

**Impact**: Reduces time dispatchers spend on phone briefings and enables
voice-to-driver delivery of instructions via TTS without bespoke template
maintenance.

---

## 14. Geospatial Clustering for Batch Optimisation

When `bulk_plan_routes` receives more than 50 requests, use DBSCAN spatial
clustering to group requests by origin geography before solving.  Requests within
the same cluster share a graph tile cache hit, reducing solver overhead by ~40%.
Return a `cluster_id` per result so callers can group billing or reporting by zone.

**Impact**: Makes bulk optimisation tractable at scale (1 000+ routes/minute) without
requiring distributed infrastructure.

---

## 15. Webhook & Event-Bus Outbound Integration

Replace the stub `_audit` method with a full outbound event publisher that supports:
- PostgreSQL NOTIFY / LISTEN (low-latency in-process)
- Bytewax stream (`apg.transport.route.lifecycle`)
- Configurable webhook URL per tenant (HMAC-signed POST)
- APG `ntfy` capability push notifications for reroute and ETA breach events

Expose a `RouteEventBus` interface so consumers can subscribe with
`async def on_event(event: RouteEvent)` callbacks without coupling to a specific
broker.

**Impact**: Decouples transport_rou from downstream consumers (transport_dis,
transport_del, intel_esg) and enables reactive architectures without polling.
