# Route Optimisation — User Guide

**Capability ID**: `transport_rou` | **Domain**: `transport` | **Version**: `1.1.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

---

## Description

The Route Optimisation capability provides multi-stop route planning with time-window
enforcement, 8 optimisation objectives, dynamic traffic-triggered rerouting,
multi-modal segment planning (road, rail, sea, air), geospatial address validation,
fleet utilisation analytics, CO2 budget-constrained routing, HOS compliance
checking, and stochastic ETA prediction with confidence bands.

---

## Installation

```bash
pip install apg-transport-rou
```

---

## Quick Start

```python
from capabilities.transport.rou.service import RouteOptimisationService
import asyncio

svc = RouteOptimisationService(tenant_id="acme")

# Plan a simple route
route = svc.plan_route(
    "R001", "acme", "delivery",
    "Nairobi CBD", "Westlands",
    "TRUCK-01", "road", 3,
    12.5, 25, "minimize_cost",
)

# Optimise multi-stop
result = asyncio.run(svc.optimise_route(
    waypoints=[
        {"id": "A", "lat": -1.286, "lng": 36.817, "address": "Nairobi CBD"},
        {"id": "B", "lat": -1.290, "lng": 36.830, "address": "Westlands"},
        {"id": "C", "lat": -1.300, "lng": 36.820, "address": "Kilimani"},
    ],
    constraints={"max_stops": 10},
    objective="minimize_distance",
    vehicle_id="TRUCK-01",
))
print(result["total_distance_km"])
```

---

## Core Concepts

### Tenancy
Every entity is scoped to a `tenant_id`.  Pass it at construction time or per-method.
Cross-tenant writes are blocked by business rules.

### Route Lifecycle
```
plan_route → add_route_stop (×N) → add_constraint → [dispatch]
           → record_traffic_event → trigger_reroute (if needed)
           → mark_stop_completed (×N) → route complete
```

### Optimisation Objectives
`minimize_cost`, `minimize_distance`, `minimize_time`, `minimize_co2`,
`maximize_utilisation`, `balanced`, `priority_first`, `time_window_compliance`

### Transport Modes
`road`, `rail`, `sea`, `air`, `walk`, `bicycle`, `multimodal`

---

## API Reference

### plan_route

```python
svc.plan_route(
    route_id, tenant_id, route_type,
    origin, destination, vehicle_id,
    transport_mode="road",
    stop_count=1,
    total_distance_km=0.0,
    estimated_duration_minutes=0,
    optimisation_objective="minimize_cost",
    address_validated=True,
)
```

Creates a `Route` record.  Address must be validated; unvalidated origin/destination
raise `PermissionError("unvalidated_address_dispatch_denied")`.

---

### optimise_route (async)

```python
result = await svc.optimise_route(
    waypoints=[{"id": str, "lat": float, "lng": float, "address": str}, ...],
    constraints={"max_stops": 50, "avoid_tolls": True},
    objective="minimize_distance",
    vehicle_id="TRUCK-01",
)
```

Nearest-neighbour optimisation.  Returns `optimised_waypoints`, `segments`,
`total_distance_km`, `estimated_duration_minutes`.

---

### vehicle_routing_problem (async)

Solve a multi-depot, multi-vehicle VRP.

```python
result = await svc.vehicle_routing_problem(
    depots=[
        {"id": "D1", "lat": -1.28, "lng": 36.82, "name": "Nairobi Depot"},
        {"id": "D2", "lat": -1.40, "lng": 36.95, "name": "Thika Depot"},
    ],
    stops=[
        {"id": "S01", "lat": -1.29, "lng": 36.83, "demand_kg": 200, "address": "Stop A"},
        {"id": "S02", "lat": -1.31, "lng": 36.81, "demand_kg": 150, "address": "Stop B"},
    ],
    vehicles=[
        {"id": "V1", "capacity_kg": 500, "depot_id": "D1"},
        {"id": "V2", "capacity_kg": 300, "depot_id": "D2"},
    ],
)
print(result["total_fleet_km"])
print(result["unassigned_stop_ids"])  # stops that exceeded capacity
```

Returns per-vehicle routes with capacity utilisation percentages and a list of
`unassigned_stop_ids` when demand exceeds fleet capacity.

---

### predict_eta_with_traffic (async)

```python
eta = await svc.predict_eta_with_traffic(
    route_id="R001",
    current_location={"lat": -1.288, "lng": 36.820},
    live_delay_minutes=12,
    percentiles=[50, 90, 95],
)
print(eta["eta_estimates"]["p90_minutes"])
```

Returns p50/p90/p95 ETA estimates using a log-normal speed distribution (CV=0.25 for
road).  Use `p90` for SLA commitments to 90% on-time delivery targets.

---

### hours_of_service_check (async)

```python
result = await svc.hours_of_service_check(
    route_id="R001",
    driver_profile={
        "accumulated_driving_minutes": 200,
        "daily_driving_minutes": 360,
        "weekly_driving_minutes": 2400,
    },
    regulation="eu_561_2006",
)
if not result["compliant"]:
    print(result["violations"])
    print(f"Breaks required: {result['breaks_required']}")
```

Supported regulations: `eu_561_2006` (EU), `us_fmcsa` (US).  Violations list each
specific limit exceeded with the overage amount.

---

### carbon_budget_route (async)

```python
result = await svc.carbon_budget_route(
    origin="Mombasa Port",
    destination="Nairobi ICD",
    cargo_tonnes=20.0,
    max_co2_kg=1500.0,
    available_modes=["road", "rail"],
)
if result["feasible"]:
    print(result["recommended"]["mode"])
else:
    print("No mode satisfies the CO2 budget — consider rail intermodal")
```

Returns the fastest mode within the CO2 budget.  When no mode qualifies,
`feasible=False` and `all_options` shows CO2 for each mode so the budget can be
revised.

---

### priority_stop_insert (async)

Insert an urgent stop into a live, partially-completed route at the minimum-detour
position.

```python
result = await svc.priority_stop_insert(
    route_id="R001",
    new_stop={
        "id": "URGENT-01",
        "lat": -1.295,
        "lng": 36.825,
        "address": "Emergency Clinic",
        "time_window_start": "14:00",
        "time_window_end": "15:00",
        "service_time_minutes": 10,
    },
    priority="urgent",
)
print(f"Inserted at position {result['inserted_at_position']}, "
      f"extra distance: {result['extra_distance_km']} km")
```

Completed stops are not reordered.  Only remaining stops are evaluated for insertion
position.

---

### geospatial_cluster_stops (async)

Pre-partition a large stop set for parallel per-cluster VRP solving.

```python
result = await svc.geospatial_cluster_stops(
    stops=[{"id": "S01", "lat": -1.28, "lng": 36.82}, ...],
    n_clusters=4,
)
for cluster in result["clusters"]:
    print(cluster["cluster_id"], cluster["stop_count"])
```

Uses iterative k-means (max 20 iterations).  Pass each cluster's `stop_ids` to a
separate `vehicle_routing_problem` call for parallel solving.

---

### route_replay (async)

Reconstruct the full mutation history of a route for audit.

```python
history = await svc.route_replay(route_id="R001")
for event in history["timeline"]:
    print(event["type"], event.get("event_type"), event.get("reference_id"))
```

Useful for SLA dispute resolution, incident post-mortems, and generating ML training
data.

---

### fleet_utilisation_report (async)

```python
report = await svc.fleet_utilisation_report(
    vehicle_ids=["V01", "V02", "V03", "V04"],
)
print(report["underutilised_vehicles"])   # < 70% of fleet average
print(report["overloaded_vehicles"])      # > 130% of fleet average
```

Use the output to trigger vehicle rebalancing requests to `transport_flt`.

---

### time_window_feasibility (async)

Validate whether a stop sequence can satisfy all time windows before committing.

```python
result = await svc.time_window_feasibility(
    stops=[
        {"id": "S1", "lat": -1.28, "lng": 36.82},
        {"id": "S2", "lat": -1.29, "lng": 36.83},
    ],
    time_windows={
        "S1": {"open": "09:00", "close": "10:30"},
        "S2": {"open": "11:00", "close": "12:00"},
    },
    depot_lat=-1.27,
    depot_lng=36.81,
    start_time="08:30",
    average_speed_kmph=50.0,
    service_time_minutes=15,
)
for stop in result["stop_results"]:
    if not stop["feasible"]:
        print(f"{stop['stop_id']} late by {stop['late_by_minutes']} min")
```

---

## Traffic & Dynamic Rerouting

```python
# 1. Record a live traffic incident
svc.record_traffic_event("TE-001", "acme", "google_maps", "R001", 25, "2026-06-11T09:15:00Z", "accident")

# 2. Trigger reroute
rr = await svc.dynamic_reroute(
    current_route={"route_id": "R001", "remaining_stops": [...]},
    traffic_event={"type": "accident", "affected_segment": "A1-B1", "delay_minutes": 25},
)
print(rr["new_route_id"])
```

---

## Multi-Modal Routes

```python
mm = await svc.multi_modal_route(
    origin="Mombasa Port",
    destination="Kampala ICD",
    modes=["sea", "rail", "road"],
)
for seg in mm["segments"]:
    print(seg["transport_mode"], seg["distance_km_stub"], "km")
```

---

## Analytics

```python
# Period KPI roll-up
kpis = await svc.route_analytics("2026-Q2")

# Fleet summary
fleet = await svc.fleet_route_summary()

# Carbon optimisation comparison
co2 = await svc.route_comparison([
    {"route_id": "R1", "distance_km": 120, "duration_minutes": 90,
     "transport_mode": "road", "cost_usd": 80},
    {"route_id": "R2", "distance_km": 140, "duration_minutes": 70,
     "transport_mode": "road", "cost_usd": 100},
])
print(co2["recommended_route_id"])
```

---

## Composability

```apg
use transport_rou;
```

| Consumer | Data shared |
|----------|-------------|
| `transport_dis` | Planned routes for dispatch assignment |
| `transport_sch` | Stop time windows synced with schedules |
| `transport_tra` | Traffic feed integration (live position) |
| `transport_del` | Route completion → delivery ETA updates |
| `transport_flt` | Fleet utilisation signals → vehicle rebalancing |
| `intel_esg` | Carbon budget actuals → scope-3 reporting |

---

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `routes.max_stops_per_route` | int | 200 | Hard stop limit per route |
| `optimisation.default_objective` | str | `minimize_cost` | Solver objective |
| `optimisation.max_optimisation_seconds` | int | 30 | Solver time budget |
| `rerouting.auto_reroute_enabled` | bool | true | Auto-trigger on traffic events |
| `hos.regulation` | str | `eu_561_2006` | Default HOS regulation |
| `carbon.max_co2_kg_default` | float | none | Tenant-level CO2 cap |
| `fleet.utilisation_low_threshold_pct` | float | 70 | Under-utilisation alert threshold |
| `fleet.utilisation_high_threshold_pct` | float | 130 | Over-load alert threshold |

---

## Error Reference

| Exception | Condition |
|-----------|-----------|
| `PermissionError("unvalidated_address_dispatch_denied")` | Address not validated |
| `PermissionError("capacity_constraint_violation")` | Load exceeds capacity |
| `PermissionError("max_stops_exceeded")` | >200 stops on a single route |
| `ValueError("waypoints list is empty")` | No waypoints passed to optimise_route |
| `ValueError("rating must be 1–5")` | Invalid driver feedback rating |
| `ValueError("n_clusters must be >= 1")` | Invalid cluster count |
| `KeyError(f"Route {id} not found")` | Route does not exist for tenant |

---

## Further Reading

- `service.py` — Full business logic implementation
- `models.py` — SQLAlchemy + Pydantic data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 prioritised enhancements
- `cap_spec.md` — Formal capability specification
- `SPECIFICATION.md` — Detailed functional specification
