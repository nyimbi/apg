# Dispatch Operations — User Guide

**Capability ID**: `transport_dis` | **Domain**: `transport` | **Version**: `1.0.0`

---

## Description

The Dispatch Operations capability manages the full lifecycle of fleet dispatch: load planning, driver assignment with hours-of-service (HOS) compliance, multi-objective route optimisation, real-time GPS tracking, exception management, proof-of-delivery capture, SLA breach prediction, driver performance scoring, and backhaul load matching.

---

## Installation

```bash
pip install apg-transport-dis
```

---

## Quick Start

```python
import asyncio
from apg_transport_dis.service import DispatchOperationsService

async def main():
    svc = DispatchOperationsService(tenant_id="acme", actor_id="ops-user-1")

    # 1. Plan a load
    load = svc.plan_load(
        "LP-001", "acme", "full_truckload", "TRK-001",
        total_weight_kg=18000.0, total_volume_cbm=72.0, stop_count=4,
        optimisation_mode="cost",
    )

    # 2. Create a dispatch
    dispatch = svc.create_dispatch("DSP-001", "acme", "LP-001", "TRK-001", "DR-001", "RTE-001")

    # 3. Assign driver
    assignment = svc.assign_driver("ASN-001", "acme", "DSP-001", "DR-001", "primary", "2026-06-11T06:00:00Z", 9.0)

    # 4. Dispatch the vehicle (async)
    result = await svc.dispatch_vehicle("DSP-001", tenant_id="acme")
    print(result["dispatched_at"])

asyncio.run(main())
```

---

## Core Workflows

### Load Planning

Create a load plan manually or let the bin-packing engine allocate orders to vehicles:

```python
# Manual
load = svc.plan_load("LP-002", "acme", "less_than_truckload", "VAN-007", 3500, 14.0, 6, "balanced")

# Automated bin-packing across a fleet
result = await svc.create_load_plan(
    orders=[
        {"order_id": "O1", "weight_kg": 800, "volume_cbm": 3.2},
        {"order_id": "O2", "weight_kg": 1200, "volume_cbm": 4.8},
    ],
    vehicles_available=[
        {"vehicle_id": "VAN-001", "max_weight_kg": 3500, "max_volume_cbm": 14.0},
        {"vehicle_id": "VAN-002", "max_weight_kg": 3500, "max_volume_cbm": 14.0},
    ],
    optimisation_mode="cost",
    tenant_id="acme",
)
print(result["total_vehicles_used"], result["unallocated_order_ids"])
```

### Route Optimisation

After creating a load plan, optimise the stop sequence using the nearest-neighbour heuristic:

```python
optimised = await svc.optimise_dispatch("LP-001", tenant_id="acme")
print(f"Saving: {optimised['distance_saving_pct']}%")
print(optimised["optimised_stop_sequence"])
```

### Driver Assignment & HOS Compliance

Before assigning a driver to a long dispatch, check HOS margin:

```python
hos = await svc.predict_hos_violation("DR-001", planned_duration_minutes=480.0, tenant_id="acme")
if hos["violation_risk"]:
    print(f"HOS risk: {hos['recommendation']}")
else:
    svc.assign_driver("ASN-002", "acme", "DSP-002", "DR-001", "primary", "2026-06-11T07:00:00Z", 9.0)
```

Check aggregate compliance retrospectively:

```python
compliance = await svc.compliance_hours_check("DR-001", tenant_id="acme")
print(f"Compliant: {compliance['compliant']}, hours remaining: {compliance['hours_remaining']}")
```

### Live Dispatch

```python
# Formally dispatch (transitions status, notifies driver)
await svc.dispatch_vehicle("DSP-001", tenant_id="acme")

# Ingest GPS ping from telematics
await svc.real_time_tracking_update(
    "TRK-001",
    gps={"lat": -1.2864, "lng": 36.8172},
    speed=62.5,
    status="moving",
    eta_minutes=47,
    tenant_id="acme",
)

# Update ETA only
await svc.update_eta("DSP-001", 55, tenant_id="acme")
```

### Fleet Position Snapshot

Pull last-known positions for all active vehicles — for map rendering or ops dashboards:

```python
snapshot = await svc.fleet_position_snapshot(tenant_id="acme")
for pos in snapshot["positions"]:
    print(pos["vehicle_id"], pos["lat"], pos["lng"], pos["eta_minutes"])
```

### In-Flight Driver Reassignment

Swap a driver on a live dispatch atomically. Triggers driver notification and ETA recalculation:

```python
result = await svc.reassign_driver_in_flight(
    "DSP-001",
    new_driver_id="DR-002",
    reason="DR-001 reported medical issue",
    new_hours_available=10.0,
    tenant_id="acme",
)
print(result["new_driver_id"], result["reassigned_at"])
```

### Exception Management

```python
# Raise an exception
exc = await svc.exception_management(
    "DSP-001", "vehicle_breakdown", tenant_id="acme"
)
print(exc["escalation_level"])

# Resolve it
svc.resolve_exception(exc["exception"]["id"], "acme", "2026-06-11T10:00:00Z", "Replacement truck dispatched")
```

### SLA Breach Prediction

Proactively scan all active dispatches for SLA risk. Dispatches above threshold get an auto-raised `time_window_missed` exception for operator review:

```python
scan = await svc.predict_sla_breach(breach_probability_threshold=0.70, tenant_id="acme")
print(f"At-risk: {scan['at_risk_count']}, escalated: {scan['escalated_count']}")
for d in scan["at_risk"]:
    print(d["dispatch_id"], d["breach_probability"])
```

### Proof of Delivery

Capture PoD after each stop is completed:

```python
pod = await svc.record_proof_of_delivery(
    "DSP-001",
    stop_id="S3",
    pod_type="signature",
    payload_ref="sig:sha256:3f4a...",
    recipient_name="Jane Doe",
    tenant_id="acme",
)
print(pod["pod_id"], pod["status"])
```

Supported `pod_type` values: `signature`, `photo_ref`, `barcode`.

### Driver Performance Scoring

Compute a 0–100 composite score for any driver:

```python
score = await svc.score_driver_performance("DR-001", tenant_id="acme")
print(f"Score: {score['composite_score']} ({score['tier']})")
print(score["signals"])
```

Tiers: `platinum` (≥90), `gold` (≥75), `silver` (≥55), `bronze` (<55).

Custom signal weights:

```python
score = await svc.score_driver_performance(
    "DR-001",
    weights={"on_time_rate": 0.50, "stop_completion_rate": 0.25, "exception_rate_inverse": 0.15, "communication_responsiveness": 0.10},
    tenant_id="acme",
)
```

### Dispatch Completion & Metrics

```python
completion = await svc.load_completion(
    "DSP-001",
    actual_stops_completed=4,
    exceptions_encountered=1,
    tenant_id="acme",
)
print(f"Stop completion: {completion['stop_completion_rate_pct']}%")
print(f"Exception rate: {completion['exception_rate_pct']}%")
```

### Backhaul Planning

After a dispatch completes, search pending loads near the vehicle's final position to fill the return trip:

```python
backhaul = await svc.plan_backhaul(
    "DSP-001",
    pending_loads=[
        {"order_id": "O-BHL-1", "weight_kg": 4000, "volume_cbm": 16, "origin_lat": -1.30, "origin_lng": 36.83},
        {"order_id": "O-BHL-2", "weight_kg": 8000, "volume_cbm": 30, "origin_lat": -2.00, "origin_lng": 37.50},
    ],
    max_deviation_km=60.0,
    tenant_id="acme",
)
if backhaul["backhaul_viable"]:
    print(f"Backhaul dispatch: {backhaul['backhaul_dispatch_id']}, saving ~{backhaul['empty_km_saved']} km")
```

### Audit Trail Replay

Reconstruct the full ordered state history of any dispatch from its audit events:

```python
trail = await svc.replay_audit_trail("DSP-001", tenant_id="acme")
print(f"Events: {trail['event_count']}")
print(f"State sequence: {trail['state_sequence']}")
for event in trail["ledger"]:
    print(event["event_type"], event["reference_id"])
```

---

## Analytics

```python
# Period KPIs
analytics = await svc.dispatch_analytics("2026-06", tenant_id="acme")
print(f"Completion rate: {analytics['completion_rate_pct']}%")
print(f"Exception rate: {analytics['exception_rate_pct']}%")

# Cost analysis
costs = await svc.cost_analysis("2026-06", tenant_id="acme")
print(f"Total cost: ${costs['total_cost_usd']:.2f}")

# Driver availability before mass dispatch
avail = await svc.driver_availability_check(["DR-001","DR-002","DR-003"], tenant_id="acme")
print(f"Available: {avail['available_count']} / {avail['checked_count']}")
```

---

## UI Routes

| Path | Permission | Description |
|------|-----------|-------------|
| `/transport-dispatch/dashboard` | `transport_dis:view` | Operations overview |
| `/transport-dispatch/loads` | `transport_dis:loads` | Load plan console |
| `/transport-dispatch/loads/create` | `transport_dis:loads_write` | New load plan form |
| `/transport-dispatch/board` | `transport_dis:dispatch` | Dispatch board |
| `/transport-dispatch/drivers` | `transport_dis:drivers` | Driver assignment console |
| `/transport-dispatch/tracking` | `transport_dis:tracking` | Live GPS map |
| `/transport-dispatch/exceptions` | `transport_dis:exceptions` | Exception queue |
| `/transport-dispatch/optimisation` | `transport_dis:optimisation` | Route optimisation |
| `/transport-dispatch/communication` | `transport_dis:communication` | Driver messaging |
| `/transport-dispatch/reports` | `transport_dis:reports` | Export and reporting |
| `/transport-dispatch/agents` | `transport_dis:admin` | AI agent workbench |

---

## Supported Enumerations

| Field | Values |
|-------|--------|
| `load_type` | full_truckload, less_than_truckload, partial_load, express_load, intermodal, bulk_load, temperature_controlled, oversized_load |
| `dispatch status` | planned, assigned, dispatched, in_transit, at_stop, completed, cancelled, exception |
| `exception_type` | vehicle_breakdown, driver_unavailable, traffic_delay, customs_hold, weather_delay, cargo_damage, route_deviation, time_window_missed |
| `assignment_type` | primary, co_driver, relay, standby, temp_assignment |
| `optimisation_mode` | cost, time, distance, co2, balanced, priority_first |
| `tracking update_type` | departure, arrival, waypoint, stop_completed, exception, eta_update, checkpoint |
| `channel` | driver_app, radio, sms, in_cab_terminal, telematics_platform, phone |
| `pod_type` | signature, photo_ref, barcode |

---

## Business Rules Quick Reference

| Rule | Trigger | Action |
|------|---------|--------|
| overload_dispatch_denied | weight > 44,000 kg | `PermissionError` |
| driver_hours_exceeded | hours_available ≤ 0 | `PermissionError` |
| valid_licence_required | licence_valid = false | `PermissionError` |
| vehicle_required | no vehicle_id | `PermissionError` |
| driver_required | no driver_id | `PermissionError` |
| cross_tenant_dispatch_denied | cross-tenant write | `PermissionError` |
| load_type_not_supported | unknown load type | `PermissionError` |

---

## Composition Keywords

When referencing this capability in `.apg` source files:

```apg
use transport_dis;
```

Composes with:
- `transport_rou` — route assignment
- `transport_fle` — vehicle and driver registry
- `transport_sch` — shift-based driver availability
- `transport_tra` — GPS tracking ingestion
- `invoic` — billing trigger on PoD capture
- `ntfy` — exception and SLA breach notifications

---

## Further Reading

- `service.py` — Full service implementation
- `models.py` — Dataclass models
- `capability_contract.py` — Rules engine and contract definitions
- `api.py` — REST endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised enhancement designs
- `README.md` — Quick reference
