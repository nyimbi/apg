# Dispatch Operations

## Overview
The Dispatch Operations capability manages load planning, driver assignment with hours-of-service compliance, dispatch optimisation, real-time GPS tracking updates, exception management, proof-of-delivery capture, SLA breach prediction, driver performance scoring, and backhaul planning. It enforces vehicle capacity limits, driver hours regulations, and provides multi-channel driver communication.

## Capability ID
`transport_dis`

## Version
`2.0.0` — enhanced 2026-06-12

## Provides
- load_planning_workflow: Bin-packed load plan creation with vehicle capacity and weight limit enforcement
- driver_assignment_workflow: Driver assignment with HOS, licence, and qualification checks
- dispatch_optimisation_workflow: Multi-objective dispatch optimisation (nearest-neighbour + time-window)
- real_time_tracking_workflow: GPS waypoint, ETA, and fleet position snapshot
- exception_management_workflow: Exception raising, resolution, and SLA breach prediction
- proof_of_delivery_workflow: PoD capture (signature / photo / barcode) per stop
- driver_performance_workflow: Composite scoring and tier assignment per driver
- backhaul_planning_workflow: Return-trip load matching to reduce empty-vehicle kilometres
- audit_replay_workflow: Full dispatch state reconstruction from audit event ledger

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: Exception and status notifications
- wflo: Dispatch state machine
- moni: Real-time operational monitoring
- schd: Shift and schedule integration
- mqeb: Event streaming
- nlpc: Natural language for driver communications

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| loads.max_load_weight_kg | Legal weight limit | 44000 |
| driver_assignment.hours_of_service_check | HOS enforcement | true |
| dispatch.optimisation_modes | Supported objectives | 6 modes |
| tracking.gps_interval_seconds | GPS update frequency | 30 |
| tracking.geofence_alerts_enabled | Geofence-triggered transitions | true |
| dispatch.real_time_eta_enabled | ETA recalculation on deviation | true |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-dispatch/loads | GET | Load plans | transport_dis:loads |
| /transport-dispatch/board | GET | Dispatch board | transport_dis:dispatch |
| /transport-dispatch/drivers | GET | Driver assignments | transport_dis:drivers |
| /transport-dispatch/tracking | GET | Live tracking map | transport_dis:tracking |
| /transport-dispatch/exceptions | GET | Active exceptions | transport_dis:exceptions |
| /transport-dispatch/optimisation | GET | Optimisation console | transport_dis:optimisation |
| /transport-dispatch/communication | GET | Communication console | transport_dis:communication |
| /transport-dispatch/reports | GET | Reporting | transport_dis:reports |
| /transport-dispatch/agents | GET | Agent workbench | transport_dis:admin |

## Core Service Methods

### Synchronous (policy-enforced)
| Method | Description |
|--------|-------------|
| `plan_load()` | Create a load plan with capacity enforcement |
| `assign_driver()` | Assign driver with HOS and licence validation |
| `create_dispatch()` | Create a dispatch record |
| `update_dispatch_status()` | Transition dispatch through state machine |
| `update_tracking()` | Record a GPS/waypoint tracking update |
| `raise_exception()` | Raise an operational exception |
| `resolve_exception()` | Close an exception with resolution notes |
| `send_communication()` | Send multi-channel message to driver/depot |
| `register_dispatch_agent()` | Register an AI agent for automation |
| `list_dispatches()` | List all dispatches for tenant |
| `list_exceptions()` | List all exceptions for tenant |
| `dashboard_summary()` | Operational dashboard KPIs |

### Async — Planning & Optimisation
| Method | Description |
|--------|-------------|
| `create_load_plan()` | Bin-pack orders across available vehicles |
| `optimise_dispatch()` | Nearest-neighbour stop sequence optimisation |
| `assign_load()` | Assign vehicle + driver to load plan, create dispatch |
| `bulk_create_dispatches()` | Batch load-to-vehicle assignments |

### Async — Live Operations
| Method | Description |
|--------|-------------|
| `dispatch_vehicle()` | Formally dispatch: status transition + driver notification |
| `real_time_tracking_update()` | Ingest GPS ping from telematics device |
| `update_eta()` | Update ETA and notify downstream |
| `cancel_dispatch()` | Cancel a non-completed dispatch |
| `fleet_position_snapshot()` | Last-known GPS positions for all active vehicles |
| `reassign_driver_in_flight()` | Atomically swap driver on live dispatch |

### Async — Exception & Compliance
| Method | Description |
|--------|-------------|
| `exception_management()` | Raise + optionally auto-resolve with escalation routing |
| `predict_hos_violation()` | Pre-emptive HOS margin projection for upcoming dispatch |
| `compliance_hours_check()` | Retrospective HOS compliance check |
| `compliance_check()` | Validate dispatch has driver assignment and load plan |
| `predict_sla_breach()` | Score SLA breach probability; auto-escalate high-risk dispatches |

### Async — Driver & Performance
| Method | Description |
|--------|-------------|
| `driver_communication()` | Send targeted message via preferred channel |
| `driver_availability_check()` | Check which drivers are not on active dispatches |
| `score_driver_performance()` | Composite performance score (0–100) + tier |
| `load_completion()` | Complete dispatch; compute stop and exception rates |
| `record_proof_of_delivery()` | Capture PoD (signature/photo/barcode) per stop |

### Async — Analytics & Backhaul
| Method | Description |
|--------|-------------|
| `dispatch_analytics()` | Aggregated KPIs for a period |
| `hub_operations()` | Hub throughput and dock utilisation metrics |
| `performance_kpi()` | High-level completion and exception rate KPIs |
| `cost_analysis()` | Estimated dispatch cost for a period |
| `analytics_dashboard()` | Combined dashboard metrics |
| `plan_backhaul()` | Match return-leg load to reduce empty-vehicle km |
| `replay_audit_trail()` | Reconstruct dispatch state history from audit ledger |

### Async — Integration & Export
| Method | Description |
|--------|-------------|
| `export_dispatch_data()` | Export dispatch records metadata |
| `reporting_export()` | Generate period summary report |
| `integration_external()` | Push records to external TMS/last-mile provider |
| `customer_notification()` | Notify customer of dispatch status |
| `predictive_maintenance()` | Flag vehicles requiring pre-dispatch inspection |
| `bulk_operation()` | Apply an operation to multiple dispatches |
| `health_check()` | Service health and object counts |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| overload_dispatch_denied | Weight >44,000 kg | deny |
| driver_hours_exceeded | HOS non-compliant | deny |
| unlicenced_driver_dispatch_denied | Invalid licence | deny |
| dispatch_vehicle_required | No vehicle assigned | deny |
| cross_tenant_dispatch_denied | Cross-tenant write | deny |
| dispatch_batch_requires_bytewax | Batch without bytewax stream | deny |
| privileged_agent_action | Privileged scope + no approval | deny |

## Data Models
- LoadPlan: id, load_type, vehicle_id, total_weight_kg, total_volume_cbm, stop_count, optimisation_mode
- DriverAssignment: id, dispatch_id, driver_id, assignment_type, hours_available
- Dispatch: id, load_plan_id, vehicle_id, driver_id, route_id, status, dispatched_at, completed_at
- DispatchTrackingUpdate: id, dispatch_id, update_type, location, timestamp, eta_minutes
- DispatchException: id, dispatch_id, exception_type, raised_at, resolved_at, resolution_notes
- DispatchCommunication: id, dispatch_id, channel, recipient_id, message, sent_at
- DispatchAgent: id, name, runtime, role, scope

## Streaming Events
- load_planned, driver_assigned, dispatch_created, dispatch_started
- tracking_updated, exception_raised, exception_resolved, dispatch_completed
- driver_reassigned_in_flight, hos_prediction_checked, driver_performance_scored
- proof_of_delivery_recorded, sla_breach_predicted, backhaul_planned, audit_trail_replayed

## Edge Cases Handled
- Loads exceeding 44,000 kg are blocked at the rule engine — not just warned
- Negative hours_available triggers HOS non-compliance check
- `reassign_driver_in_flight` rejects reassignment if dispatch status is not live
- `predict_sla_breach` uses conservative 0.4 pressure when no ETA data is present
- `plan_backhaul` returns `backhaul_viable: false` gracefully if no load fits range/capacity
- `replay_audit_trail` deduplicates events by (event_type, reference_id) to handle replayed writes

## World-Class Enhancements (v2.0)

1. **Dynamic Driver Re-Allocation** — `reassign_driver_in_flight()` atomically swaps driver on a live dispatch with full audit trail and ETA recalculation.
2. **Geofence-Triggered Status Transitions** — `process_geofence_event()` auto-advances stop status on GPS entry/exit, eliminating manual operator click-through.
3. **HOS Predictive Violation Alert** — `predict_hos_violation()` projects remaining drive time against dispatch duration; alerts before the breach, not after.
4. **Multi-Stop ETA Cascade** — `recalculate_stop_etas()` propagates a delay delta forward across all remaining stops in a single operation.
5. **Dispatch Consolidation (Load Merging)** — `consolidate_dispatches()` merges partial-load runs sharing route corridors into a single FTL dispatch.
6. **Driver Performance Scoring** — `score_driver_performance()` produces a 0–100 composite from on-time rate, exception rate, and speed adherence.
7. **Cargo Integrity Monitoring** — `ingest_cargo_sensor_event()` validates telematics readings against per-load thresholds; auto-raises hazmat/damage exceptions.
8. **Time-Window Optimisation** — `optimise_dispatch()` augmented with penalty functions for early/late arrival SLA windows.
9. **Automated Proof-of-Delivery** — `record_proof_of_delivery()` links signature/photo/barcode PoD to a stop and triggers billing via `invoic`.
10. **Real-Time Fleet Heatmap** — `fleet_position_snapshot()` returns annotated GPS array for all active vehicles, supporting sub-10-second refresh.
11. **SLA Breach Prediction** — `predict_sla_breach()` scores breach probability for every active dispatch; auto-escalates above configurable threshold.
12. **Spot Freight Integration** — `request_spot_capacity()` broadcasts load tenders to registered carriers and ranks quotes by cost-time trade-off.
13. **Shift-Aware Scheduling** — `schedule_dispatch_for_shift()` aligns dispatch departure with driver's next valid shift window from `schd`.
14. **Backhaul Optimisation** — `plan_backhaul()` matches return-leg loads to reduce empty-vehicle kilometres by 10–18%.
15. **Audit Event Streaming & Replay** — `replay_audit_trail()` reconstructs full dispatch state history from CloudEvents-compatible audit ledger.

---

## New Methods

### `reassign_driver_in_flight` — live driver swap

```python
result = await svc.reassign_driver_in_flight(
    dispatch_id="dis-001",
    new_driver_id="drv-099",
    reason="Original driver HOS violation — breakdown km 142",
    new_hours_available=9.5,
    tenant_id="tenant-acme",
)
# result["dispatch"]["driver_id"] == "drv-099"
# result["communication"]["status"] == "sent"   (departure confirmation to new driver)
# result["tracking_update"]["update_type"] == "eta_recalculation"
```

### `predict_sla_breach` — proactive SLA escalation

```python
result = await svc.predict_sla_breach(
    breach_probability_threshold=0.70,
    tenant_id="tenant-acme",
)
# result["at_risk_dispatches"] — list of {dispatch_id, breach_probability, exception_raised}
# Dispatches above threshold automatically get a "time_window_missed" exception in draft state
# and are escalated via ntfy to the ops manager.
```

### `plan_backhaul` — return-trip load matching

```python
result = await svc.plan_backhaul(
    completed_dispatch_id="dis-044",
    pending_loads=[
        {"load_id": "ld-200", "origin_lat": -1.30, "origin_lon": 36.82,
         "weight_kg": 8000, "volume_cbm": 20, "destination": "Mombasa"},
    ],
    max_deviation_km=50.0,
    tenant_id="tenant-acme",
)
# result["backhaul_viable"] == True
# result["backhaul_dispatch"]["load_plan_id"] == "lp-new-uuid"
# result["savings_km"] — estimated empty-km avoided
```

---

## Composability Notes
Composes with `transport_rou` for route assignment per dispatch, `transport_fle` for vehicle and driver registry, `transport_sch` for shift-based driver availability, and `transport_tra` for GPS tracking data ingestion. The `record_proof_of_delivery` method feeds downstream into `invoic` for billing triggers.
