# Transport Scheduling

## Overview
The Transport Scheduling capability manages load scheduling, driver shift planning with tachograph and HOS compliance, vehicle assignment, charter management (school, corporate, tourist, medical), schedule optimisation, and conflict detection. It blocks schedule publication when unresolved conflicts exist and enforces tacho compliance on all shifts.

## Capability ID
`transport_sch`

## Provides
- load_scheduling_workflow: Scheduled load planning with advance booking horizon
- driver_shift_planning_workflow: Shift creation with HOS and tacho compliance
- vehicle_assignment_workflow: Vehicle-to-schedule assignment with double-booking detection
- charter_management_workflow: Charter booking with customer confirmation enforcement
- schedule_optimisation_workflow: Multi-objective schedule optimisation

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: Schedule publication and conflict notifications
- wflo: Schedule state machine
- moni: Conflict and resource monitoring
- schd: Core scheduling engine
- mqeb: Event streaming
- comp: HOS and tacho regulatory compliance

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| shifts.max_daily_hours | Maximum daily hours | 10 |
| shifts.max_weekly_hours | Maximum weekly hours | 56 |
| shifts.tacho_compliance_enabled | Tacho check | true |
| conflicts.block_publish_on_conflict | Block publish | true |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-scheduling/schedules | GET | Schedule list | transport_sch:schedules |
| /transport-scheduling/calendar | GET | Calendar view | transport_sch:view |
| /transport-scheduling/shifts | GET | Driver shifts | transport_sch:shifts |
| /transport-scheduling/vehicles | GET | Vehicle assignments | transport_sch:vehicles |
| /transport-scheduling/charters | GET | Charter bookings | transport_sch:charters |
| /transport-scheduling/conflicts | GET | Conflicts console | transport_sch:conflicts |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| driver_hours_breach_denied | HOS non-compliant | deny |
| double_booking_denied | Double booking detected | deny |
| publish_blocked_on_conflict | Unresolved conflicts | deny |
| charter_customer_confirmation_required | No confirmation | deny |
| tacho_compliance_required | Tacho non-compliant | deny |

## Data Models
- Schedule: id, schedule_type, status, start_date, end_date, optimisation_mode
- DriverShift: id, schedule_id, driver_id, shift_type, start_time, end_time, hours, tacho_compliant
- VehicleAssignment: id, schedule_id, vehicle_id, route_id, assigned_from, assigned_until
- Charter: id, charter_type, customer_id, vehicle_id, driver_id, charter_date, customer_confirmed
- ScheduleConflict: id, schedule_id, conflict_type, resource_id, detected_at, resolved_at

## Streaming Events
- schedule_created, schedule_published, shift_assigned, vehicle_assigned
- charter_confirmed, conflict_detected, conflict_resolved, schedule_optimised

## Edge Cases Handled
- Publication is hard-blocked if any conflict remains unresolved
- Charter dispatch requires vehicle inspection to be completed first
- Double booking is checked at assignment time, not just at publish
- Customer confirmation is required for all charter types — no default assumed
- Negative or zero shift hours trigger HOS non-compliance

## Key Service Methods

### Core (sync)
| Method | Description |
|--------|-------------|
| `describe()` | Return capability contract |
| `evaluate(context)` | Evaluate policy rules |
| `create_schedule(...)` | Create a schedule in draft |
| `publish_schedule(...)` | Publish (blocked if conflicts exist) |
| `create_shift(...)` | Create a driver shift with HOS check |
| `assign_vehicle(...)` | Assign vehicle; detect double-bookings |
| `create_charter(...)` | Create a charter booking |
| `record_conflict(...)` | Record a scheduling conflict |
| `resolve_conflict(...)` | Resolve a conflict |
| `send_notification(...)` | Send a scheduling notification |
| `register_scheduling_agent(...)` | Register an AI scheduling agent |
| `dashboard_summary(...)` | Aggregate KPI dashboard card |
| `list_schedules(...)` | List all schedules for a tenant |
| `list_open_conflicts(...)` | List unresolved conflicts |

### Async Workflow Methods
| Method | Description |
|--------|-------------|
| `create_schedule_async(...)` | Create schedule + vehicle stubs for multiple routes |
| `driver_shift_planning(...)` | Assign drivers to shifts with HOS constraint checking |
| `vehicle_assignment(...)` | Bulk-assign vehicles with double-booking detection |
| `charter_booking(...)` | Create and price a charter with fuel surcharge |
| `schedule_conflict_check(...)` | Scan schedule for driver/vehicle conflicts |
| `schedule_analytics(...)` | KPI aggregation for a period |
| `capacity_planning(...)` | Demand vs. scheduled capacity gap analysis |
| `driver_hours_compliance(...)` | EU HOS compliance check for a driver |
| `schedule_publish(...)` | Publish with auto conflict check + driver notifications |
| `schedule_disruption_management(...)` | Log and triage a scheduling disruption |
| `schedule_optimise_ml(...)` | ML-informed shift consolidation optimisation |
| `schedule_kpi_summary(...)` | Concise KPI card for dashboard |
| `passenger_load_forecast(...)` | Forecast load over a planning horizon |
| `schedule_deviation_alert(...)` | Raise alert when departure deviates beyond threshold |
| `schedule_compare(...)` | Compare two schedules by shifts and conflicts |
| `shift_swap_approve(...)` | Approve and record a driver shift swap |
| `schedule_analytics_detail(...)` | Detailed analytics: distribution and charter usage |
| `tachograph_compliance_report(...)` | Per-driver tachograph compliance report |
| `bulk_assign_vehicles(...)` | Bulk vehicle-to-route assignment |
| `driver_roster(...)` | Driver roster with shift detail for a schedule |
| `export_schedule_data(...)` | Export schedule data metadata |
| `health_check()` | Service health status |
| `cancel_shift(...)` | Cancel a shift with reason and audit |
| `charter_cost_summary(...)` | Summarise charter bookings and estimated revenue |
| `driver_shift_summary(...)` | Shift statistics for a schedule |

### World-Class Enhancement Methods (v1.1+)
| Method | Description |
|--------|-------------|
| `schedule_version_snapshot(...)` | Snapshot schedule state for versioning |
| `schedule_rollback(...)` | Roll schedule back to a prior version |
| `driver_preference_update(...)` | Record driver shift preferences and constraints |
| `schedule_sla_report(...)` | On-time performance SLA aggregation |
| `charter_dynamic_price(...)` | Dynamic pricing with surge, lead-time, and utilisation factors |
| `gtfs_export(...)` | GTFS Static feed skeleton for a published schedule |
| `notification_escalation_ladder(...)` | SLA-aware multi-step notification escalation |
| `driver_wellbeing_score(...)` | Composite wellbeing score from shift data |
| `multi_schedule_capacity_summary(...)` | Cross-schedule capacity aggregation |
| `compliance_audit_pack(...)` | Full compliance audit pack for regulatory submission |

## Composability Notes
Feeds driver and vehicle availability to `transport_dis`. Charter dates and vehicle assignments inform `transport_fle` utilisation analytics. Maintenance downtime from `transport_mai` creates schedule blocks. Route plans from `transport_rou` are referenced in vehicle assignments.

## Further Reading
- `service.py` — Business logic and async workflow methods
- `models.py` — Dataclass models
- `capability_contract.py` — Policy rules and supported enumerations
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `docs/user_guide.md` — Detailed usage guide
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 strategic improvement proposals
