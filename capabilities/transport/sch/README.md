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

## Composability Notes
Feeds driver and vehicle availability to `transport_dis`. Charter dates and vehicle assignments inform `transport_fle` utilisation analytics. Maintenance downtime from `transport_mai` creates schedule blocks. Route plans from `transport_rou` are referenced in vehicle assignments.
