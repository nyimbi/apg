# Dispatch Operations

## Overview
The Dispatch Operations capability manages load planning, driver assignment with hours-of-service compliance, dispatch optimisation, real-time GPS tracking updates, and exception management. It enforces vehicle capacity limits, driver hours regulations, and provides multi-channel driver communication.

## Capability ID
`transport_dis`

## Provides
- load_planning_workflow: Load plan creation with vehicle capacity and weight limit enforcement
- driver_assignment_workflow: Driver assignment with HOS, licence, and qualification checks
- dispatch_optimisation_workflow: Multi-objective dispatch optimisation
- real_time_tracking_workflow: GPS waypoint and ETA tracking updates
- exception_management_workflow: Exception raising and resolution workflow

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

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-dispatch/loads | GET | Load plans | transport_dis:loads |
| /transport-dispatch/board | GET | Dispatch board | transport_dis:dispatch |
| /transport-dispatch/drivers | GET | Driver assignments | transport_dis:drivers |
| /transport-dispatch/tracking | GET | Live tracking map | transport_dis:tracking |
| /transport-dispatch/exceptions | GET | Active exceptions | transport_dis:exceptions |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| overload_dispatch_denied | Weight >44,000 kg | deny |
| driver_hours_exceeded | HOS non-compliant | deny |
| unlicenced_driver_dispatch_denied | Invalid licence | deny |
| dispatch_vehicle_required | No vehicle assigned | deny |
| cross_tenant_dispatch_denied | Cross-tenant write | deny |

## Data Models
- LoadPlan: id, load_type, vehicle_id, total_weight_kg, total_volume_cbm, stop_count, optimisation_mode
- DriverAssignment: id, dispatch_id, driver_id, assignment_type, hours_available
- Dispatch: id, load_plan_id, vehicle_id, driver_id, route_id, status
- DispatchTrackingUpdate: id, dispatch_id, update_type, location, timestamp, eta_minutes
- DispatchException: id, dispatch_id, exception_type, raised_at, resolved_at
- DispatchCommunication: id, dispatch_id, channel, recipient_id, message

## Streaming Events
- load_planned, driver_assigned, dispatch_created, dispatch_started
- tracking_updated, exception_raised, exception_resolved, dispatch_completed

## Edge Cases Handled
- Loads exceeding 44,000 kg are blocked at the rule engine — not just warned
- Negative hours_available triggers HOS non-compliance check
- Exceptions must be explicitly resolved before dispatch can complete
- Cross-tenant dispatch writes are denied regardless of role

## Composability Notes
Composes with `transport_rou` for route assignment per dispatch, `transport_fle` for vehicle and driver registry, `transport_sch` for shift-based driver availability, and `transport_tra` for GPS tracking data ingestion.
