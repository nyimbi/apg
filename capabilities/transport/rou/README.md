# Route Optimisation

## Overview
The Route Optimisation capability provides multi-stop route planning with time-window enforcement, 8 optimisation objectives, dynamic traffic-triggered rerouting, multi-modal segment planning (road, rail, sea, air), and geospatial address validation. It integrates with HERE Maps, Google Maps, TomTom, and other traffic providers for real-time incident awareness.

## Capability ID
`transport_rou`

## Provides
- multi_stop_route_planning_workflow: Up to 200-stop route plans with sequence management
- dynamic_rerouting_workflow: 8-trigger dynamic rerouting with driver notification
- traffic_integration_workflow: Real-time traffic incident ingestion from 7 providers
- time_window_constraint_workflow: Hard time-window enforcement per stop
- multimodal_routing_workflow: Road/rail/sea/air segment planning

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: Reroute and ETA update notifications
- wflo: Route lifecycle management
- moni: Traffic and route health monitoring
- nlpc: Address parsing and geocoding
- mqeb: Event streaming
- schd: Schedule-aware route planning

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| routes.max_stops_per_route | Maximum stops | 200 |
| optimisation.default_objective | Default objective | minimize_cost |
| optimisation.max_optimisation_seconds | Solver time limit | 30 |
| rerouting.auto_reroute_enabled | Auto dynamic reroute | true |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-route/routes | GET | Route list | transport_rou:routes |
| /transport-route/optimisation | GET | Optimisation console | transport_rou:optimisation |
| /transport-route/traffic | GET | Traffic events | transport_rou:traffic |
| /transport-route/rerouting | GET | Reroute history | transport_rou:rerouting |
| /transport-route/multimodal | GET | Multimodal segments | transport_rou:multimodal |
| /transport-route/geocoding | GET | Geocoding tools | transport_rou:geocoding |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| unvalidated_address_dispatch_denied | Address not validated | deny |
| capacity_constraint_violation | Capacity exceeded | deny |
| max_stops_exceeded | >200 stops | deny |
| route_origin_required | Origin absent | deny |
| cross_tenant_route_denied | Cross-tenant write | deny |

## Data Models
- Route: id, route_type, origin, destination, vehicle_id, transport_mode, stop_count, total_distance_km
- RouteStop: id, route_id, sequence, location, address, time_window_start, time_window_end
- RouteConstraint: id, route_id, constraint_type, parameters
- TrafficIntegration: id, provider, route_id, incident_type, delay_minutes
- RerouteEvent: id, original_route_id, new_route_id, trigger, distance_delta_km
- MultimodalSegment: id, route_id, transport_mode, segment_origin, segment_destination

## Streaming Events
- route_planned, route_optimised, route_dispatched, traffic_incident_detected
- reroute_triggered, reroute_completed, constraint_violation_detected, multimodal_segment_planned

## Edge Cases Handled
- All addresses must be validated before route dispatch (not just origin/destination)
- Capacity constraint violation blocks route plan regardless of stop count
- Stop count exceeding 200 triggers a split-route recommendation
- Reroute triggers must come from the supported list — free-form triggers are rejected
- Multimodal sea segments use the same origin/destination model but with carrier_ref

## Composability Notes
Routes are consumed by `transport_dis` for dispatch assignment. Stop time windows are synchronised with `transport_sch` schedules. Traffic feeds integrate with `transport_tra` for live vehicle position data. Route completion events feed delivery ETAs back to `transport_del`.
