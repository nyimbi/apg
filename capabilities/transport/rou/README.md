# Route Optimisation

## Overview
The Route Optimisation capability provides multi-stop route planning with time-window
enforcement, 8 optimisation objectives, dynamic traffic-triggered rerouting,
multi-modal segment planning (road, rail, sea, air), and geospatial address
validation.  It integrates with HERE Maps, Google Maps, TomTom, and other traffic
providers for real-time incident awareness.

## Capability ID
`transport_rou`

## Provides
- multi_stop_route_planning_workflow: Up to 200-stop route plans with sequence management
- dynamic_rerouting_workflow: 8-trigger dynamic rerouting with driver notification
- traffic_integration_workflow: Real-time traffic incident ingestion from 7 providers
- time_window_constraint_workflow: Hard time-window enforcement per stop
- multimodal_routing_workflow: Road/rail/sea/air segment planning
- vrp_workflow: Multi-depot, multi-vehicle VRP using Clarke-Wright savings heuristic
- carbon_budget_routing_workflow: Route planning within hard CO2 emission budgets
- hos_compliance_workflow: EU 561/2006 and US FMCSA hours-of-service checking
- priority_insert_workflow: Minimum-detour insertion of urgent stops into live routes
- fleet_utilisation_workflow: Per-vehicle utilisation metrics and rebalancing signals

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
| hos.regulation | HOS regulation to enforce | eu_561_2006 |
| carbon.max_co2_kg_default | Default CO2 budget per route | none |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-route/routes | GET | Route list | transport_rou:routes |
| /transport-route/optimisation | GET | Optimisation console | transport_rou:optimisation |
| /transport-route/traffic | GET | Traffic events | transport_rou:traffic |
| /transport-route/rerouting | GET | Reroute history | transport_rou:rerouting |
| /transport-route/multimodal | GET | Multimodal segments | transport_rou:multimodal |
| /transport-route/geocoding | GET | Geocoding tools | transport_rou:geocoding |
| /transport-route/fleet | GET | Fleet utilisation | transport_rou:fleet |
| /transport-route/carbon | GET | Carbon analytics | transport_rou:carbon |

## Key Service Methods

### Core CRUD
| Method | Description |
|--------|-------------|
| `plan_route()` | Create a route record |
| `add_route_stop()` | Add a stop to an existing route |
| `add_constraint()` | Attach a constraint to a route |
| `record_traffic_event()` | Record a traffic incident |
| `trigger_reroute()` | Record a dynamic reroute event |
| `plan_multimodal_segment()` | Add a transport-mode segment |
| `register_route_agent()` | Register an AI optimisation agent |
| `mark_stop_completed()` | Mark stop as delivered (POD) |
| `stop_sequence_update()` | Reorder stops in a live route |

### Optimisation
| Method | Description |
|--------|-------------|
| `optimise_route()` | Nearest-neighbour single-vehicle route optimisation |
| `multi_stop_tsp()` | Capacity-constrained TSP heuristic |
| `time_window_routing()` | Earliest-deadline-first time-window routing |
| `vehicle_routing_problem()` | Multi-depot multi-vehicle VRP (Clarke-Wright) |
| `geospatial_cluster_stops()` | k-means stop clustering for batch pre-partitioning |
| `time_window_feasibility()` | Simulate and validate time-window feasibility |
| `priority_stop_insert()` | Minimum-detour urgent stop insertion in live route |

### Traffic & Rerouting
| Method | Description |
|--------|-------------|
| `dynamic_reroute()` | Event-driven dynamic rerouting |
| `traffic_summary()` | Aggregate active traffic delay summary |

### ETA & Cost
| Method | Description |
|--------|-------------|
| `eta_calculation()` | Point ETA to next stop from GPS position |
| `predict_eta_with_traffic()` | Stochastic p50/p90/p95 ETA with live delay |
| `route_cost_estimate()` | Distance-based operating cost estimate |

### Multi-modal & Carbon
| Method | Description |
|--------|-------------|
| `multi_modal_route()` | Build a multi-modal route with per-mode segments |
| `multi_modal_optimise()` | Rank modes by CO2, cost, and time |
| `carbon_optimised_routing()` | Select lowest-carbon mode for origin→destination |
| `co2_optimised_route()` | Full route plan optimised for minimum CO2 |
| `carbon_budget_route()` | Route plan constrained to max CO2 budget |

### Compliance & Analytics
| Method | Description |
|--------|-------------|
| `hours_of_service_check()` | EU 561/2006 / US FMCSA HOS compliance |
| `route_compliance_check()` | Weight, hazmat, and geofence rule audit |
| `route_analytics()` | Period-level KPI aggregation |
| `route_kpi_summary()` | Concise KPI card for dashboard |
| `fleet_route_summary()` | Aggregate fleet-level stats |
| `fleet_utilisation_report()` | Per-vehicle utilisation with over/under thresholds |
| `route_replay()` | Chronological event-history replay for audit |
| `historical_route_analysis()` | Per-route historical performance data |

### Driver & Export
| Method | Description |
|--------|-------------|
| `driver_route_guide()` | Human-readable turn-by-turn stop guide |
| `driver_route_feedback()` | Capture driver 1–5 star feedback |
| `export_route_data()` | Metadata record for route data export |
| `bulk_plan_routes()` | Batch route planning from list of requests |

### Geofencing & Health
| Method | Description |
|--------|-------------|
| `geofence_routing()` | Route planning with zone avoidance |
| `health_check()` | Service health status |

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
- RouteStop: id, route_id, sequence, location, address, time_window_start, time_window_end, completed
- RouteConstraint: id, route_id, constraint_type, parameters
- TrafficIntegration: id, provider, route_id, incident_type, delay_minutes
- RerouteEvent: id, original_route_id, new_route_id, trigger, distance_delta_km
- MultimodalSegment: id, route_id, transport_mode, segment_origin, segment_destination

## Streaming Events
- route_planned, route_optimised, route_dispatched, traffic_incident_detected
- reroute_triggered, reroute_completed, constraint_violation_detected, multimodal_segment_planned
- priority_stop_inserted_urgent, hos_check_completed, carbon_budget_route_planned
- fleet_utilisation_report_generated, route_compliance_checked, driver_route_feedback_recorded

## Edge Cases Handled
- All addresses must be validated before route dispatch (not just origin/destination)
- Capacity constraint violation blocks route plan regardless of stop count
- Stop count exceeding 200 triggers a split-route recommendation
- Reroute triggers must come from the supported list — free-form triggers are rejected
- Multimodal sea segments use the same origin/destination model but with carrier_ref
- HOS check accounts for accumulated driving time before the new route starts
- Carbon budget route returns `feasible=false` with all options when no mode fits the budget
- VRP with more stops than vehicle capacity infeasible slots returns them as `unassigned_stop_ids`

## Composability Notes
Routes are consumed by `transport_dis` for dispatch assignment.  Stop time windows
are synchronised with `transport_sch` schedules.  Traffic feeds integrate with
`transport_tra` for live vehicle position data.  Route completion events feed
delivery ETAs back to `transport_del`.  Carbon budget results feed `intel_esg` for
scope-3 emissions reporting.  Fleet utilisation signals feed `transport_flt` for
vehicle rebalancing.

---

## World-Class Enhancements (v2.0)

- **I1.** Route Optimisation — World-Class Improvement Roadmap
- **I2.** OR-Tools VRP Solver Integration
- **I3.** Real-Time Traffic Adapter Pool
- **I4.** Multi-Depot VRP Support
- **I5.** Stochastic ETA with Confidence Intervals
- **I6.** Carbon Budget Constraint Engine
- **I7.** Driver Fatigue & HOS Compliance
- **I8.** Predictive Load Balancing Across Fleet
- **I9.** Turn-Restriction & Road-Attribute Graph
- **I10.** Priority Delivery Escalation
- **I11.** Persistent Route History & Replay
- **I12.** Fleet Utilisation Optimisation
- **I13.** Live Map Streaming via Server-Sent Events
- **I14.** LLM-Powered Natural Language Route Briefing
- **I15.** Geospatial Clustering for Batch Optimisation

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
