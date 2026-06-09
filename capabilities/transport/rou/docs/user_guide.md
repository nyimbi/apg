# Route Optimisation

**Capability ID**: `transport_rou` | **Domain**: `transport` | **Version**: `1.0.0`

## Description

The Route Optimisation capability provides multi-stop route planning with time-window enforcement, 8 optimisation objectives, dynamic traffic-triggered rerouting, multi-modal segment planning (road, rail, sea, air), and geospatial address validation. It integrates with HERE Maps, Google Maps, TomTom, and other traffic providers for real-time incident awareness.

## Installation

```bash
pip install apg-transport-rou
```

## Provides

- `multi_stop_route_planning_workflow`
- `dynamic_rerouting_workflow`
- `traffic_integration_workflow`
- `time_window_constraint_workflow`
- `multimodal_routing_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/transport-route/dashboard` | `transport_rou:view` | Overview |
| `/transport-route/routes` | `transport_rou:routes` | Routes |
| `/transport-route/routes/create` | `transport_rou:routes_write` | Routes |
| `/transport-route/routes/<route_id>/map` | `transport_rou:routes` | Routes |
| `/transport-route/optimisation` | `transport_rou:optimisation` | Optimisation |
| `/transport-route/constraints` | `transport_rou:constraints` | Planning |
| `/transport-route/traffic` | `transport_rou:traffic` | Traffic |
| `/transport-route/rerouting` | `transport_rou:rerouting` | Dynamic |

## Key Service Methods

- `describe()`
- `evaluate()`
- `plan_route()`
- `add_route_stop()`
- `add_constraint()`
- `record_traffic_event()`
- `trigger_reroute()`
- `plan_multimodal_segment()`
- `register_route_agent()`
- `validate_batch()`

_(See `service.py` for complete API.)_

## Interoperability

`transport_rou` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use transport_rou;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_ROU_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
