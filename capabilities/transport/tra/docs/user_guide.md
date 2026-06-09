# Asset Tracking

**Capability ID**: `transport_tra` | **Domain**: `transport` | **Version**: `1.0.0`

## Description

The Asset Tracking capability provides real-time GPS tracking for vehicles, trailers, containers, pallets, and equipment. It supports geofence creation (circle, polygon, corridor, exclusion zone), cold-chain temperature monitoring with breach detection, container tracking with ISO number and seal management, and utilisation analytics. Tamper detection requires immediate escalation.

## Installation

```bash
pip install apg-transport-tra
```

## Provides

- `realtime_gps_tracking_workflow`
- `geofencing_workflow`
- `cold_chain_monitoring_workflow`
- `container_tracking_workflow`
- `asset_utilisation_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/transport-tracking/dashboard` | `transport_tra:view` | Overview |
| `/transport-tracking/map` | `transport_tra:view` | Live |
| `/transport-tracking/assets` | `transport_tra:assets` | Assets |
| `/transport-tracking/assets/<asset_id>` | `transport_tra:assets` | Assets |
| `/transport-tracking/geofencing` | `transport_tra:geofencing` | Geofencing |
| `/transport-tracking/alerts` | `transport_tra:alerts` | Alerts |
| `/transport-tracking/cold-chain` | `transport_tra:cold_chain` | Cold Chain |
| `/transport-tracking/containers` | `transport_tra:containers` | Containers |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_asset()`
- `update_asset_location()`
- `create_geofence()`
- `raise_alert()`
- `acknowledge_alert()`
- `record_cold_chain()`
- `register_container()`
- `update_container_status()`

_(See `service.py` for complete API.)_

## Interoperability

`transport_tra` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use transport_tra;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_TRA_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
