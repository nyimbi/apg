# Store Intelligence

**Capability ID**: `retail_sin` | **Domain**: `retail` | **Version**: `1.0.0`

## Description

Provides anonymised in-store analytics: foot traffic counting with multi-sensor support, zone-level dwell time and heatmap generation, AI-assisted planogram compliance auditing, real-time shelf availability alerting with automatic replenishment triggering, shopper conversion funnel tracking, store KPI scorecards with peer-group benchmarking, and a store performance dashboard. All personal data is anonymised at ingest; raw video storage and biometric identification are denied by rule engine.

## Installation

```bash
pip install apg-retail-sin
```

## Provides

- `store_foot_traffic_analytics`
- `planogram_compliance_monitoring`
- `shelf_availability_alerting`
- `store_conversion_optimisation`
- `store_performance_benchmarking`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/retail-sin/dashboard` | `retail_sin:view` | Overview |
| `/retail-sin/traffic` | `retail_sin:view` | Traffic |
| `/retail-sin/heatmaps` | `retail_sin:view` | Traffic |
| `/retail-sin/planogram` | `retail_sin:view` | Compliance |
| `/retail-sin/planogram/<id>` | `retail_sin:view` | Compliance |
| `/retail-sin/shelf-alerts` | `retail_sin:view` | Availability |
| `/retail-sin/conversion` | `retail_sin:view` | Performance |
| `/retail-sin/journey` | `retail_sin:view` | Performance |

## Key Service Methods

- `create_store()`
- `get_store()`
- `get_store_by_code()`
- `list_stores()`
- `create_zone()`
- `list_zones()`
- `register_sensor()`
- `sensor_heartbeat()`
- `list_sensors()`
- `foot_traffic_record()`

_(See `service.py` for complete API.)_

## Interoperability

`retail_sin` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use retail_sin;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `RETAIL_SIN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
