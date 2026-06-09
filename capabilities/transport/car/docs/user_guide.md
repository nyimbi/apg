# Cargo Management

**Capability ID**: `transport_car` | **Domain**: `transport` | **Version**: `1.0.0`

## Description

The Cargo Management capability provides end-to-end cargo lifecycle management including booking creation, manifest generation, dangerous goods compliance, real-time cargo tracking, and revenue management. It enforces IATA, IMDG, ADR, and C-TPAT compliance standards and integrates with bytewax for streaming cargo lifecycle events.

## Installation

```bash
pip install apg-transport-car
```

## Provides

- `cargo_booking_workflow`
- `cargo_manifest_workflow`
- `dangerous_goods_compliance_workflow`
- `cargo_tracking_workflow`
- `cargo_revenue_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/transport-cargo/dashboard` | `transport_car:view` | Overview |
| `/transport-cargo/bookings` | `transport_car:bookings` | Bookings |
| `/transport-cargo/bookings/create` | `transport_car:bookings_write` | Bookings |
| `/transport-cargo/manifests` | `transport_car:manifests` | Documentation |
| `/transport-cargo/dangerous-goods` | `transport_car:dg_compliance` | Compliance |
| `/transport-cargo/tracking` | `transport_car:tracking` | Operations |
| `/transport-cargo/tracking/<booking_id>` | `transport_car:tracking` | Operations |
| `/transport-cargo/revenue` | `transport_car:revenue` | Finance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_booking()`
- `create_manifest()`
- `declare_dangerous_goods()`
- `update_tracking()`
- `record_revenue()`
- `record_compliance()`
- `register_cargo_agent()`
- `validate_batch()`

_(See `service.py` for complete API.)_

## Interoperability

`transport_car` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use transport_car;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_CAR_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
