# Delivery Management

**Capability ID**: `transport_del` | **Domain**: `transport` | **Version**: `1.0.0`

## Description

The Delivery Management capability handles last-mile delivery planning, proof-of-delivery capture, customer notifications, failed delivery handling, rescheduling workflows, SLA tracking, and return management. It enforces geo-stamped POD capture and protects against POD falsification.

## Installation

```bash
pip install apg-transport-del
```

## Provides

- `delivery_planning_workflow`
- `proof_of_delivery_workflow`
- `customer_notification_workflow`
- `failed_delivery_workflow`
- `sla_tracking_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/transport-delivery/dashboard` | `transport_del:view` | Overview |
| `/transport-delivery/deliveries` | `transport_del:deliveries` | Operations |
| `/transport-delivery/deliveries/create` | `transport_del:deliveries_write` | Operations |
| `/transport-delivery/pod` | `transport_del:pod` | Evidence |
| `/transport-delivery/failed` | `transport_del:failed` | Exceptions |
| `/transport-delivery/rescheduling` | `transport_del:rescheduling` | Exceptions |
| `/transport-delivery/sla` | `transport_del:sla` | Performance |
| `/transport-delivery/notifications` | `transport_del:notifications` | Communications |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_delivery()`
- `record_pod()`
- `record_failed_delivery()`
- `reschedule_delivery()`
- `set_sla()`
- `send_notification()`
- `create_return()`
- `register_delivery_agent()`

_(See `service.py` for complete API.)_

## Interoperability

`transport_del` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use transport_del;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_DEL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
