# Medical Device Management

**Capability ID**: `healthcare_dev` | **Domain**: `healthcare` | **Version**: `1.0.0`

## Description

Medical device lifecycle management covering device inventory with FDA UDI tracking, preventive and corrective maintenance scheduling with work orders, calibration record management, and adverse event reporting. Enforces UDI requirements for Class II/III devices, blocks use of recalled or calibration-overdue devices, and automatically escalates serious adverse events.

## Installation

```bash
pip install apg-healthcare-dev
```

## Provides

- `device_inventory_management`
- `maintenance_schedule_management`
- `calibration_record_tracking`
- `fda_udi_tracking`
- `adverse_event_reporting`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/healthcare-dev/dashboard` | `healthcare_dev:view` | Overview |
| `/healthcare-dev/inventory` | `healthcare_dev:inventory` | Devices |
| `/healthcare-dev/inventory/register` | `healthcare_dev:inventory_write` | Devices |
| `/healthcare-dev/inventory/<id>` | `healthcare_dev:inventory` | Devices |
| `/healthcare-dev/maintenance` | `healthcare_dev:maintenance` | Maintenance |
| `/healthcare-dev/work-orders` | `healthcare_dev:maintenance` | Maintenance |
| `/healthcare-dev/calibration` | `healthcare_dev:calibration` | Calibration |
| `/healthcare-dev/adverse-events` | `healthcare_dev:adverse_events` | Safety |

## Key Service Methods

- `describe()`
- `register_device()`
- `update_device_status()`
- `get_device()`
- `list_devices()`
- `device_inventory()`
- `udi_lookup()`
- `schedule_maintenance()`
- `maintenance_schedule()`
- `log_maintenance()`

_(See `service.py` for complete API.)_

## Interoperability

`healthcare_dev` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use healthcare_dev;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `HEALTHCARE_DEV_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
