# IoT Device Integration

**Capability ID**: `iotd` | **Domain**: `common` | **Version**: `1.0.0`

## Description

IOTD provides APG applications with a tenant-scoped device-operations runtime: device identity, certificate ownership, fleet grouping, encrypted telemetry ingestion, governed command dispatch, command acknowledgement, signed firmware

## Installation

```bash
pip install apg-common-iotd
```

## Provides

- `device_registry`
- `telemetry_ingestion`
- `command_dispatch`
- `firmware_lifecycle`
- `device_security`

## Requires

- `auth`
- `encr`
- `audl`
- `conf`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/iotd/dashboard` | `iotd:view` | Overview |
| `/iotd/devices` | `iotd:register` | Devices |
| `/iotd/telemetry` | `iotd:view` | Telemetry |
| `/iotd/commands` | `iotd:command` | Control |
| `/iotd/firmware` | `iotd:manage_firmware` | Lifecycle |
| `/iotd/agents` | `iotd:admin` | Operations |
| `/iotd/health` | `iotd:view` | Operations |
| `/iotd/security` | `iotd:admin` | Security |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_device()`
- `ingest_telemetry()`
- `dispatch_command()`
- `acknowledge_command()`
- `register_firmware()`
- `deploy_firmware()`
- `health_report()`
- `register_iotd_agent()`

_(See `service.py` for complete API.)_

## Interoperability

`iotd` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use iotd;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `IOTD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
