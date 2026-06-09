# Smart Metering & AMI

**Capability ID**: `energy_met` | **Domain**: `energy` | **Version**: `1.0.0`

## Description

Smart Metering & AMI manages the full lifecycle of advanced metering infrastructure from meter registration through interval data collection, tamper detection with evidence workflows, remote connect/disconnect with approval controls, demand response event coordination with customer opt-out, and data quality flagging. It also monitors AMI head-end connectivity ratios across communication technologies.

## Installation

```bash
pip install apg-energy-met
```

## Provides

- `meter_registry`
- `ami_head_end_management`
- `interval_data_collection`
- `tamper_detection`
- `remote_connect_disconnect`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/energy-met/dashboard` | `energy_met:view` | Overview |
| `/energy-met/meters` | `energy_met:meters` | Assets |
| `/energy-met/meters/<id>` | `energy_met:meters` | Assets |
| `/energy-met/readings` | `energy_met:readings` | Data |
| `/energy-met/tamper` | `energy_met:tamper` | Security |
| `/energy-met/commands` | `energy_met:commands` | Operations |
| `/energy-met/demand-response` | `energy_met:demand_response` | Programs |
| `/energy-met/data-quality` | `energy_met:data_quality` | Quality |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_meter()`
- `update_meter_status()`
- `list_meters()`
- `get_meter()`
- `submit_reading()`
- `list_readings()`
- `report_tamper()`
- `resolve_tamper()`

_(See `service.py` for complete API.)_

## Interoperability

`energy_met` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use energy_met;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ENERGY_MET_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
