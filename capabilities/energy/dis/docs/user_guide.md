# Distribution Network

**Capability ID**: `energy_dis` | **Domain**: `energy` | **Version**: `1.0.0`

## Description

Distribution Network manages the complete operational lifecycle of electricity distribution infrastructure. It provides network topology management for feeders and equipment, real-time fault detection and isolation, switching order workflows with live-network safety controls, outage recording with SAIDI/SAIFI reliability tracking, SCADA telemetry ingestion across multiple protocols, and automated load balancing with voltage constraint enforcement.

## Installation

```bash
pip install apg-energy-dis
```

## Provides

- `network_topology_management`
- `fault_detection_and_isolation`
- `outage_restoration`
- `switching_order_management`
- `scada_integration`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/energy-dis/dashboard` | `energy_dis:view` | Overview |
| `/energy-dis/topology` | `energy_dis:topology` | Network |
| `/energy-dis/elements` | `energy_dis:topology` | Network |
| `/energy-dis/faults` | `energy_dis:faults` | Operations |
| `/energy-dis/faults/<id>` | `energy_dis:faults` | Operations |
| `/energy-dis/switching` | `energy_dis:switching` | Operations |
| `/energy-dis/outages` | `energy_dis:outages` | Operations |
| `/energy-dis/scada` | `energy_dis:scada` | Monitoring |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_feeder()`
- `list_feeders()`
- `register_element()`
- `list_elements()`
- `report_fault()`
- `isolate_fault()`
- `restore_fault()`
- `dispatch_crew()`

_(See `service.py` for complete API.)_

## Interoperability

`energy_dis` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use energy_dis;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ENERGY_DIS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
