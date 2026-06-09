# Network Management

**Capability ID**: `telecom_net` | **Domain**: `telecom` | **Version**: `1.0.0`

## Description

Network operations centre capability providing fault management with alarm correlation, performance monitoring with threshold alerting, configuration change management with freeze period enforcement, SLA monitoring, and NOC shift handover management. Designed for 24×7 NOC operations with a dark-themed UI.

## Installation

```bash
pip install apg-telecom-net
```

## Provides

- `fault_management_workflow`
- `performance_management_workflow`
- `configuration_management_workflow`
- `sla_monitoring_workflow`
- `noc_operations_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-net/dashboard` | `telecom_net:view` | Overview |
| `/telecom-net/alarms` | `telecom_net:faults` | Fault Management |
| `/telecom-net/faults` | `telecom_net:faults` | Fault Management |
| `/telecom-net/performance` | `telecom_net:performance` | Performance |
| `/telecom-net/config-changes` | `telecom_net:config` | Configuration |
| `/telecom-net/sla` | `telecom_net:sla` | SLA |
| `/telecom-net/noc` | `telecom_net:noc` | NOC |
| `/telecom-net/topology` | `telecom_net:view` | Overview |

## Key Service Methods

- `describe()`
- `evaluate()`
- `raise_alarm()`
- `update_alarm_status()`
- `suppress_alarm()`
- `open_fault_ticket()`
- `resolve_fault_ticket()`
- `escalate_fault()`
- `record_performance()`
- `submit_config_change()`

_(See `service.py` for complete API.)_

## Interoperability

`telecom_net` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use telecom_net;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_NET_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
