# Quality of Service

**Capability ID**: `telecom_qos` | **Domain**: `telecom` | **Version**: `1.0.0`

## Description

QoS policy management and enforcement covering bearer QoS, traffic shaping and policing, SLA parameter measurement, real-time degradation detection with root cause analysis, automated and manual remediation workflows, and PCRF/PCEF integration for policy enforcement on network elements.

## Installation

```bash
pip install apg-telecom-qos
```

## Provides

- `qos_policy_management_workflow`
- `traffic_prioritisation_workflow`
- `sla_enforcement_workflow`
- `degradation_detection_workflow`
- `root_cause_analysis_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-qos/dashboard` | `telecom_qos:view` | Overview |
| `/telecom-qos/policies` | `telecom_qos:policies` | Policy |
| `/telecom-qos/policies/<id>` | `telecom_qos:policies` | Policy |
| `/telecom-qos/traffic` | `telecom_qos:traffic` | Traffic |
| `/telecom-qos/enforcement` | `telecom_qos:enforcement` | Operations |
| `/telecom-qos/sla` | `telecom_qos:sla` | SLA |
| `/telecom-qos/degradation` | `telecom_qos:degradation` | Monitoring |
| `/telecom-qos/root-cause` | `telecom_qos:degradation` | Monitoring |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_qos_policy()`
- `change_qos_policy()`
- `classify_traffic()`
- `update_enforcement_status()`
- `record_sla_measurement()`
- `record_degradation()`
- `record_root_cause()`
- `trigger_remediation()`

_(See `service.py` for complete API.)_

## Interoperability

`telecom_qos` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use telecom_qos;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_QOS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
