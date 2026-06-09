# Telecom Security

**Capability ID**: `telecom_sec` | **Domain**: `telecom` | **Version**: `1.0.0`

## Description

Provides comprehensive telecom security management covering fraud detection (WANGIRI, IRSF, SIM swap), SS7/Diameter signalling security, roaming security, VoIP fraud detection, lawful intercept management, security incident response, and threat intelligence sharing. Enforces strict warrant and evidence requirements throughout.

## Installation

```bash
pip install apg-telecom-sec
```

## Provides

- `fraud_management_workflow`
- `ss7_security_workflow`
- `diameter_security_workflow`
- `lawful_intercept_workflow`
- `security_incident_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-sec/dashboard` | `telecom_sec:view` | Overview |
| `/telecom-sec/fraud` | `telecom_sec:fraud` | Fraud |
| `/telecom-sec/fraud-rules` | `telecom_sec:fraud_rules` | Fraud |
| `/telecom-sec/ss7` | `telecom_sec:ss7` | Signalling Security |
| `/telecom-sec/diameter` | `telecom_sec:diameter` | Signalling Security |
| `/telecom-sec/intercept` | `telecom_sec:intercept` | Legal |
| `/telecom-sec/incidents` | `telecom_sec:incidents` | Incidents |
| `/telecom-sec/threat-intel` | `telecom_sec:threat_intel` | Intelligence |

## Key Service Methods

- `describe()`
- `evaluate()`
- `raise_fraud_case()`
- `apply_fraud_block()`
- `record_ss7_attack()`
- `record_diameter_attack()`
- `activate_intercept()`
- `update_intercept_status()`
- `open_incident()`
- `update_incident_status()`

_(See `service.py` for complete API.)_

## Interoperability

`telecom_sec` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use telecom_sec;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_SEC_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
