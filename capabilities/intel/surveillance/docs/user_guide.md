# Digital Surveillance

**Capability ID**: `intel_surveillance` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_surveillance` is an executable APG capability for lawful, defensive digital-surveillance workflows. It can be composed into generated APG applications that need facility monitoring, endpoint telemetry review,

## Installation

```bash
pip install apg-intel-surveillance
```

## Provides

- `surveillance_authority_workflow`
- `surveillance_program_workflow`
- `surveillance_asset_workflow`
- `surveillance_sensor_workflow`
- `surveillance_observation_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `cvsn`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-surveillance/dashboard` | `intel_surveillance:view` | Overview |
| `/intel-surveillance/authorities` | `intel_surveillance:authorities` | Governance |
| `/intel-surveillance/programs` | `intel_surveillance:programs` | Planning |
| `/intel-surveillance/assets` | `intel_surveillance:assets` | Assets |
| `/intel-surveillance/sensors` | `intel_surveillance:sensors` | Collection |
| `/intel-surveillance/observations` | `intel_surveillance:observations` | Collection |
| `/intel-surveillance/alerts` | `intel_surveillance:alerts` | Analysis |
| `/intel-surveillance/risk` | `intel_surveillance:risk` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_program()`
- `record_asset()`
- `register_sensor()`
- `record_observation()`
- `record_alert()`
- `record_risk()`
- `record_referral()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_surveillance` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_surveillance;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_SURVEILLANCE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
