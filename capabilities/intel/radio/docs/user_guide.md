# Radio Intelligence Listener

**Capability ID**: `intel_radio` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_radio` is an executable APG capability for lawful, passive radio-monitoring workflows. It can be composed into generated APG applications that need public-safety monitoring, spectrum management, interference review,

## Installation

```bash
pip install apg-intel-radio
```

## Provides

- `radio_authority_workflow`
- `radio_band_plan_workflow`
- `radio_receiver_workflow`
- `radio_collection_session_workflow`
- `radio_observation_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-radio/dashboard` | `intel_radio:view` | Overview |
| `/intel-radio/authorities` | `intel_radio:authorities` | Governance |
| `/intel-radio/band-plans` | `intel_radio:band_plans` | Planning |
| `/intel-radio/receivers` | `intel_radio:receivers` | Collection |
| `/intel-radio/sessions` | `intel_radio:sessions` | Collection |
| `/intel-radio/observations` | `intel_radio:observations` | Signals |
| `/intel-radio/classifications` | `intel_radio:classifications` | Analysis |
| `/intel-radio/events` | `intel_radio:events` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_band_plan()`
- `register_receiver()`
- `record_session()`
- `record_observation()`
- `record_classification()`
- `record_event()`
- `record_referral()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_radio` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_radio;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_RADIO_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
