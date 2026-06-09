# Dark Web Monitoring

**Capability ID**: `intel_darkweb` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_darkweb` is an executable APG capability for lawful, defensive dark-web-monitoring workflows. It can be composed into generated APG applications that need exposure monitoring, fraud-market intelligence,

## Installation

```bash
pip install apg-intel-darkweb
```

## Provides

- `darkweb_authority_workflow`
- `darkweb_program_workflow`
- `darkweb_source_workflow`
- `darkweb_observation_workflow`
- `darkweb_indicator_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-darkweb/dashboard` | `intel_darkweb:view` | Overview |
| `/intel-darkweb/authorities` | `intel_darkweb:authorities` | Governance |
| `/intel-darkweb/programs` | `intel_darkweb:programs` | Planning |
| `/intel-darkweb/sources` | `intel_darkweb:sources` | Collection |
| `/intel-darkweb/observations` | `intel_darkweb:observations` | Collection |
| `/intel-darkweb/indicators` | `intel_darkweb:indicators` | Analysis |
| `/intel-darkweb/marketplace-risk` | `intel_darkweb:marketplace_risk` | Analysis |
| `/intel-darkweb/threat-actors` | `intel_darkweb:threat_actors` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_program()`
- `register_source()`
- `record_observation()`
- `record_indicator()`
- `record_marketplace_risk()`
- `record_threat_actor()`
- `record_referral()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_darkweb` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_darkweb;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_DARKWEB_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
