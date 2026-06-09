# Cyber Intelligence

**Capability ID**: `intel_cybint` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_cybint` is the APG package-backed capability for governed defensive cyber-intelligence applications. It composes authorities, indicators, sightings, enrichment, threat profiles, risk assessments, incident links, dissemination,

## Installation

```bash
pip install apg-intel-cybint
```

## Provides

- `cybint_authority_workflow`
- `cybint_indicator_workflow`
- `cybint_sighting_workflow`
- `cybint_enrichment_workflow`
- `cybint_threat_profile_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-cybint/dashboard` | `intel_cybint:view` | Overview |
| `/intel-cybint/authorities` | `intel_cybint:authorities` | Governance |
| `/intel-cybint/indicators` | `intel_cybint:indicators` | Intelligence |
| `/intel-cybint/sightings` | `intel_cybint:sightings` | Intelligence |
| `/intel-cybint/enrichment` | `intel_cybint:enrichment` | Analysis |
| `/intel-cybint/profiles` | `intel_cybint:profiles` | Analysis |
| `/intel-cybint/risk` | `intel_cybint:risk` | Analysis |
| `/intel-cybint/incidents` | `intel_cybint:incidents` | Response |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_indicator()`
- `record_sighting()`
- `record_enrichment()`
- `record_profile()`
- `record_risk()`
- `record_incident_link()`
- `record_dissemination()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_cybint` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_cybint;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_CYBINT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
