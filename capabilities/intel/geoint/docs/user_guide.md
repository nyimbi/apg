# Geospatial Intelligence

**Capability ID**: `intel_geoint` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_geoint` is the APG package-backed capability for governed geospatial intelligence applications. It composes authorities, areas of interest, imagery/geospatial sources, collection plans, observations, features, change

## Installation

```bash
pip install apg-intel-geoint
```

## Provides

- `geoint_authority_workflow`
- `geoint_area_workflow`
- `geoint_source_workflow`
- `geoint_collection_workflow`
- `geoint_observation_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-geoint/dashboard` | `intel_geoint:view` | Overview |
| `/intel-geoint/authorities` | `intel_geoint:authorities` | Governance |
| `/intel-geoint/areas` | `intel_geoint:areas` | Planning |
| `/intel-geoint/sources` | `intel_geoint:sources` | Collection |
| `/intel-geoint/collection-plans` | `intel_geoint:collection` | Collection |
| `/intel-geoint/observations` | `intel_geoint:observations` | Processing |
| `/intel-geoint/features` | `intel_geoint:features` | Analysis |
| `/intel-geoint/changes` | `intel_geoint:changes` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_area()`
- `register_source()`
- `record_collection_plan()`
- `record_observation()`
- `record_feature()`
- `record_change()`
- `record_assessment()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_geoint` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_geoint;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_GEOINT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
