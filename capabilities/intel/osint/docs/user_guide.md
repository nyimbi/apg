# Open Source Intelligence

**Capability ID**: `intel_osint` | **Domain**: `intel` | **Version**: `2.0.0`

## Description

`intel_osint` is the APG package-backed capability for governed open-source intelligence applications. It composes requirements, sources, collection plans, evidence, triage, assessments, dissemination, reviews, Bytewax lifecycle

## Installation

```bash
pip install apg-intel-osint
```

## Provides

- `osint_source_workflow`
- `osint_collection_task_workflow`
- `osint_raw_intel_workflow`
- `osint_processed_intel_workflow`
- `osint_entity_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `intel_crawler`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-osint/dashboard` | `intel_osint:view` | Overview |
| `/intel-osint/sources` | `intel_osint:sources` | Collection |
| `/intel-osint/tasks` | `intel_osint:tasks` | Collection |
| `/intel-osint/raw-intel` | `intel_osint:raw_intel` | Processing |
| `/intel-osint/triage` | `intel_osint:triage` | Processing |
| `/intel-osint/processed-intel` | `intel_osint:processed_intel` | Analysis |
| `/intel-osint/entities` | `intel_osint:entities` | Analysis |
| `/intel-osint/relationships` | `intel_osint:relationships` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate_rules()`
- `register_source()`
- `update_source()`
- `get_source()`
- `list_sources()`
- `delete_source()`
- `create_task()`
- `start_task()`
- `complete_task()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_osint` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_osint;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_OSINT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
