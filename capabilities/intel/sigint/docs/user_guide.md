# Signals Intelligence

**Capability ID**: `intel_sigint` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_sigint` is the APG package-backed capability for governed signals-intelligence applications. It composes authorities, sources, collection tasks, observations, processing batches, patterns, assessments, reviews, Bytewax

## Installation

```bash
pip install apg-intel-sigint
```

## Provides

- `sigint_authority_workflow`
- `sigint_source_workflow`
- `sigint_collection_workflow`
- `sigint_observation_workflow`
- `sigint_processing_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `intel_radio`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-sigint/dashboard` | `intel_sigint:view` | Overview |
| `/intel-sigint/authorities` | `intel_sigint:authorities` | Governance |
| `/intel-sigint/sources` | `intel_sigint:sources` | Collection |
| `/intel-sigint/collection-tasks` | `intel_sigint:collection` | Collection |
| `/intel-sigint/observations` | `intel_sigint:observations` | Processing |
| `/intel-sigint/processing` | `intel_sigint:processing` | Processing |
| `/intel-sigint/patterns` | `intel_sigint:patterns` | Analysis |
| `/intel-sigint/assessments` | `intel_sigint:assessments` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `register_source()`
- `record_collection_task()`
- `record_observation()`
- `record_processing_batch()`
- `record_pattern()`
- `record_assessment()`
- `record_review()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_sigint` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_sigint;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_SIGINT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
