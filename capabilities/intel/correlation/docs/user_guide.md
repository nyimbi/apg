# Data Correlation

**Capability ID**: `intel_correlation` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_correlation` is an executable APG capability for governed, evidence-backed cross-source data correlation. It can be composed into generated APG applications that need entity resolution, link analysis, fraud

## Installation

```bash
pip install apg-intel-correlation
```

## Provides

- `correlation_authority_workflow`
- `correlation_workspace_workflow`
- `correlation_source_workflow`
- `correlation_entity_workflow`
- `correlation_observation_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-correlation/dashboard` | `intel_correlation:view` | Overview |
| `/intel-correlation/authorities` | `intel_correlation:authorities` | Governance |
| `/intel-correlation/workspaces` | `intel_correlation:workspaces` | Planning |
| `/intel-correlation/sources` | `intel_correlation:sources` | Data |
| `/intel-correlation/entities` | `intel_correlation:entities` | Data |
| `/intel-correlation/observations` | `intel_correlation:observations` | Data |
| `/intel-correlation/rules` | `intel_correlation:rules` | Analysis |
| `/intel-correlation/runs` | `intel_correlation:runs` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_workspace()`
- `register_source()`
- `record_entity()`
- `record_observation()`
- `record_rule()`
- `record_run()`
- `record_cluster()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_correlation` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_correlation;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_CORRELATION_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
