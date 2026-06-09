# Threat Intelligence

**Capability ID**: `intel_threats` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_threats` is an executable APG capability package for building governed threat-intelligence applications. It gives generated APG apps a concrete runtime for lawful authority, threat workspaces, source lineage, indicators,

## Installation

```bash
pip install apg-intel-threats
```

## Provides

- `threat_authority_workflow`
- `threat_workspace_workflow`
- `threat_source_workflow`
- `threat_indicator_workflow`
- `threat_actor_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-threats/dashboard` | `intel_threats:view` | Overview |
| `/intel-threats/authorities` | `intel_threats:authorities` | Governance |
| `/intel-threats/workspaces` | `intel_threats:workspaces` | Planning |
| `/intel-threats/sources` | `intel_threats:sources` | Evidence |
| `/intel-threats/indicators` | `intel_threats:indicators` | Evidence |
| `/intel-threats/actors` | `intel_threats:actors` | Analysis |
| `/intel-threats/campaigns` | `intel_threats:campaigns` | Analysis |
| `/intel-threats/assessments` | `intel_threats:assessments` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_workspace()`
- `register_source()`
- `record_indicator()`
- `record_actor()`
- `record_campaign()`
- `record_assessment()`
- `record_report()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_threats` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_threats;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_THREATS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
