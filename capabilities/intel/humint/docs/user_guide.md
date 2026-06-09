# Human Intelligence

**Capability ID**: `intel_humint` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_humint` is the APG package-backed capability for governed human-intelligence applications. It composes authorities, human sources, contact plans, contact reports, debriefings, reliability assessments, leads,

## Installation

```bash
pip install apg-intel-humint
```

## Provides

- `humint_authority_workflow`
- `humint_source_workflow`
- `humint_contact_plan_workflow`
- `humint_contact_report_workflow`
- `humint_debriefing_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-humint/dashboard` | `intel_humint:view` | Overview |
| `/intel-humint/authorities` | `intel_humint:authorities` | Governance |
| `/intel-humint/sources` | `intel_humint:sources` | Source Management |
| `/intel-humint/contact-plans` | `intel_humint:contacts` | Operations |
| `/intel-humint/contact-reports` | `intel_humint:reports` | Operations |
| `/intel-humint/debriefings` | `intel_humint:analysis` | Analysis |
| `/intel-humint/reliability` | `intel_humint:analysis` | Analysis |
| `/intel-humint/leads` | `intel_humint:leads` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `register_source()`
- `record_contact_plan()`
- `record_contact_report()`
- `record_debriefing()`
- `record_reliability()`
- `record_lead()`
- `record_dissemination()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_humint` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_humint;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_HUMINT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
