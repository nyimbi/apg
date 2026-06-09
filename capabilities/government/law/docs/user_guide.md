# Law Enforcement and Justice

**Capability ID**: `government_law` | **Domain**: `government` | **Version**: `1.0.0`

## Description

Incident reporting with OB number generation, case docket management, evidence chain of custody, court scheduling, and prosecution tracking from arrest to conviction. Enforces strict chain-of-custody rules and requires DPP reference numbers before prosecution can commence.

## Installation

```bash
pip install apg-government-law
```

## Provides

- `incident_reporting_workflow`
- `docket_management_workflow`
- `evidence_chain_of_custody_workflow`
- `court_scheduling_workflow`
- `prosecution_tracking_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/government-law/dashboard` | `government_law:view` | Overview |
| `/government-law/incidents` | `government_law:incidents` | Incidents |
| `/government-law/dockets` | `government_law:dockets` | Investigations |
| `/government-law/evidence` | `government_law:evidence` | Evidence |
| `/government-law/custody` | `government_law:custody` | Evidence |
| `/government-law/court-scheduling` | `government_law:court` | Courts |
| `/government-law/prosecution` | `government_law:prosecution` | Prosecution |
| `/government-law/map` | `government_law:view` | Intelligence |

## Key Service Methods

- `describe()`
- `evaluate()`
- `report_incident()`
- `incident_report()`
- `assign_case()`
- `evidence_intake()`
- `suspect_record()`
- `arrest_record()`
- `court_scheduling()`
- `prosecution_handover()`

_(See `service.py` for complete API.)_

## Interoperability

`government_law` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use government_law;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GOVERNMENT_LAW_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
