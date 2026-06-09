# Emergency Management

**Capability ID**: `government_eme` | **Domain**: `government` | **Version**: `1.0.0`

## Description

Incident command, resource mobilisation, multi-agency coordination, EOC management, situation reporting, and after-action reviews. Implements the Incident Command System (ICS) framework with mandatory after-action reviews and strict EOC activation authority controls.

## Installation

```bash
pip install apg-government-eme
```

## Provides

- `incident_command_workflow`
- `resource_mobilisation_workflow`
- `multi_agency_coordination_workflow`
- `eoc_management_workflow`
- `situation_reporting_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/government-eme/dashboard` | `government_eme:view` | Overview |
| `/government-eme/incidents` | `government_eme:incidents` | Incidents |
| `/government-eme/resources` | `government_eme:resources` | Resources |
| `/government-eme/agencies` | `government_eme:agencies` | Coordination |
| `/government-eme/eoc` | `government_eme:eoc` | Command |
| `/government-eme/situation-reports` | `government_eme:reports` | Reporting |
| `/government-eme/map` | `government_eme:view` | Situational Awareness |
| `/government-eme/after-action` | `government_eme:aar` | Learning |

## Key Service Methods

- `describe()`
- `evaluate()`
- `declare_incident()`
- `declare_emergency()`
- `activate_eoc()`
- `resource_mobilisation()`
- `multi_agency_coordination()`
- `situation_report()`
- `evacuation_management()`
- `relief_distribution()`

_(See `service.py` for complete API.)_

## Interoperability

`government_eme` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use government_eme;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GOVERNMENT_EME_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
