# Incident and Case Management

**Capability ID**: `grc_icm` | **Domain**: `grc` | **Version**: `1.0.0`

## Description

Incident and Case Management provides a world-class, standalone-deployable implementation of incident and case management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Installation

```bash
pip install apg-grc-icm
```

## Provides

- `incident_lifecycle_management`
- `case_management_workflow`
- `incident_evidence_workflow`
- `regulatory_notification_workflow`
- `post_incident_review_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/grc-icm/dashboard` | `grc_icm:view` | Overview |
| `/grc-icm/incidents` | `grc_icm:manage_incidents` | Incidents |
| `/grc-icm/incidents/:id` | `grc_icm:view` | Incidents |
| `/grc-icm/cases` | `grc_icm:manage_cases` | Cases |
| `/grc-icm/cases/:id` | `grc_icm:view` | Cases |
| `/grc-icm/evidence` | `grc_icm:manage_evidence` | Evidence |
| `/grc-icm/notifications` | `grc_icm:view` | Notifications |
| `/grc-icm/timeline` | `grc_icm:view` | Investigation |

## Key Service Methods

- `_audit_event()`
- `_get_incident()`
- `report_incident()`
- `incident_triage()`
- `incident_investigation()`
- `root_cause_analysis()`
- `corrective_action()`
- `corrective_action_update()`
- `close_incident()`
- `regulatory_notification()`

_(See `service.py` for complete API.)_

## Interoperability

`grc_icm` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use grc_icm;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GRC_ICM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
