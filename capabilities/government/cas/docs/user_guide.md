# Case Management

**Capability ID**: `government_cas` | **Domain**: `government` | **Version**: `1.0.0`

## Description

Citizen case intake, assignment, workflow routing, SLA tracking, escalation, and outcome recording for government service delivery. Handles complaints, enquiries, applications, and regulatory referrals across all intake channels with full audit trail.

## Installation

```bash
pip install apg-government-cas
```

## Provides

- `case_intake_workflow`
- `case_assignment_workflow`
- `case_routing_workflow`
- `sla_tracking_workflow`
- `case_escalation_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/government-cas/dashboard` | `government_cas:view` | Overview |
| `/government-cas/intake` | `government_cas:create` | Intake |
| `/government-cas/cases` | `government_cas:cases` | Cases |
| `/government-cas/assignments` | `government_cas:assign` | Operations |
| `/government-cas/escalations` | `government_cas:escalate` | Operations |
| `/government-cas/sla` | `government_cas:sla` | Monitoring |
| `/government-cas/outcomes` | `government_cas:outcomes` | Resolution |
| `/government-cas/notifications` | `government_cas:notify` | Communications |

## Key Service Methods

- `describe()`
- `evaluate()`
- `open_case()`
- `create_case()`
- `assign_officer()`
- `case_update()`
- `schedule_hearing()`
- `record_decision()`
- `close_case()`
- `appeal_management()`

_(See `service.py` for complete API.)_

## Interoperability

`government_cas` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use government_cas;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GOVERNMENT_CAS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
