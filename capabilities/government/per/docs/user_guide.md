# Permits Management

**Capability ID**: `government_per` | **Domain**: `government` | **Version**: `1.0.0`

## Description

Building permits, environmental permits, conditional approvals, inspection scheduling, and compliance monitoring. Prevents construction before permit issuance, enforces occupation certificate requirements, and triggers enforcement actions on condition breaches.

## Installation

```bash
pip install apg-government-per
```

## Provides

- `permit_application_workflow`
- `permit_issuance_workflow`
- `conditional_approval_workflow`
- `inspection_scheduling_workflow`
- `permit_compliance_monitoring_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/government-per/dashboard` | `government_per:view` | Overview |
| `/government-per/applications` | `government_per:apply` | Applications |
| `/government-per/permits` | `government_per:permits` | Permits |
| `/government-per/conditions` | `government_per:conditions` | Conditions |
| `/government-per/inspections` | `government_per:inspect` | Inspections |
| `/government-per/compliance` | `government_per:compliance` | Compliance |
| `/government-per/map` | `government_per:view` | Geography |
| `/government-per/enforcement` | `government_per:enforce` | Compliance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `submit_application()`
- `apply_permit()`
- `technical_review()`
- `schedule_inspection()`
- `record_inspection()`
- `issue_permit()`
- `reject_permit()`
- `permit_renewal()`

_(See `service.py` for complete API.)_

## Interoperability

`government_per` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use government_per;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GOVERNMENT_PER_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
