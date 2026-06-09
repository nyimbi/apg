# Licensing and Permits

**Capability ID**: `government_lic` | **Domain**: `government` | **Version**: `1.0.0`

## Description

Business and professional licence applications, renewals, inspections, revocations, and fee collection with full compliance monitoring. Enforces that licences cannot be renewed if the last inspection failed, prevents duplicate licences, and requires formal notice before revocation.

## Installation

```bash
pip install apg-government-lic
```

## Provides

- `licence_application_workflow`
- `licence_issuance_workflow`
- `inspection_scheduling_workflow`
- `licence_renewal_workflow`
- `fee_collection_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/government-lic/dashboard` | `government_lic:view` | Overview |
| `/government-lic/applications` | `government_lic:apply` | Applications |
| `/government-lic/licences` | `government_lic:licences` | Licences |
| `/government-lic/inspections` | `government_lic:inspect` | Inspections |
| `/government-lic/renewals` | `government_lic:renew` | Renewals |
| `/government-lic/fees` | `government_lic:fees` | Payments |
| `/government-lic/revocations` | `government_lic:revoke` | Compliance |
| `/government-lic/compliance` | `government_lic:compliance` | Compliance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `submit_application()`
- `apply_licence()`
- `background_check()`
- `premises_inspection()`
- `issue_licence()`
- `renew_licence()`
- `licence_renewal()`
- `suspend_licence()`

_(See `service.py` for complete API.)_

## Interoperability

`government_lic` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use government_lic;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GOVERNMENT_LIC_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
