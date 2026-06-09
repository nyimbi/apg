# Citizen Services Portal

**Capability ID**: `government_csr` | **Domain**: `government` | **Version**: `1.0.0`

## Description

Self-service citizen portal supporting application submission, status tracking, e-payment, document verification, and service delivery analytics. Provides a unified interface for all government-to-citizen service transactions across web, mobile, USSD, and kiosk channels.

## Installation

```bash
pip install apg-government-csr
```

## Provides

- `citizen_self_service_workflow`
- `service_application_workflow`
- `application_status_tracking_workflow`
- `epayment_workflow`
- `document_verification_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/government-csr/dashboard` | `government_csr:view` | Overview |
| `/government-csr/services` | `government_csr:services` | Services |
| `/government-csr/apply` | `government_csr:apply` | Services |
| `/government-csr/applications` | `government_csr:applications` | Applications |
| `/government-csr/payments` | `government_csr:payments` | Payments |
| `/government-csr/verifications` | `government_csr:verify` | Verification |
| `/government-csr/notifications` | `government_csr:notify` | Communications |
| `/government-csr/analytics` | `government_csr:analytics` | Reporting |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_service()`
- `submit_application()`
- `submit_service_request()`
- `track_application()`
- `schedule_appointment()`
- `citizen_portal_login()`
- `document_verification_request()`
- `payment_for_service()`

_(See `service.py` for complete API.)_

## Interoperability

`government_csr` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use government_csr;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GOVERNMENT_CSR_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
