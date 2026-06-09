# Audit Management

**Capability ID**: `grc_aud` | **Domain**: `grc` | **Version**: `1.0.0`

## Description

Audit Management provides a world-class, standalone-deployable implementation of audit management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Installation

```bash
pip install apg-grc-aud
```

## Provides

- `audit_program_lifecycle`
- `audit_finding_lifecycle`
- `audit_evidence_workflow`
- `audit_report_workflow`
- `audit_dashboard_service`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/grc-aud/dashboard` | `grc_aud:view` | Overview |
| `/grc-aud/audits` | `grc_aud:manage_audits` | Audits |
| `/grc-aud/audits/:id` | `grc_aud:view` | Audits |
| `/grc-aud/findings` | `grc_aud:manage_findings` | Findings |
| `/grc-aud/findings/:id` | `grc_aud:view` | Findings |
| `/grc-aud/evidence` | `grc_aud:manage_evidence` | Evidence |
| `/grc-aud/reports` | `grc_aud:manage_reports` | Reports |
| `/grc-aud/calendar` | `grc_aud:view` | Planning |

## Key Service Methods

- `_audit_event()`
- `_get_engagement()`
- `_get_finding()`
- `create_audit_plan()`
- `create_audit_engagement()`
- `fieldwork_record()`
- `draft_audit_report()`
- `management_response()`
- `finalise_report()`
- `issue_tracking()`

_(See `service.py` for complete API.)_

## Interoperability

`grc_aud` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use grc_aud;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GRC_AUD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
