# Consent and Privacy Management

**Capability ID**: `cons` | **Domain**: `common` | **Version**: `1.0.0`

## Description

CONS is the APG capability for governed consent, privacy preferences, privacy requests, consent-gated processing, and auditable privacy operations. It lets generated APG applications publish notices, register lawful purposes, capture

## Installation

```bash
pip install apg-common-cons
```

## Provides

- `purpose_registry`
- `consent_capture`
- `privacy_requests`
- `preference_center`
- `privacy_audit`

## Requires

- `comp`
- `auth`
- `dlpd`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/cons/dashboard` | `cons:view` | Overview |
| `/cons/purposes` | `cons:manage_purposes` | Policy |
| `/cons/notices` | `cons:manage_purposes` | Policy |
| `/cons/consents` | `cons:view` | Consent |
| `/cons/requests` | `cons:process_requests` | Requests |
| `/cons/preferences` | `cons:capture` | Consent |
| `/cons/agents` | `cons:process_requests` | Agents |
| `/cons/analytics` | `cons:view` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `publish_notice()`
- `create_purpose()`
- `capture_consent()`
- `withdraw_consent()`
- `update_preferences()`
- `process_consent_gated_data()`
- `submit_privacy_request()`
- `complete_privacy_request()`

_(See `service.py` for complete API.)_

## Interoperability

`cons` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use cons;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `CONS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
