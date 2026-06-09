# Accessibility Services

**Capability ID**: `accs` | **Domain**: `common` | **Version**: `1.0.0`

## Description

ACCS makes accessibility governance an executable APG capability. It gives generated applications a tenant-scoped way to register accessibility standards, register UI/content/media targets, run deterministic audits, record findings,

## Installation

```bash
pip install apg-common-accs
```

## Provides

- `accessibility_audits`
- `remediation_workflows`
- `accessibility_exceptions`
- `assistive_metadata`
- `media_accessibility`

## Requires

- `them`
- `i18n`
- `nlpc`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/accs/dashboard` | `accs:view` | Overview |
| `/accs/audits` | `accs:audit` | Audits |
| `/accs/findings` | `accs:view` | Audits |
| `/accs/remediation` | `accs:remediate` | Remediation |
| `/accs/exceptions` | `accs:review` | Governance |
| `/accs/assistive` | `accs:audit` | Assistive |
| `/accs/media` | `accs:remediate` | Content |
| `/accs/compliance` | `accs:manage_standards` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_standard()`
- `list_standards()`
- `register_target()`
- `list_targets()`
- `list_records()`
- `create_record()`
- `run_audit()`
- `list_audits()`

_(See `service.py` for complete API.)_

## Interoperability

`accs` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use accs;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ACCS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
