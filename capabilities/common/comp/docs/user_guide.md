# Compliance Management

**Capability ID**: `comp` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`comp` is APG's package-backed Compliance Management capability. It gives generated applications a tenant-scoped compliance runtime for frameworks, obligations, controls, encrypted evidence, assessments, findings, remediation,

## Installation

```bash
pip install apg-common-comp
```

## Provides

_(see capability contract)_

## Requires

_(none)_

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/comp/dashboard` | `comp:view` | Overview |
| `/comp/frameworks` | `comp:manage_controls` | Frameworks |
| `/comp/controls` | `comp:manage_controls` | Controls |
| `/comp/evidence` | `comp:collect_evidence` | Evidence |
| `/comp/assessments` | `comp:manage_controls` | Assurance |
| `/comp/findings` | `comp:remediate` | Remediation |
| `/comp/exceptions` | `comp:remediate` | Remediation |
| `/comp/reports` | `comp:approve_reports` | Reporting |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_framework()`
- `create_control()`
- `record_evidence()`
- `assess_control()`
- `open_finding()`
- `resolve_finding()`
- `escalate_overdue_findings()`
- `prepare_report()`

_(See `service.py` for complete API.)_

## Interoperability

`comp` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use comp;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `COMP_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
