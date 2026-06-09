# Quality Management System

**Capability ID**: `pharma_qms` | **Domain**: `pharma` | **Version**: `1.0.0`

## Description

End-to-end pharmaceutical QMS covering change control, CAPA management, deviation handling, controlled document management, audit management, validation lifecycle, and risk assessment. All workflows enforce GMP compliance, electronic signature requirements, and effectiveness check obligations before closure.

## Installation

```bash
pip install apg-pharma-qms
```

## Provides

- `change_control_workflow`
- `capa_management_workflow`
- `deviation_management_workflow`
- `document_control_workflow`
- `audit_management_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-qms/dashboard` | `pharma_qms:view` | Overview |
| `/pharma-qms/change-control` | `pharma_qms:change_control` | Change Control |
| `/pharma-qms/change-control/<id>` | `pharma_qms:change_control` | Change Control |
| `/pharma-qms/capa` | `pharma_qms:capa` | CAPA |
| `/pharma-qms/capa/<id>` | `pharma_qms:capa` | CAPA |
| `/pharma-qms/deviations` | `pharma_qms:deviations` | Deviations |
| `/pharma-qms/documents` | `pharma_qms:documents` | Document Control |
| `/pharma-qms/documents/<id>` | `pharma_qms:documents` | Document Control |

## Key Service Methods

- `describe()`
- `evaluate()`
- `initiate_change()`
- `approve_change()`
- `implement_change()`
- `close_change()`
- `list_changes()`
- `create_capa()`
- `close_capa()`
- `check_overdue_capas()`

_(See `service.py` for complete API.)_

## Interoperability

`pharma_qms` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pharma_qms;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PHARMA_QMS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
