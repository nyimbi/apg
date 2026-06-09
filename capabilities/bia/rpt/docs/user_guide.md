# Report Builder

**Capability ID**: `bia_rpt` | **Domain**: `bia` | **Version**: `1.0.0`

## Description

The Report Builder capability (bia_rpt) provides parameterised report authoring, multi-format export (PDF/Excel/CSV/HTML/DOCX), report scheduling with 7 frequency options, governed distribution across 7 channels with external-distribution approval, run history, and a complete audit trail.

## Installation

```bash
pip install apg-bia-rpt
```

## Provides

- `parameterised_report_authoring`
- `report_scheduling`
- `report_distribution`
- `multi_format_export`
- `report_audit_trail`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `schd`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bia/rpt/dashboard` | `bia_rpt:view` | Overview |
| `/bia/rpt/reports` | `bia_rpt:view` | Reports |
| `/bia/rpt/reports/<id>` | `bia_rpt:view` | Reports |
| `/bia/rpt/reports/<id>/build` | `bia_rpt:edit` | Reports |
| `/bia/rpt/reports/new` | `bia_rpt:create` | Reports |
| `/bia/rpt/reports/<id>/run` | `bia_rpt:run` | Reports |
| `/bia/rpt/schedules` | `bia_rpt:schedule` | Scheduling |
| `/bia/rpt/schedules/<id>` | `bia_rpt:schedule` | Scheduling |

## Key Service Methods

- `describe()`
- `create_report()`
- `get_report()`
- `list_reports()`
- `update_report()`
- `publish_report()`
- `archive_report()`
- `delete_report()`
- `add_column()`
- `list_columns()`

_(See `service.py` for complete API.)_

## Interoperability

`bia_rpt` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bia_rpt;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BIA_RPT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
