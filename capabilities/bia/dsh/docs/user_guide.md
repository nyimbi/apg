# Dashboard Management

**Capability ID**: `bia_dsh` | **Domain**: `bia` | **Version**: `1.0.0`

## Description

Dashboard Management (bia_dsh) provides dynamic dashboard creation, a widget library with 15 chart types, real-time data binding, responsive layout engines, cross-widget filtering, scheduled snapshot capture, and governed sharing — all tenant-scoped and audit-logged.

## Installation

```bash
pip install apg-bia-dsh
```

## Provides

- `dashboard_creation`
- `widget_library`
- `real_time_data_binding`
- `responsive_layout_engine`
- `scheduled_snapshots`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `schd`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bia/dsh/` | `bia_dsh:view` | Overview |
| `/bia/dsh/gallery` | `bia_dsh:view` | Dashboards |
| `/bia/dsh/<id>/view` | `bia_dsh:view` | Dashboards |
| `/bia/dsh/<id>/build` | `bia_dsh:edit` | Dashboards |
| `/bia/dsh/new` | `bia_dsh:create` | Dashboards |
| `/bia/dsh/widgets` | `bia_dsh:view` | Widgets |
| `/bia/dsh/widgets/<id>` | `bia_dsh:view` | Widgets |
| `/bia/dsh/widgets/new` | `bia_dsh:edit` | Widgets |

## Key Service Methods

- `describe()`
- `create_dashboard()`
- `get_dashboard()`
- `list_dashboards()`
- `update_dashboard()`
- `publish_dashboard()`
- `archive_dashboard()`
- `delete_dashboard()`
- `refresh_dashboard()`
- `share_dashboard()`

_(See `service.py` for complete API.)_

## Interoperability

`bia_dsh` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bia_dsh;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BIA_DSH_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
