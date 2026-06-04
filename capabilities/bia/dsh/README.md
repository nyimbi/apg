# Dashboard Management

## Overview
Dashboard Management (bia_dsh) provides dynamic dashboard creation, a widget library with 15 chart types, real-time data binding, responsive layout engines, cross-widget filtering, scheduled snapshot capture, and governed sharing — all tenant-scoped and audit-logged.

## Capability ID
`bia_dsh`

## Provides
- dashboard_creation: Draft, publish, archive lifecycle for dashboards
- widget_library: 15 widget types with datasource binding
- real_time_data_binding: Configurable per-widget refresh intervals
- responsive_layout_engine: Grid, freeform, tabbed, stacked layouts
- scheduled_snapshots: Cron-triggered PNG/PDF/HTML snapshots
- cross_widget_filtering: Dashboard-wide filter propagation
- dashboard_sharing: Team and organisation-scoped access
- dashboard_export: Export to PNG, PDF, HTML, JSON
- dashboard_embedding: Embed tokens for portal integration

## Requires
| Capability | Reason |
|------------|--------|
| auth | User identity and permission checks |
| audl | Audit trail for views and edits |
| mten | Tenant context enforcement |
| conf | Runtime configuration |
| schd | Snapshot scheduling |
| mqeb | Streaming dashboard lifecycle events |
| ntfy | Snapshot delivery notifications |
| bia_anl | Metric and query datasource resolution |

## Configuration
| Option | Default | Description |
|--------|---------|-------------|
| max_widgets_per_dashboard | 50 | Hard widget limit |
| max_filters_per_dashboard | 20 | Hard filter limit |
| snapshot_retention_days | 90 | Days to retain snapshots |
| default_layout | responsive_grid | Default layout type |
| require_approval_for_public | true | Public dashboards need approval |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/dsh/dashboards | GET | List dashboards | bia_dsh:view |
| /api/bia/dsh/dashboards | POST | Create dashboard | bia_dsh:create |
| /api/bia/dsh/dashboards/<id> | GET | Get dashboard | bia_dsh:view |
| /api/bia/dsh/dashboards/<id> | PUT | Update dashboard | bia_dsh:edit |
| /api/bia/dsh/dashboards/<id> | DELETE | Delete dashboard | bia_dsh:edit |
| /api/bia/dsh/dashboards/<id>/publish | POST | Publish | bia_dsh:edit |
| /api/bia/dsh/dashboards/<id>/widgets | GET | List widgets | bia_dsh:view |
| /api/bia/dsh/dashboards/<id>/widgets | POST | Add widget | bia_dsh:edit |
| /api/bia/dsh/dashboards/<id>/snapshots | POST | Take snapshot | bia_dsh:snapshots |
| /api/bia/dsh/dashboards/<id>/filters | POST | Add filter | bia_dsh:edit |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| cross_tenant_dashboard_denied | Cross-tenant access | deny |
| public_access_requires_approval | public + not approved | deny |
| widget_limit_enforced | >50 widgets | deny |
| snapshot_requires_published | draft dashboard | deny |
| share_requires_published | draft dashboard | deny |
| archived_dashboard_cannot_be_published | state=archived | deny |

## Data Models
- DashboardResponse: id, tenant_id, name, layout_type, access_level, state, owner_id, widget_count
- WidgetResponse: id, tenant_id, dashboard_id, widget_type, datasource_id, config, position, size
- SnapshotResponse: id, dashboard_id, format, storage_ref, label, requested_by
- DashboardFilterResponse: id, dashboard_id, filter_type, target_field, config

## Streaming Events
- dashboard_created, dashboard_published, dashboard_archived
- widget_added, widget_updated, widget_removed
- snapshot_taken, snapshot_scheduled, dashboard_shared
- filter_applied, data_refreshed

## Edge Cases Handled
- Archived dashboards reject re-publish — require creating new dashboard
- Snapshot scheduling requires published state to prevent empty captures
- Widget count is decremented atomically when a widget is removed
- Sharing a draft dashboard is rejected with explicit publish-first instruction
- Snapshot retention limit triggers explicit cleanup requirement before new snapshot

## Composability Notes
- Consumes metrics and query results from bia_anl for widget data
- Snapshot delivery hooks into ntfy for email/webhook distribution
- wflo can gate dashboard publishing with multi-step approval
- Embedded dashboard tokens can be consumed by sbi (Self-Service BI) portals
- rpt can reference published dashboards as report attachments
