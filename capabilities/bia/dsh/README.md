# Dashboard Management

## Overview
Dashboard Management (`bia_dsh`) provides dynamic dashboard creation, a widget library with 15 chart types, real-time data binding, responsive layout engines, cross-widget filtering, scheduled snapshot capture, and governed sharing — all tenant-scoped and audit-logged.

New in this release: dashboard versioning with rollback, widget annotations, Decimal-precise KPI financials, dashboard template library, async export jobs, row-level security filter injection, and collaborative session presence.

## Capability ID
`bia_dsh`

## Provides
- `dashboard_creation` — Draft, publish, archive lifecycle for dashboards
- `widget_library` — 15 widget types with datasource binding
- `real_time_data_binding` — Configurable per-widget refresh intervals
- `responsive_layout_engine` — Grid, freeform, tabbed, stacked layouts
- `scheduled_snapshots` — Cron-triggered PNG/PDF/HTML snapshots
- `cross_widget_filtering` — Dashboard-wide filter propagation
- `dashboard_sharing` — Team and organisation-scoped access
- `dashboard_export` — Async export jobs: PNG, PDF, HTML, JSON, CSV
- `dashboard_embedding` — Signed embed tokens for portal integration
- `dashboard_versioning` — Full version history with one-click rollback
- `widget_annotations` — Timestamped chart annotations (events, markers)
- `kpi_financial_precision` — Decimal-arithmetic KPI aggregation (ROUND_HALF_EVEN)
- `template_library` — Reusable dashboard templates with datasource rebinding
- `row_level_security` — Per-role field-value constraint injection at query time
- `collaboration_presence` — Live session participants and cursor awareness

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
| max_dashboard_versions | 50 | Version history depth per dashboard |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/dsh/dashboards | GET | List dashboards | bia_dsh:view |
| /api/bia/dsh/dashboards | POST | Create dashboard | bia_dsh:create |
| /api/bia/dsh/dashboards/id | GET | Get dashboard | bia_dsh:view |
| /api/bia/dsh/dashboards/id | PUT | Update dashboard | bia_dsh:edit |
| /api/bia/dsh/dashboards/id/publish | POST | Publish | bia_dsh:edit |
| /api/bia/dsh/dashboards/id/widgets | GET | List widgets | bia_dsh:view |
| /api/bia/dsh/dashboards/id/widgets | POST | Add widget | bia_dsh:edit |
| /api/bia/dsh/dashboards/id/snapshots | POST | Take snapshot | bia_dsh:snapshots |
| /api/bia/dsh/dashboards/id/filters | POST | Add filter | bia_dsh:edit |
| /api/bia/dsh/dashboards/id/versions | GET | List versions | bia_dsh:view |
| /api/bia/dsh/dashboards/id/versions/vid/rollback | POST | Rollback | bia_dsh:edit |
| /api/bia/dsh/dashboards/id/rls | PUT | Set RLS policy | bia_dsh:admin |
| /api/bia/dsh/widgets/id/annotations | GET/POST | Annotations | bia_dsh:edit |
| /api/bia/dsh/widgets/id/kpi | POST | KPI financials | bia_dsh:view |
| /api/bia/dsh/templates | GET/POST | Template library | bia_dsh:view |
| /api/bia/dsh/templates/id/instantiate | POST | Instantiate | bia_dsh:create |
| /api/bia/dsh/exports | POST | Submit export job | bia_dsh:view |
| /api/bia/dsh/exports/job_id | GET | Poll export job | bia_dsh:view |
| /api/bia/dsh/dashboards/id/session | POST/DELETE | Presence | bia_dsh:view |

## Usage Examples

### Create and publish a dashboard

```python
svc = DashboardService(tenant_id="acme", actor_id="analyst_1")

dsh = await svc.create_dashboard(
	tenant_id="acme",
	name="Revenue Overview",
	owner_id="analyst_1",
	layout_type="responsive_grid",
	theme="dark",
	auto_refresh_seconds=300,
)

widget = await svc.add_widget(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	name="Monthly Revenue",
	widget_type="bar_chart",
	datasource_type="sql",
	datasource_id="ds_postgres_main",
	owner_id="analyst_1",
	refresh_interval="5m",
)

await svc.publish_dashboard(tenant_id="acme", dashboard_id=dsh["id"])
```

### KPI financials with Decimal precision

```python
result = await svc.compute_kpi_financials(
	tenant_id="acme",
	widget_id=widget["id"],
	raw_values=["1234567.89", "987654.32", "2345678.10"],
	currency="KES",
)
# result["sum_formatted"] -> "KES 4,567,900.31"
```

### Dashboard versioning and rollback

```python
ver = await svc.snapshot_dashboard_version(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	change_summary="Added 3 new KPI widgets",
)

await svc.rollback_dashboard(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	version_id=ver["version_id"],
)
```

### Widget annotations

```python
ann = await svc.add_widget_annotation(
	tenant_id="acme",
	widget_id=widget["id"],
	label="Product Launch",
	timestamp_iso="2026-03-01T00:00:00",
	color="#16A34A",
	icon="rocket",
)
```

### Row-level security

```python
await svc.set_rls_policy(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	policy={
		"field_manager": {"region": "East Africa"},
		"country_analyst": {"country_code": ["KE", "UG", "TZ"]},
	},
)

filters = await svc.resolve_rls_filters(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	actor_id="user_042",
	actor_roles=["field_manager"],
)
# filters["filters"] -> {"region": "East Africa"}
```

### Template library

```python
tmpl = await svc.register_dashboard_template(
	tenant_id="acme",
	template_name="SaaS Revenue Overview",
	category="saas",
	dashboard_config={"layout_type": "responsive_grid", "theme": "dark"},
	widget_specs=[
		{"name": "MRR", "widget_type": "kpi_card", "datasource_id": "placeholder_ds"},
	],
)

result = await svc.instantiate_from_template(
	tenant_id="acme",
	template_id=tmpl["id"],
	name="Q2 SaaS Review",
	owner_id="analyst_1",
	datasource_overrides={"placeholder_ds": "ds_postgres_main"},
)
```

### Async export job

```python
job = await svc.submit_export_job(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	format="pdf",
	requested_by="analyst_1",
)
# job["status"] -> "queued"

status = await svc.get_export_job_status(tenant_id="acme", job_id=job["job_id"])
# Second poll: status["status"] -> "complete", status["download_url"] populated
```

### Collaborative presence

```python
await svc.enter_dashboard_session(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	actor_id="analyst_2",
	cursor_position={"x": 420.0, "y": 180.0},
)

participants = await svc.get_session_participants(tenant_id="acme", dashboard_id=dsh["id"])
```

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
- `DashboardResponse` — id, tenant_id, name, layout_type, access_level, state, owner_id, widget_count
- `WidgetResponse` — id, tenant_id, dashboard_id, widget_type, datasource_id, config, position, size
- `SnapshotResponse` — id, dashboard_id, format, storage_ref, label, requested_by
- `DashboardFilterResponse` — id, dashboard_id, filter_type, target_field, config
- Version record — version_id, version_number, config_hash, dashboard_snapshot, widgets_snapshot
- Annotation record — id, widget_id, label, description, timestamp_iso, color, icon, author_id
- Template record — id, name, category, dashboard_config, widget_specs, use_count
- Export job — job_id, status, progress_pct, download_url, estimated_seconds
- RLS policy — dashboard_id, policy dict (role to field to value), active
- Session entry — actor_id, cursor_position, widget_focus, joined_at

## Streaming Events
- dashboard_created, dashboard_published, dashboard_archived, dashboard_rolled_back
- widget_added, widget_updated, widget_removed, widget_cloned
- snapshot_taken, snapshot_scheduled, dashboard_shared
- filter_applied, data_refreshed
- dashboard_version_snapshotted
- widget_annotation_added, widget_annotation_deleted
- kpi_financials_computed
- dashboard_template_registered, dashboard_instantiated_from_template
- export_job_submitted, export_job_completed, export_job_cancelled
- rls_policy_set, rls_policy_applied
- session_entered, session_left

## Edge Cases Handled
- Archived dashboards reject re-publish — require creating new dashboard
- Snapshot scheduling requires published state to prevent empty captures
- Widget count is decremented atomically when a widget is removed
- Sharing a draft dashboard is rejected with explicit publish-first instruction
- Version rollback auto-snapshots current state before overwriting (reversible rollback)
- RLS policy union merges values for the same field across multiple roles
- Export job simulates background completion on second poll
- Presence sessions upsert on re-entry (same actor never produces duplicate entries)

## Composability Notes
- Consumes metrics and query results from `bia_anl` for widget data
- Snapshot delivery hooks into `ntfy` for email/webhook distribution
- `wflo` can gate dashboard publishing with multi-step approval
- Embedded dashboard tokens can be consumed by `sbi` (Self-Service BI) portals
- `rpt` can reference published dashboards as report attachments
- `mqeb` receives session presence events for co-editor awareness
- Template library accelerates `sbi` portal provisioning with pre-built blueprints

---

## World-Class Enhancements (v2.0)

- **I1.** Dashboard Management — World-Class Improvement Catalogue
- **I2.** Improvement 1 — Parameterised Collaboration Cursors (Presence Awareness)
- **I3.** Improvement 2 — AI-Assisted Layout Optimisation
- **I4.** Improvement 3 — Cross-Dashboard Drill-Through with Breadcrumb Context
- **I5.** Improvement 4 — Semantic Color-Blind-Safe Palette Enforcement
- **I6.** Improvement 5 — Snapshot Diff & Change Detection
- **I7.** Improvement 6 — Widget Data Caching with Stale-While-Revalidate
- **I8.** Improvement 7 — Dashboard Versioning and Rollback
- **I9.** Improvement 8 — Row-Level Security Filter Injection
- **I10.** Improvement 9 — Dashboard Revenue Attribution (Decimal-Precise Financials)
- **I11.** Improvement 10 — Adaptive Refresh Scheduling (Backpressure-Aware)
- **I12.** Improvement 11 — Dashboard Subscription and Digest Delivery
- **I13.** Improvement 12 — Widget Annotation and Commentary Layer
- **I14.** Improvement 13 — Dashboard Template Library and Instant Provisioning
- **I15.** Improvement 14 — Anomaly Flagging on KPI Widgets

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
