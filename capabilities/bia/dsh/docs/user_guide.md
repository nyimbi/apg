# Dashboard Management — User Guide

**Capability ID**: `bia_dsh` | **Domain**: `bia` | **Version**: `1.1.0`
**Company**: Datacraft | **Copyright**: (c) 2025 | **Author**: Nyimbi Odero

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Dashboard Lifecycle](#dashboard-lifecycle)
5. [Widget Library](#widget-library)
6. [Filters and Cross-Widget Filtering](#filters-and-cross-widget-filtering)
7. [Snapshots and Scheduled Reports](#snapshots-and-scheduled-reports)
8. [Sharing and Embedding](#sharing-and-embedding)
9. [Dashboard Versioning and Rollback](#dashboard-versioning-and-rollback)
10. [KPI Financial Precision](#kpi-financial-precision)
11. [Widget Annotations](#widget-annotations)
12. [Template Library](#template-library)
13. [Async Export Jobs](#async-export-jobs)
14. [Row-Level Security](#row-level-security)
15. [Collaborative Presence](#collaborative-presence)
16. [Datasource Management](#datasource-management)
17. [Themes](#themes)
18. [Audit and Compliance](#audit-and-compliance)
19. [Drill-Through Navigation](#drill-through-navigation)
20. [Configuration Reference](#configuration-reference)

---

## Overview

`bia_dsh` is the Dashboard Management capability for the APG platform.  It provides:

- A full dashboard lifecycle (draft to published to archived)
- A widget library with 15 chart types
- Real-time and scheduled data refresh
- Cross-widget filter propagation
- Tenant-scoped access controls and audit logging
- Dashboard versioning with one-click rollback
- Decimal-precise KPI financials
- Widget annotations for event marking
- Reusable dashboard templates
- Async export jobs for large dashboards
- Row-level security for multi-role deployments
- Collaborative live-presence with cursor awareness

All operations are async-first, tenant-scoped, and emit structured audit events.

---

## Installation

```bash
pip install apg-bia-dsh
```

Environment variables (all optional unless noted):

| Variable | Description |
|----------|-------------|
| `BIA_DSH_DB_URL` | PostgreSQL connection string |
| `OLLAMA_BASE_URL` | Enable AI-powered layout and insight features |
| `BIA_DSH_MAX_WIDGETS` | Override max widgets per dashboard (default 50) |
| `BIA_DSH_SNAPSHOT_RETENTION_DAYS` | Override snapshot retention (default 90) |

---

## Quick Start

```python
import asyncio
from capabilities.bia.dsh.service import DashboardService

async def main():
	svc = DashboardService(tenant_id="acme", actor_id="user_001")

	# 1. Create a dashboard
	dsh = await svc.create_dashboard(
		tenant_id="acme",
		name="Sales Overview",
		owner_id="user_001",
		layout_type="responsive_grid",
		theme="dark",
	)

	# 2. Register a datasource
	ds = await svc.register_datasource(
		tenant_id="acme",
		name="Main PostgreSQL",
		source_type="postgresql",
		connection_config={"host": "pg.acme.internal", "db": "analytics"},
		owner_id="user_001",
	)

	# 3. Add widgets
	kpi = await svc.add_widget(
		tenant_id="acme",
		dashboard_id=dsh["id"],
		name="Monthly Revenue",
		widget_type="kpi_card",
		datasource_type="sql",
		datasource_id=ds["id"],
		owner_id="user_001",
		refresh_interval="5m",
	)

	# 4. Publish
	await svc.publish_dashboard(tenant_id="acme", dashboard_id=dsh["id"])

asyncio.run(main())
```

---

## Dashboard Lifecycle

Dashboards move through three states: `draft` to `published` to `archived`.

| State | Allowed Operations |
|-------|--------------------|
| draft | create, update, add_widget, add_filter |
| published | share, embed, schedule_snapshot, refresh, drill_through |
| archived | read-only; cannot be republished |

### State transitions

```python
# Draft to Published
await svc.publish_dashboard(tenant_id, dashboard_id)

# Published to Archived
await svc.archive_dashboard(tenant_id, dashboard_id)
```

### Force refresh

```python
result = await svc.refresh_dashboard(tenant_id, dashboard_id, actor_id="scheduler")
# result["widget_statuses"] -> per-widget refresh latency and row counts
```

---

## Widget Library

Supported widget types:

| Type | Use Case |
|------|----------|
| `bar_chart` | Categorical comparisons |
| `line_chart` | Trends over time |
| `pie_chart` | Part-to-whole (7 slices or fewer) |
| `donut_chart` | Part-to-whole with central KPI |
| `scatter_plot` | Correlation analysis |
| `heatmap` | Density / intensity maps |
| `table` | Tabular data with sorting |
| `kpi_card` | Single metric highlight |
| `gauge` | Threshold / progress indicator |
| `treemap` | Hierarchical proportions |
| `funnel` | Conversion analysis |
| `map` | Geographic distribution |
| `text` | Markdown narrative panels |
| `image` | Static image embeds |
| `iframe` | Embedded third-party content |

### Adding and updating widgets

```python
w = await svc.add_widget(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	name="Churn Rate",
	widget_type="gauge",
	datasource_type="sql",
	datasource_id=ds["id"],
	owner_id="user_001",
	config={"min": 0, "max": 100, "thresholds": [{"value": 5, "color": "red"}]},
	size={"w": 4, "h": 3},
	refresh_interval="15m",
)

await svc.update_widget(tenant_id, w["id"], {"refresh_interval": "5m"})
await svc.clone_widget(tenant_id, w["id"], target_dashboard_id=other_id, new_name="Churn Rate (Copy)")
await svc.remove_widget(tenant_id, w["id"])
```

### Bulk creation

```python
result = await svc.bulk_create_widgets(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	owner_id="user_001",
	widget_specs=[
		{"name": "MRR", "widget_type": "kpi_card", "datasource_id": ds["id"], "datasource_type": "sql"},
		{"name": "ARR", "widget_type": "kpi_card", "datasource_id": ds["id"], "datasource_type": "sql"},
	],
)
```

---

## Filters and Cross-Widget Filtering

```python
f = await svc.add_filter(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	name="Date Range",
	filter_type="date_range",
	target_field="transaction_date",
	owner_id="user_001",
	config={"default_range": "last_30_days"},
)

ctx = await svc.filter_context(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	filters={"transaction_date": {"gte": "2026-01-01", "lte": "2026-03-31"}, "region": "East Africa"},
)
# ctx["widget_results"] -> per-widget filtered row counts
```

Supported filter types: `date_range`, `dropdown`, `multi_select`, `text_search`, `slider`, `checkbox`.

---

## Snapshots and Scheduled Reports

```python
# On-demand
snap = await svc.take_snapshot(
	tenant_id="acme", dashboard_id=dsh["id"],
	format="pdf", requested_by="user_001", label="Q1 2026 Board Pack",
)

# Scheduled
sched = await svc.schedule_snapshot(
	tenant_id="acme", dashboard_id=dsh["id"],
	frequency="weekly",
	recipients=["board@acme.co.ke"],
	format="pdf",
)
# sched["cron_expression"] -> "0 8 * * 1"
```

---

## Sharing and Embedding

```python
share = await svc.share_dashboard(
	tenant_id="acme", dashboard_id=dsh["id"],
	share_config={
		"recipients": ["alice@acme.co.ke"],
		"permission": "view",
		"expiry_days": 30,
		"require_login": True,
	},
)
print(share["shareable_link"])

embed = await svc.embed_dashboard(
	tenant_id="acme", dashboard_id=dsh["id"],
	embed_params={
		"allowed_domains": ["portal.acme.co.ke"],
		"hide_toolbar": True,
		"ttl_seconds": 7200,
	},
)
print(embed["iframe_snippet"])
```

---

## Dashboard Versioning and Rollback

Up to 50 versions are retained per dashboard (FIFO).

```python
v1 = await svc.snapshot_dashboard_version(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	change_summary="Before restructure — baseline",
)

# ... make changes ...

versions = await svc.list_dashboard_versions(tenant_id="acme", dashboard_id=dsh["id"])
# versions[0] is newest first

result = await svc.rollback_dashboard(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	version_id=v1["version_id"],
)
print(result["restored_version_number"])  # 1
```

Rollback auto-snapshots current state before overwriting (fully reversible).

---

## KPI Financial Precision

All monetary aggregations use `decimal.Decimal` with `ROUND_HALF_EVEN` (banker's rounding).

```python
result = await svc.compute_kpi_financials(
	tenant_id="acme",
	widget_id=kpi["id"],
	raw_values=["1234567.89", "987654.32", "2345678.10"],
	currency="KES",
)
print(result["sum_formatted"])   # KES 4,567,900.31
print(result["mean_formatted"])  # KES 1,522,633.44
```

Pass values as strings, not floats, to guarantee precision.

---

## Widget Annotations

```python
ann = await svc.add_widget_annotation(
	tenant_id="acme",
	widget_id=kpi["id"],
	label="Premium Tier Launch",
	description="Expected +15% MRR uplift over 90 days",
	timestamp_iso="2026-03-01T00:00:00",
	color="#16A34A",
	icon="rocket",
)

annotations = await svc.list_widget_annotations(
	tenant_id="acme",
	widget_id=kpi["id"],
	start_ts="2026-01-01",
	end_ts="2026-06-30",
)

await svc.delete_widget_annotation(tenant_id="acme", widget_id=kpi["id"], annotation_id=ann["id"])
```

---

## Template Library

```python
tmpl = await svc.register_dashboard_template(
	tenant_id="acme",
	template_name="SaaS Metrics Overview",
	category="saas",
	dashboard_config={"layout_type": "responsive_grid", "theme": "dark"},
	widget_specs=[
		{"name": "MRR", "widget_type": "kpi_card", "datasource_id": "placeholder_ds", "datasource_type": "sql"},
		{"name": "Churn Rate", "widget_type": "gauge", "datasource_id": "placeholder_ds", "datasource_type": "sql"},
	],
)

templates = await svc.list_dashboard_templates(tenant_id="acme", category="saas")

result = await svc.instantiate_from_template(
	tenant_id="acme",
	template_id=tmpl["id"],
	name="Q3 SaaS Review",
	owner_id="user_001",
	datasource_overrides={"placeholder_ds": ds["id"]},
)
print(result["widgets_created"])  # 2
```

---

## Async Export Jobs

```python
job = await svc.submit_export_job(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	format="pdf",
	options={"include_filters": True, "page_size": "A3"},
	requested_by="user_001",
)
print(job["status"])  # "queued"

# Poll until complete
status = await svc.get_export_job_status(tenant_id="acme", job_id=job["job_id"])
# Second poll: status["status"] -> "complete", status["download_url"] populated

await svc.cancel_export_job(tenant_id="acme", job_id=job["job_id"])
```

Supported formats: `png`, `pdf`, `html`, `json`, `csv`.

---

## Row-Level Security

```python
await svc.set_rls_policy(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	policy={
		"field_manager":   {"region": "East Africa"},
		"country_analyst": {"country_code": ["KE", "UG", "TZ"]},
		"global_viewer":   {},
	},
)

rls = await svc.resolve_rls_filters(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	actor_id="user_042",
	actor_roles=["field_manager"],
)
# rls["filters"] -> {"region": "East Africa"}

effective_filters = {**rls["filters"], **user_supplied_filters}
ctx = await svc.filter_context(tenant_id, dsh["id"], effective_filters)
```

Multi-role resolution unions values for the same field across all matched roles.

---

## Collaborative Presence

```python
session = await svc.enter_dashboard_session(
	tenant_id="acme",
	dashboard_id=dsh["id"],
	actor_id="analyst_2",
	cursor_position={"x": 420.0, "y": 180.0},
	widget_focus=kpi["id"],
)
print(session["participant_count"])  # 2

participants = await svc.get_session_participants(tenant_id="acme", dashboard_id=dsh["id"])

await svc.leave_dashboard_session(tenant_id="acme", dashboard_id=dsh["id"], actor_id="analyst_2")
```

Sessions upsert on re-entry — same actor never produces duplicate entries.

---

## Datasource Management

```python
ds = await svc.register_datasource(
	tenant_id="acme",
	name="ClickHouse Analytics",
	source_type="clickhouse",
	connection_config={"host": "ch.acme.internal", "port": 9000},
	owner_id="user_001",
)

all_ds = await svc.list_datasources(tenant_id="acme")

test = await svc.test_datasource(tenant_id="acme", datasource_id=ds["id"])
print(test["status"], test["latency_ms"])  # "connected" 12
```

---

## Themes

```python
theme = await svc.create_theme(
	tenant_id="acme",
	name="Acme Corporate",
	colors={"primary": "#1A3C6E", "accent": "#F59E0B", "success": "#16A34A"},
	fonts={"body": "Inter", "heading": "Poppins"},
	owner_id="user_001",
)

themes = await svc.list_themes(tenant_id="acme")
```

---

## Audit and Compliance

```python
events = await svc.get_audit_events(tenant_id="acme")
stats = await svc.get_dashboard_stats(tenant_id="acme")
report = await svc.compliance_audit(tenant_id="acme")
print(report["compliant"])
```

---

## Drill-Through Navigation

```python
result = await svc.drill_through(
	tenant_id="acme",
	widget_id=kpi["id"],
	context={
		"dimension": "product_category",
		"member":    "Premium",
		"measure":   "revenue",
		"value":     125000.00,
	},
)
# result["detail_rows"] -> 20 underlying transaction rows
```

---

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_widgets_per_dashboard` | int | 50 | Hard cap on widgets per dashboard |
| `max_filters_per_dashboard` | int | 20 | Hard cap on filters per dashboard |
| `snapshot_retention_days` | int | 90 | Days before snapshots are purged |
| `default_layout` | str | `responsive_grid` | Default layout for new dashboards |
| `require_approval_for_public` | bool | true | Public dashboards need explicit approval |
| `max_dashboard_versions` | int | 50 | Version history depth per dashboard |
| `cross_widget_filtering` | bool | true | Enable filter propagation across widgets |
| `scheduled_snapshots_enabled` | bool | true | Allow cron-based snapshot scheduling |

All keys are tenant-scoped and can be overridden via the `conf` capability or `BIA_DSH_`-prefixed environment variables.

---

*For service method signatures and Pydantic models see `service.py` and `models.py`.
For REST API details see `api.py`.
For Flask-AppBuilder views see `views.py`.*
