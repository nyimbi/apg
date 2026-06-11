# Dashboard Management — World-Class Improvement Catalogue

**Capability**: `bia_dsh` | **Domain**: `bia` | **Version**: `1.0.0`  
**Author**: Nyimbi Odero | **Company**: Datacraft | **Date**: 2026-06-11

---

## Improvement 1 — Parameterised Collaboration Cursors (Presence Awareness)

**Category**: Collaboration / Real-Time UX

**Justification**: Competing BI platforms (Tableau Cloud, Sigma Computing) ship multiplayer editing with live cursors. Users building dashboards together currently have no awareness of co-editors, causing overwrites and confusion. Presence metadata costs < 50 bytes per user over WebSocket.

**Implementation**:
- Add `enter_dashboard_session(tenant_id, dashboard_id, actor_id, cursor_position)` and `leave_dashboard_session(...)` methods.
- Store a `_sessions: dict[tuple[str,str], list[dict]]` — keyed on `(tenant_id, dashboard_id)`.
- Each session entry carries `{actor_id, cursor_x, cursor_y, widget_focus, joined_at, last_seen_at}`.
- Broadcast via `mqeb` event stream on every cursor move (debounced to 100 ms client-side).
- Return current participants from `get_session_participants(tenant_id, dashboard_id)`.

**Competitor Reference**: Sigma Computing "Multiplayer" (2023), Notion Live Cursors (2022)

---

## Improvement 2 — AI-Assisted Layout Optimisation

**Category**: AI / Generative UX

**Justification**: Looker Studio and Power BI Auto-Layout use heuristics to distribute widgets based on importance score. Users with >12 widgets spend disproportionate time arranging. An AI pass can score widgets by datasource cardinality and suggest a canonical grid.

**Implementation**:
- Add `suggest_layout(tenant_id, dashboard_id, strategy="importance")` returning a ranked list of `{widget_id, suggested_x, suggested_y, suggested_w, suggested_h, score}`.
- Score based on: widget_type weight (KPI > bar > table), datasource refresh rate, and last-viewed frequency.
- Optionally call Ollama for natural-language layout rationale if `OLLAMA_BASE_URL` is set.
- Add `apply_suggested_layout(tenant_id, dashboard_id, layout_plan)` that bulk-updates widget positions.

**Competitor Reference**: Power BI Smart Narration + Auto-Layout (2024), Google Looker Studio Auto-Arrange

---

## Improvement 3 — Cross-Dashboard Drill-Through with Breadcrumb Context

**Category**: Analytics Depth / Navigation

**Justification**: Superset and Tableau support parameterised navigation between dashboards (click on a KPI → land on the relevant detail dashboard with context pre-applied). Current `drill_through` only exposes raw rows within the same widget.

**Implementation**:
- Add `create_drill_link(tenant_id, source_dashboard_id, target_dashboard_id, mapping)` where `mapping` maps source dimension members to target filter fields.
- Add `resolve_drill_link(tenant_id, link_id, member_value)` that returns a `{target_dashboard_id, pre_applied_filters, breadcrumb}` payload.
- Store links in `_drill_links: dict[tuple[str,str], dict]`.
- Breadcrumb carries: `[{dashboard_name, filter_applied, timestamp}]` — append on each hop, max depth 5.

**Competitor Reference**: Tableau Dashboard Actions (drill-through links), Superset Cross-Filter Dashboard Links

---

## Improvement 4 — Semantic Color-Blind-Safe Palette Enforcement

**Category**: Accessibility / Compliance

**Justification**: WCAG 2.2 Level AA requires sufficient color contrast. Dashboards deployed in regulated sectors (fintech, healthcare) must pass color-blind safety checks. Competitors Qlik Sense and Dundas BI ship built-in WCAG validators.

**Implementation**:
- Add `validate_theme_accessibility(tenant_id, theme_id)` that reads the theme's color tokens and computes WCAG contrast ratios (4.5:1 for normal text, 3:1 for large).
- Returns `{passed, violations: [{token, contrast_ratio, required_ratio, fix_suggestion}]}`.
- Add `apply_colorblind_safe_palette(tenant_id, theme_id, mode)` where `mode ∈ {"deuteranopia", "protanopia", "tritanopia", "monochrome"}`.
- Use perceptual luminance formula: `L = 0.2126 R + 0.7152 G + 0.0722 B` (sRGB linearised).

**Competitor Reference**: Qlik Sense Accessibility Checker, Dundas BI WCAG Theme Validator

---

## Improvement 5 — Snapshot Diff & Change Detection

**Category**: Governance / Audit

**Justification**: Regulators in finance (Basel IV reporting) require evidence that dashboards have not been altered between sign-off and distribution. No open-source BI platform ships cryptographic snapshot diff natively.

**Implementation**:
- Add `diff_snapshots(tenant_id, snapshot_id_a, snapshot_id_b)` returning `{changed, diff_summary: [{field, old, new}], hash_a, hash_b}`.
- At snapshot creation, compute `sha256` of the dashboard config JSON and store as `config_hash` on the snapshot record.
- `diff_snapshots` compares `config_hash`, widget count delta, and filter delta — no pixel comparison needed for contract compliance.
- Add `sign_snapshot(tenant_id, snapshot_id, signer_id)` appending a `{signer_id, signed_at, config_hash}` record to the snapshot.

**Competitor Reference**: Sigma Computing Audit History (2024), dbt Cloud Semantic Change Detection

---

## Improvement 6 — Widget Data Caching with Stale-While-Revalidate

**Category**: Performance / Scalability

**Justification**: Dashboards with high-refresh-rate widgets (30 s) under multi-user load hammer the datasource layer. Stale-While-Revalidate (SWR) pattern — as used by Vercel, Cloudflare, and Grafana — serves cached data immediately while revalidating in the background.

**Implementation**:
- Add `_widget_cache: dict[tuple[str,str], dict]` keyed on `(tenant_id, widget_id)`.
- Each cache entry: `{data, cached_at, ttl_seconds, revalidating}`.
- Add `get_widget_data(tenant_id, widget_id, max_staleness_seconds)` that checks cache freshness, returns stale data if within `max_staleness_seconds`, and schedules background revalidation.
- Add `invalidate_widget_cache(tenant_id, widget_id)` for forced eviction.
- Cache hit/miss metrics emitted on every call.

**Competitor Reference**: Grafana Query Caching (v10, 2023), Redash SWR Cache Layer

---

## Improvement 7 — Dashboard Versioning and Rollback

**Category**: Governance / Change Management

**Justification**: Power BI and Tableau both ship version history for reports. When a dashboard breaks post-update, analysts need one-click rollback. Currently there is no version ledger — updates are destructive.

**Implementation**:
- Add `_dashboard_versions: dict[tuple[str,str], list[dict]]` keyed on `(tenant_id, dashboard_id)`.
- On every `update_dashboard` and `add_widget`/`remove_widget`, push a version snapshot `{version_id, dashboard_snapshot, widgets_snapshot, actor_id, created_at, change_summary}`.
- Cap at `max_versions=50` per dashboard (FIFO eviction).
- Add `list_dashboard_versions(tenant_id, dashboard_id)` and `rollback_dashboard(tenant_id, dashboard_id, version_id)`.
- Rollback atomically restores dashboard config and widget list, records a `dashboard_rolled_back` audit event.

**Competitor Reference**: Power BI Version History (2023), Tableau Workbook Revision History

---

## Improvement 8 — Row-Level Security Filter Injection

**Category**: Security / Multi-Tenancy

**Justification**: Enterprise BI deployments (Salesforce CRM Analytics, Looker) enforce row-level security (RLS) at query time. A field manager should only see their region's data on a shared dashboard — enforced server-side, not via separate dashboard copies.

**Implementation**:
- Add `set_rls_policy(tenant_id, dashboard_id, policy: dict)` where policy maps `{actor_role → {field: value_constraint}}`.
- Add `resolve_rls_filters(tenant_id, dashboard_id, actor_id, actor_roles)` returning the effective filter set to inject before query execution.
- Store policies in `_rls_policies: dict[tuple[str,str], dict]`.
- Integrate into `filter_context` — always merge RLS filters with user-supplied filters (RLS takes precedence).
- Log RLS application in audit trail with `rls_policy_applied` event.

**Competitor Reference**: Looker User Attributes / RLS, Salesforce CRM Analytics Row-Level Security

---

## Improvement 9 — Dashboard Revenue Attribution (Decimal-Precise Financials)

**Category**: Financial Analytics / Precision

**Justification**: KPI widgets displaying monetary values (revenue, cost, margin) must use `Decimal` arithmetic — IEEE 754 float arithmetic produces visible rounding errors at scale (e.g., $0.1 + $0.2 ≠ $0.3). Bloomberg Terminal and Refinitiv Eikon enforce `Decimal` throughout their display layer.

**Implementation**:
- Add `compute_kpi_financials(tenant_id, widget_id, raw_values: list[str], currency: str)` that accepts string-encoded decimal values, performs aggregation using `Decimal` (Python `decimal` module, `ROUND_HALF_EVEN`), and returns `{sum, mean, min, max, stddev, currency, formatted}`.
- Store results in the widget's `config["financials"]`.
- Use `Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)` for all monetary arithmetic.
- Return ISO 4217 currency code alongside formatted string (`"KES 1,234,567.89"`).

**Competitor Reference**: Bloomberg Terminal decimal precision model, Refinitiv Workspace financial display layer

---

## Improvement 10 — Adaptive Refresh Scheduling (Backpressure-Aware)

**Category**: Performance / Reliability

**Justification**: Fixed-interval refresh creates thundering-herd load spikes. Grafana's smart refresh and Datadog's adaptive sampling jitter refresh timing and back off when datasource latency exceeds threshold. This prevents cascade failures under peak load.

**Implementation**:
- Add `configure_adaptive_refresh(tenant_id, dashboard_id, base_interval_seconds, jitter_pct, max_backoff_multiplier)`.
- Add `_refresh_backoff: dict[tuple[str,str], dict]` tracking `{current_multiplier, consecutive_failures, last_latency_ms}`.
- On each `refresh_dashboard` call, if `latency_ms > threshold` → increment backoff multiplier (capped at `max_backoff_multiplier`). On success → decay multiplier by 0.5.
- `next_refresh_at` emitted in refresh result so clients can schedule accordingly.
- Emit `refresh_backoff_applied` audit event when multiplier > 1.

**Competitor Reference**: Grafana Smart Refresh (2023), Datadog Adaptive Sampling

---

## Improvement 11 — Dashboard Subscription and Digest Delivery

**Category**: Collaboration / Notifications

**Justification**: Metabase and Redash ship scheduled email digests with widget screenshots inline. Static PDF snapshots are insufficient for executive stakeholders who want a curated selection of KPIs in an email body, not an attachment.

**Implementation**:
- Add `subscribe_to_dashboard(tenant_id, dashboard_id, subscriber_id, config: dict)` where config carries `{frequency, widget_ids, format, include_sparklines, subject_template}`.
- Store in `_subscriptions: dict[tuple[str,str], dict]`.
- Add `generate_digest(tenant_id, subscription_id)` returning `{html_body, subject, attachments, widget_snapshots}`.
- Digest renders each subscribed widget's last data as an inline table or KPI block.
- Hook into `ntfy` for actual delivery; `generate_digest` is delivery-channel-agnostic.

**Competitor Reference**: Metabase Subscriptions (2022), Redash Email Digest, Tableau Subscription Emails

---

## Improvement 12 — Widget Annotation and Commentary Layer

**Category**: Collaboration / Context

**Justification**: Annotations on time-series charts (marking events like product launches, outages) are a standard feature in Grafana, Datadog, and New Relic. Analysts currently have no structured way to attach context to data spikes within a widget.

**Implementation**:
- Add `add_widget_annotation(tenant_id, widget_id, annotation: dict)` where annotation carries `{label, description, timestamp_iso, color, icon, author_id}`.
- Store in `_annotations: dict[tuple[str,str], list[dict]]` keyed on `(tenant_id, widget_id)`.
- Add `list_widget_annotations(tenant_id, widget_id, start_ts, end_ts)` with time-range filtering.
- Add `delete_widget_annotation(tenant_id, widget_id, annotation_id)`.
- Annotations are included in `export_dashboard_config` output.

**Competitor Reference**: Grafana Annotations (native), Datadog Event Overlays, New Relic Deployment Markers

---

## Improvement 13 — Dashboard Template Library and Instant Provisioning

**Category**: Developer Experience / Time-to-Value

**Justification**: Tableau Public and Grafana dashboard.json marketplace reduce time-to-first-dashboard from hours to minutes. Teams repeatedly build the same patterns (revenue overview, funnel analysis, cohort retention). Codified templates eliminate this rework.

**Implementation**:
- Add `register_dashboard_template(tenant_id, template_name, category, dashboard_config, widget_specs, filter_specs, owner_id)`.
- Store in `_templates: dict[tuple[str,str], dict]`.
- Add `list_dashboard_templates(tenant_id, category=None)` with optional category filter.
- Add `instantiate_from_template(tenant_id, template_id, name, owner_id, datasource_overrides)` that clones the template config and rebinds datasources per `datasource_overrides`.
- Templates carry `{template_id, name, category, tags, preview_url, use_count, created_at}`.

**Competitor Reference**: Grafana Dashboard Marketplace, Tableau Public Templates, Metabase Starting Questions

---

## Improvement 14 — Anomaly Flagging on KPI Widgets

**Category**: AI / Proactive Analytics

**Justification**: Sisense and Thoughtspot ship auto-generated anomaly alerts on KPI widgets using simple statistical thresholds (±2σ). Users currently discover metric anomalies manually during periodic dashboard reviews — too late for operational response.

**Implementation**:
- Add `configure_kpi_anomaly_detection(tenant_id, widget_id, config: dict)` where config carries `{method: "zscore"|"iqr"|"ewma", threshold, lookback_periods, alert_recipients}`.
- Store configs in `_anomaly_configs: dict[tuple[str,str], dict]`.
- Add `evaluate_kpi_anomaly(tenant_id, widget_id, current_value: str, historical_values: list[str])` using `Decimal` arithmetic for all statistical calculations.
- Returns `{is_anomaly, score, method, threshold, direction: "above"|"below", suggested_alert}`.
- Integrates with `ntfy` to dispatch anomaly alerts; logs `kpi_anomaly_detected` audit event.

**Competitor Reference**: Sisense Pulse Alerts, Thoughtspot Monitor (2024), Grafana Alerting ML

---

## Improvement 15 — Export Pipeline with Async Job Tracking

**Category**: Performance / UX

**Justification**: Large dashboard exports (PDF with 20 widgets, full data tables) block the request thread for 30+ seconds. Looker and Tableau route exports through an async job queue, returning a job_id immediately with a polling endpoint for status.

**Implementation**:
- Add `submit_export_job(tenant_id, dashboard_id, format, options: dict, requested_by)` returning `{job_id, status: "queued", estimated_seconds}` immediately.
- Store jobs in `_export_jobs: dict[tuple[str,str], dict]` with `{job_id, status, progress_pct, download_url, error, created_at, completed_at}`.
- Add `get_export_job_status(tenant_id, job_id)` for polling.
- Add `cancel_export_job(tenant_id, job_id)` for cancellation.
- Simulate async completion by marking job `"complete"` on second `get_export_job_status` call (real implementation delegates to Celery/ARQ worker).
- Emit `export_job_submitted`, `export_job_completed`, `export_job_failed` audit events.

**Competitor Reference**: Looker Async Downloads (API v4), Tableau Extract Refresh Jobs API, Power BI Export API (async mode)
