# Self-Service BI (bia_sbi) — World-Class Improvement Proposals

> 15 concrete improvements to make bia_sbi 10x better.  
> Each is grounded in production patterns from Tableau, Looker, Metabase, ThoughtSpot, Power BI, Hex, and Mode.

---

### I1. Semantic Layer with Reusable Metrics
**Category**: Data Modelling  
**Justification**: Raw SQL NLQ is brittle — the same "revenue" question produces inconsistent results across teams. A semantic layer locks metric definitions (joins, aggregations, grain) so every question resolves to the same number. Looker's LookML and dbt Metrics prove this eliminates 80%+ of BI trust issues.  
**Implementation**: Add `MetricDefinition` model (name, expression, grain, dimensions, filters). `natural_language_query` resolves dimension/metric names via the semantic layer before generating SQL. Expose `create_metric`, `list_metrics`, `validate_metric_expression` methods. Cache compiled metric SQL per tenant.  
**Competitor**: Looker LookML, dbt Semantic Layer, Cube.dev

---

### I2. Incremental NLQ Feedback Loop with User Corrections
**Category**: AI / ML Quality  
**Justification**: First-generation NLQ has ~84% confidence — good enough for demos, not for daily finance use. ThoughtSpot's SpotIQ and Power BI Q&A improve by learning from user corrections. Storing accepted/rejected SQL pairs with edited versions creates a fine-tuning dataset that improves confidence to 95%+ within weeks.  
**Implementation**: Add `nlq_feedback` table: `(nlq_id, user_id, verdict: accepted|corrected|rejected, corrected_sql, corrected_chart_type)`. `submit_nlq_feedback` method persists feedback. `natural_language_query` queries local feedback store to prefer correction-validated SQL patterns. Expose `export_nlq_training_data` for periodic Ollama fine-tune jobs.  
**Competitor**: ThoughtSpot SpotIQ, Google Looker Studio Q&A, Microsoft Power BI Q&A

---

### I3. Scheduled Report Snapshots with Email/Webhook Delivery
**Category**: Delivery & Distribution  
**Justification**: Business users need Monday-morning reports in their inbox without opening the UI. Tableau Server, Mode, and Metabase all offer scheduled delivery as a top retention driver. Without it, BI tools stay in "project" phase and never reach daily-driver status.  
**Implementation**: Add `ReportSchedule` model: `(report_id, cron_expr, format: pdf|csv|png, recipients: list[email|webhook_url], next_run_at, last_run_at, last_status)`. `schedule_report`, `list_report_schedules`, `cancel_report_schedule`, `trigger_report_run` methods. Integrates with `ntfy` capability for delivery. Cron parsed with `croniter`.  
**Competitor**: Tableau Server Subscriptions, Metabase Subscriptions, Mode Scheduled Reports

---

### I4. Row-Level Security (RLS) with Dynamic Filter Injection
**Category**: Governance & Security  
**Justification**: Sharing a single datasource across a sales team where each rep only sees their territory is impossible without RLS. Without it, governed data catalogue is a compliance liability — one misconfigured permission leaks cross-tenant data. Snowflake, BigQuery, and dbt all implement RLS as a first-class feature.  
**Implementation**: Add `RLSPolicy` model: `(dataset_id, user_attribute: str, filter_column: str, operator: str, condition_template: str)`. `create_rls_policy`, `list_rls_policies`, `evaluate_rls_for_user` methods. `dataset_preview` and NLQ SQL generation call `evaluate_rls_for_user` and inject `WHERE` clauses before query execution. Policies stored per tenant, evaluated at query time.  
**Competitor**: Tableau Row-Level Security, Looker User Attributes, Power BI RLS

---

### I5. Collaborative Dashboard Builder with Real-Time Presence
**Category**: Collaboration  
**Justification**: BI reports are built by teams, not individuals. Hex's multiplayer notebooks and Figma's real-time collaboration show that presence awareness (who is editing what widget) reduces edit conflicts and increases team adoption by 3x. Current annotation model is async-only.  
**Implementation**: Add `DashboardSession` model: `(dashboard_id, user_id, cursor_widget_id, last_heartbeat, color)`. `join_dashboard_session`, `leave_dashboard_session`, `broadcast_cursor_position`, `list_active_session_users` methods. WebSocket event emission via `mqeb`. Presence expires after 30s without heartbeat. SSE fallback for HTTP-only clients.  
**Competitor**: Hex (multiplayer), Notion, Figma, Google Docs

---

### I6. Embedded Analytics with Signed JWT Tokens
**Category**: Distribution / Integration  
**Justification**: SaaS customers want to white-label dashboards inside their own product. Embedding is the highest-value BI monetisation model — Tableau Embedded and Metabase Embedding drive 40%+ of enterprise contract value. Unsigned iframe embeds are a security anti-pattern.  
**Implementation**: Add `EmbedToken` model: `(workspace_id, audience: str, allowed_filters: list, expires_at, scopes: list[str], signed_jwt: str)`. `create_embed_token`, `validate_embed_token`, `revoke_embed_token` methods. JWT signed with HS256 using tenant-scoped secret. Token carries `tenant_id`, `workspace_id`, `rls_overrides`, `expiry`. API validates token on every embed request.  
**Competitor**: Tableau Embedded, Metabase Signed Embedding, Grafana Embedding

---

### I7. AI-Powered Anomaly Alerts with Configurable Thresholds
**Category**: Proactive Intelligence  
**Justification**: Waiting for a user to open a dashboard to discover a 40% revenue drop is unacceptable for a finance team. Monte Carlo, Atlan, and BigEye charge six figures specifically for anomaly alerting. Surfacing alerts proactively converts BI from reactive to operational intelligence.  
**Implementation**: Add `AnomalyAlert` model: `(dataset_id, metric_expr, window: str, sensitivity: float, alert_channels: list, last_evaluated_at, last_triggered_at, status)`. `create_anomaly_alert`, `evaluate_anomaly_alerts`, `list_triggered_alerts`, `dismiss_alert` methods. Evaluation uses z-score (> sensitivity σ triggers). Integrates with `ntfy` for push notifications. Ollama model optionally provides narrative explanation for each anomaly.  
**Competitor**: Monte Carlo, Atlan Anomaly Detection, Lightdash Alerts, BigEye

---

### I8. Version-Controlled Report Lineage with Diff View
**Category**: Governance / Auditability  
**Justification**: "What changed in last quarter's revenue report?" is impossible to answer without version history. dbt's model versioning and Looker's Git-backed LookML show that lineage builds analyst confidence and satisfies SOC 2 audit requirements for BI artefacts.  
**Implementation**: Add `ReportVersion` model: `(report_id, version_num: int, snapshot: dict, changed_by, change_summary, created_at)`. `create_report_version`, `list_report_versions`, `diff_report_versions`, `restore_report_version` methods. Snapshot stores full chart/filter/layout JSON. Diff computes added/removed/modified fields. Exposes `/api/bia/sbi/reports/<id>/versions` endpoint.  
**Competitor**: dbt Cloud versioning, Looker Git integration, Mode version history

---

### I9. Parameterised Templates with Variable Injection
**Category**: Productivity  
**Justification**: Finance teams build the same revenue-by-region chart for 12 different business units. Parameterised templates (date range, region, metric) reduce duplication 10x and ensure consistency. Metabase's question parameters and Retool's parameterised queries are adoption catalysts.  
**Implementation**: Add `ReportTemplate` model: `(name, description, category, chart_config_template: dict, parameters: list[TemplateParam], created_by, usage_count)`. `TemplateParam`: `(name, type: date|string|number|enum, default, allowed_values)`. `create_template`, `list_templates`, `instantiate_template`, `clone_template` methods. Variable substitution via `{{param_name}}` in SQL and filter expressions. Template marketplace surfaced via `data_catalogue_search`.  
**Competitor**: Metabase Parameters, Redash Parameters, Retool Query Variables

---

### I10. Monetary Precision with Decimal Arithmetic Throughout
**Category**: Correctness / Financial Compliance  
**Justification**: IEEE 754 floating-point arithmetic produces `0.1 + 0.2 = 0.30000000000000004` — unacceptable in a BI tool used for financial reporting. KES, USD, EUR figures must round correctly for balance sheets. Python's `decimal.Decimal` with `ROUND_HALF_UP` eliminates this class of bug.  
**Implementation**: Add `MonetaryValue` model: `(amount: Decimal, currency: str, precision: int)`. Wrap all chart aggregation results that touch `currency`, `amount`, `revenue`, `cost`, `price` column names with `Decimal` conversion and `quantize(Decimal("0.01"), ROUND_HALF_UP)`. Add `format_monetary` helper. Expose `currency_rounding_mode` as tenant config. NLQ result rows containing monetary columns serialise via `MonetaryValue.to_display()`.  
**Competitor**: Tableau Finance Pack, Power BI Financial Reporting, Hex Financial Notebooks

---

### I11. Data Freshness SLA Monitoring with Staleness Badges
**Category**: Trust & Transparency  
**Justification**: A dashboard showing yesterday's data labelled as "live" destroys analyst trust faster than any bug. Atlan and Monte Carlo show data freshness SLAs (expected update interval) as the #1 data reliability feature requested by data consumers. Current quality badge omits freshness lag.  
**Implementation**: Add `FreshnessSLA` model: `(dataset_id, expected_update_interval_minutes: int, last_updated_at: datetime, sla_status: on_time|warning|breached)`. `set_freshness_sla`, `evaluate_freshness_sla`, `list_breached_slas` methods. `data_quality_badge` incorporates freshness into overall score. UI renders a green/amber/red freshness badge per dataset. Integrates with `ntfy` to alert data engineers on SLA breach.  
**Competitor**: Atlan Freshness SLAs, Monte Carlo Data Freshness, Bigeye Freshness

---

### I12. Cross-Dataset Join Builder (Visual Join Designer)
**Category**: Data Modelling / Builder  
**Justification**: Real-world BI always involves joining orders + customers + products. Drag-and-drop join builders (Tableau Data Pane, Metabase Visual Editor) are the #1 feature that converts SQL-fluent analysts to self-service users. Without it, complex reports route back to the data team.  
**Implementation**: Add `JoinDefinition` model: `(left_dataset_id, right_dataset_id, join_type: inner|left|right|full, join_conditions: list[JoinCondition], alias)`. `JoinCondition`: `(left_column, right_column, operator)`. `create_join_definition`, `validate_join`, `list_joins`, `preview_join_result` methods. Join definitions stored per workspace. NLQ and drag-drop builder resolve multi-dataset questions via registered joins. Visual join graph serialised as `{"nodes": [...], "edges": [...]}`.  
**Competitor**: Tableau Data Pane Joins, Metabase Visual Editor, Looker Explore

---

### I13. Collaborative Data Stories with Narrative Blocks
**Category**: Communication / Presentation  
**Justification**: Charts alone don't make decisions. Hex's data narratives and Observable's notebooks show that embedding prose + chart + insight together in a scrollable story increases executive adoption by 5x versus standalone dashboards. Mode Analytics built its entire product around this concept.  
**Implementation**: Add `DataStory` model: `(id, title, blocks: list[StoryBlock], collaborators, published: bool)`. `StoryBlock`: `(type: text|chart|insight|nlq_result|metric, content: dict, order: int)`. `create_story`, `add_story_block`, `reorder_story_blocks`, `publish_story`, `share_story` methods. Stories render as paginated scrollable documents. Published stories get a public URL with optional password protection.  
**Competitor**: Hex Notebooks, Mode Analytics Stories, Observable Notebooks

---

### I14. Multi-Tenant Usage Quotas with Cost Attribution
**Category**: Operations / FinOps  
**Justification**: Without query quotas, a single runaway NLQ job can consume all warehouse compute and cause a $10,000 Snowflake bill. Databricks Unity Catalog and Snowflake Resource Monitors implement per-user/per-team compute budgets. Self-service BI without quotas is operationally unsafe at scale.  
**Implementation**: Add `UsageQuota` model: `(tenant_id, user_id | team_id, quota_type: queries|rows|compute_seconds, limit: Decimal, used: Decimal, period: daily|monthly, reset_at)`. `set_usage_quota`, `check_quota`, `record_quota_usage`, `list_quota_violations` methods. `natural_language_query` and `dataset_preview` call `check_quota` before execution and `record_quota_usage` after. Quota violations emit events to `mqeb`. Admins see cost attribution dashboards per user/team.  
**Competitor**: Databricks Unity Catalog Quotas, Snowflake Resource Monitors, BigQuery Quotas

---

### I15. Smart Chart Recommendations with Explainability
**Category**: AI / UX  
**Justification**: Users picking the wrong chart type (pie chart with 20 slices, line chart for categorical data) is the most common BI mistake. Current NLQ suggests one chart type; Tableau's Show Me and Power BI's AI Visuals provide ranked alternatives with explanations. Explainability builds analyst trust in AI suggestions.  
**Implementation**: Add `ChartRecommendation` model: `(query_id, recommendations: list[RankedChart], column_analysis: dict, selected: str | None)`. `RankedChart`: `(chart_type, score: float, rationale: str, config_suggestion: dict)`. `recommend_chart_types`, `accept_chart_recommendation`, `reject_chart_recommendation` methods. Recommendation engine evaluates: cardinality of x-axis, whether y is numeric/temporal, number of series, data volume. Rationale explains in plain English why each chart type was ranked. Rejection feedback trains the feedback loop (I2).  
**Competitor**: Tableau Show Me, Power BI AI Visuals, Google Looker Explore suggestions
