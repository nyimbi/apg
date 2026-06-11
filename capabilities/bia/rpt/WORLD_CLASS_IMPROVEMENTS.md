# Report Builder — World-Class Improvement Catalogue

**Capability**: `bia_rpt` | **Domain**: `bia` | **Author**: Nyimbi Odero | **© 2025 Datacraft**

---

### I1. Parameterised Report Templates with Semantic Variables
**Category**: Authoring | **Justification**: Reports rewritten from scratch per-tenant waste 80% of analyst time. Template inheritance with semantic variable substitution (`{{fiscal_year}}`, `{{region_filter}}`) lets a single template serve 1 000 tenant variants without duplication — same model as Sigma Computing's workbooks. | **Implementation**: Add `create_report_template()` + `instantiate_from_template()` methods; store template variable schema as JSON Schema draft-7; resolve variables at run time via a `ParameterResolver` that pulls from tenant conf, request context, and user profile. Reject runs where required variables are unbound. | **Competitor**: Sigma Computing (workbook variables), Looker (LookML parameter blocks), Retool (query parameters)

---

### I2. Incremental / Streaming Report Generation
**Category**: Performance | **Justification**: Current `run_report()` blocks until row_count is known, capping throughput at ~5 000 rows. Streaming via server-sent events lets users see the first page in <200 ms regardless of dataset size — the Metabase approach for large SQL results. Eliminates timeout errors on queries that take >30 s. | **Implementation**: Yield `RunChunk` events (page_num, rows, partial_output_ref) from an async generator `stream_report()`; integrate with the `mqeb` event bus; store chunks in object storage as they arrive; assemble final output ref once `status=complete` sentinel is emitted. Use `asyncio.Queue` + producer/consumer pattern. | **Competitor**: Metabase (streaming SQL results), Redash (query result streaming), Apache Superset (async queries)

---

### I3. Monetary Column Precision with Decimal Arithmetic
**Category**: Data Integrity | **Justification**: Float aggregations on monetary columns introduce rounding drift that accumulates to material errors in financial reports (KES 0.01 per row × 10 M rows = KES 100 000 misstatement). `decimal.Decimal` with `ROUND_HALF_UP` eliminates this — IFRS 9 disclosure requirement. | **Implementation**: Column `data_type="money"` triggers `Decimal` coercion in the run pipeline; format string `"KES #,##0.00"` applied at render time; `add_column()` validates `data_type` against extended set including `money`, `percentage`, `basis_points`; aggregation functions use `Decimal` accumulators. Guard with `assert isinstance(value, Decimal)` at service boundary. | **Competitor**: SAP Analytics Cloud (currency translation), IBM Cognos (currency type columns), Oracle OBIEE (data types)

---

### I4. Report Bursting — Per-Recipient Parameter Injection
**Category**: Distribution | **Justification**: Sending each regional manager their own slice of the same report (filtered to their region) today requires N separate reports and N schedules. Bursting reduces that to 1 report + 1 burst schedule — the Crystal Reports feature that drove its enterprise dominance. | **Implementation**: `create_burst_schedule()` takes a `burst_list: list[BurstTarget]` where each target carries `{recipient, parameters: dict}`; `execute_burst()` fans out via `asyncio.gather()` calling `run_report()` per target with injected parameters; each output has a dedicated signed URL; max burst width = 500 recipients per run. | **Competitor**: Crystal Reports (bursting), SSRS (data-driven subscriptions), Cognos (burst distribution)

---

### I5. Semantic Caching of Report Runs
**Category**: Performance | **Justification**: Identical parameter sets on the same report within a cache TTL hit the database redundantly. A content-addressed cache keyed on `(report_id, sorted_parameters_hash, format)` cuts database load by 60–90% for frequently-viewed reports — the Looker result caching model. | **Implementation**: Compute SHA-256 of `json.dumps(sorted(parameters.items()))` + report version + format; check `BoundedCache` before executing; store `(output_ref, run_id, expires_at)` in cache entry; invalidate on `update_report()` or `publish_report()`; expose `cache_hit: bool` in run response; configurable TTL per report. | **Competitor**: Looker (result caching), Sisense (semantic cache layer), Qlik (associative engine caching)

---

### I6. Column-Level Data Masking and PII Redaction
**Category**: Governance | **Justification**: Reports containing PII (MSISDN, ID numbers, account numbers) distributed to external parties violate Kenya's Data Protection Act 2019 and GDPR. Column-level masking (`full_mask`, `partial_mask`, `tokenise`, `suppress`) enforced at render time removes the need for separate sanitised report variants. | **Implementation**: `add_column()` accepts `mask_policy: MaskPolicy | None`; `MaskPolicy` enum: `none`, `full`, `partial_last4`, `tokenise`, `suppress`; masking applied in `run_report()` after data retrieval before serialisation; mask decisions logged per-column in audit trail; `mask_override_requires_role: str` blocks privilege escalation. | **Competitor**: Microsoft Power BI (RLS + column security), Snowflake Dynamic Data Masking, Databricks column masks

---

### I7. Automated Anomaly Flagging in Report Outputs
**Category**: Intelligence | **Justification**: Users manually scan report output for outliers — a task that degrades with volume. Automated z-score and IQR anomaly detection on numeric columns surfaces flags directly in report metadata without requiring a separate analytics pass, the way Tableau Explain Data works. | **Implementation**: `async anomaly_scan_report(tenant_id, report_id, run_id, sensitivity: float = 2.0)` computes column-level z-scores from run data; flags cells exceeding `sensitivity` standard deviations; returns `AnomalyReport` with `flagged_rows`, `column_stats`, `severity` (info/warn/critical); integrates with ntfy for critical anomaly alerts; uses `statistics.stdev` (stdlib, no ML dependency). | **Competitor**: Tableau (Explain Data), ThoughtSpot (auto-insights), Qlik (associative anomaly detection)

---

### I8. Cross-Report Diff — Version-to-Version Comparison
**Category**: Auditability | **Justification**: Compliance teams need to explain why this month's revenue figure differs from last month's. A structural diff showing which filters changed, which columns were added/removed, and which parameter defaults shifted reduces manual forensics from hours to seconds. | **Implementation**: `async diff_report_runs(tenant_id, run_id_a, run_id_b)` compares `(columns, filters, parameters, grouping_config)` between two runs via `deepdiff`-style recursive comparison; returns `ReportDiff` with `added`, `removed`, `changed` sections; UI renders as a side-by-side JSON diff; diff stored in audit trail with approver sign-off workflow. | **Competitor**: dbt (model diffing), Great Expectations (data doc diffs), Lightdash (metric version diffs)

---

### I9. Natural Language Report Builder (Ollama-backed)
**Category**: Intelligence | **Justification**: Non-technical users cannot navigate column pickers and filter operators. NL-to-report translation (`"show me top 10 customers by revenue last quarter, exclude churned"`) converts intent to a full `ReportCreate` payload using a locally-hosted Ollama model — no cloud data egress, aligned with project AI strategy. | **Implementation**: `async nl_to_report(tenant_id, prompt: str, datasource_schema: dict)` calls Ollama `/api/generate` with a structured extraction prompt; parses JSON response into `ReportCreate`; validates against datasource schema; presents `draft_report` + `confidence_score` to user for confirmation before `create_report()` is called; falls back gracefully when OLLAMA_BASE_URL unset. | **Competitor**: Tableau Ask Data, ThoughtSpot SearchIQ, Qlik Natural Language Insights

---

### I10. Subscription Self-Service Portal with Preference Centre
**Category**: Distribution | **Justification**: Current distribution is admin-configured. Users cannot manage their own subscriptions (format preference, frequency, pause/resume), leading to support tickets for every change and unsubscribe requests going unhandled. Self-service reduces distribution admin load by ~70%. | **Implementation**: `async upsert_subscription(tenant_id, user_id, report_id, preferences: SubscriptionPreferences)` where `SubscriptionPreferences` carries `format`, `frequency`, `paused_until: datetime | None`, `delivery_channel`; `async list_my_subscriptions(tenant_id, user_id)` returns user-scoped view; `async pause_subscription()` / `async resume_subscription()` update state; unsubscribe honours a 24 h cool-down before hard-delete. | **Competitor**: Salesforce CRM Analytics (subscription management), SSRS (My Subscriptions), Jasper Reports (subscription portal)

---

### I11. Report Output Watermarking and Digital Signature
**Category**: Governance | **Justification**: Reports distributed externally can be leaked without traceability. Per-recipient watermarking (recipient name + export timestamp embedded invisibly in PDF metadata and visibly in footer) plus a detached digital signature enables forensic attribution. Mandatory for financial institutions under CBK Digital Banking Guidelines. | **Implementation**: `async watermark_export(tenant_id, export_id, recipient_id, visible: bool = True, invisible: bool = True)` embeds `f"Confidential — {recipient_name} — {timestamp}"` in PDF footer via `reportlab`/`pypdf`; SHA-256 hash of output bytes stored in audit record; `verify_export_signature(tenant_id, export_id, file_bytes)` re-hashes and compares; watermark bypass requires `bia_rpt:admin` role. | **Competitor**: MicroStrategy (watermarking), IBM Cognos (PDF signing), Adobe Experience Manager

---

### I12. Report Data Lineage Graph (Column-to-Source Tracing)
**Category**: Governance | **Justification**: When a report number is questioned, analysts must manually trace through dashboards, queries, and ETL jobs to find the source. An automated lineage graph from output column → report column → datasource field → ETL job → raw table reduces mean time to explain from days to minutes — the dbt lineage model applied to reports. | **Implementation**: Extend `report_lineage()` to emit a DAG: `{nodes: list[LineageNode], edges: list[LineageEdge]}` where `LineageNode` types are `report_column`, `datasource_field`, `etl_job`, `raw_table`; edges carry `transform_type` (direct, aggregation, derived); DAG serialised as `networkx`-compatible adjacency list; rendered via D3.js in UI; persisted as JSON blob per report version. | **Competitor**: dbt (column lineage), Atlan (data lineage), Alation (lineage graph)

---

### I13. Multi-Tenant Report Marketplace
**Category**: Composability | **Justification**: Datacraft operates multiple tenants that solve similar reporting problems independently. A governed marketplace where tenants can publish report templates for peer consumption (with data-source binding step) reduces duplicated authoring effort across the platform. Cross-tenant revenue model: template licensing fees. | **Implementation**: `async publish_to_marketplace(tenant_id, report_id, listing: MarketplaceListing)` strips data and publishes template schema; `async browse_marketplace(tenant_id, query: str, category: str)` searches listings; `async clone_from_marketplace(tenant_id, listing_id, datasource_mapping: dict)` instantiates template with local datasource bindings; marketplace entries stored in shared `_marketplace` dict keyed by listing_id; access control: publisher must be `marketplace:publisher` role. | **Competitor**: Tableau Exchange, Looker Marketplace, Sigma Template Gallery

---

### I14. Adaptive Report Caching with Staleness Budget
**Category**: Performance | **Justification**: Fixed TTL caches waste freshness budget on slow-changing data (daily aggregates) while under-caching fast-changing data (real-time ops metrics). Adaptive TTL sets cache lifetime proportional to the data's measured change velocity — the Presto / Trino query result reuse model. | **Implementation**: Track `change_velocity` per datasource as `(new_row_count - cached_row_count) / elapsed_seconds` measured across last 5 runs; compute adaptive TTL as `base_ttl / max(change_velocity, 0.001)`; clamp to `[min_ttl_secs=60, max_ttl_secs=86400]`; expose `cache_ttl_seconds` and `staleness_budget_pct` in run response; configurable per report via `CachePolicy` model; tenant admin can override per datasource. | **Competitor**: Trino (result caching), ClickHouse (query cache), Redshift (result cache)

---

### I15. Scheduled Report Health Monitoring with SLA Alerting
**Category**: Reliability | **Justification**: Scheduled reports silently fail due to datasource timeouts, schema changes, or permission revocations — users discover the failure when they need the report urgently. Proactive SLA monitoring with escalating alerts (warn at 15 min late, critical at 30 min, page-on-call at 60 min) mirrors PagerDuty-style reliability engineering applied to reporting. | **Implementation**: `async check_schedule_health(tenant_id)` computes for each active schedule: `expected_next_run`, `actual_last_run`, `latency_seconds`, `sla_status` (on_time / late / breached); `async get_schedule_sla_report(tenant_id, period)` aggregates SLA attainment % per schedule; `async register_schedule_sla(tenant_id, schedule_id, warn_minutes, critical_minutes, on_call_group)` stores policy; integration with ntfy sends tiered alerts at each breach level; SLA breach count stored on schedule record. | **Competitor**: dbt Cloud (job monitoring), Airbyte (connector health), Prefect (SLA enforcement)

---

*© 2025 Datacraft — www.datacraft.co.ke*
