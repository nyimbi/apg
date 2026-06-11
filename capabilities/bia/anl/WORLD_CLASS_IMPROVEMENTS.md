# Analytics Engine (bia_anl) — World-Class Improvement Plan

**Date**: 2026-06-11 | **Domain**: BIA | **Author**: Nyimbi Odero

---

## 1. Materialized Result Cache with TTL Invalidation

**Category**: Performance

**Justification**: The current `cache_policy` field is stored but never enforced — every query re-executes. Production analytical engines (Looker, Metabase) implement result-level caching keyed on `(tenant_id, query_hash, parameter_hash)` with TTL-based invalidation. Without this, repeated dashboard loads hammer the datasource unnecessarily.

**Implementation**: Add a `_result_cache: dict[str, tuple[dict, float]]` keyed by SHA-256 of `(tenant_id + sql + params_json)`. On `execute_query` and `ad_hoc_query`, check cache first; populate on miss with monotonic expiry derived from `cache_policy` enum ("session"→30 min, "hourly"→60 min, "daily"→1440 min). Expose `invalidate_cache(tenant_id, query_id)` and `purge_tenant_cache(tenant_id)` methods.

**Competitor Reference**: Looker's "datagroup" TTL cache; Apache Superset's query-result cache backed by Redis.

---

## 2. Query Explain / Cost Estimation

**Category**: Developer Experience

**Justification**: EXPLAIN ANALYZE output is the primary tool analysts use to debug slow queries. No current path exists to retrieve query plans before execution. Redash and Metabase both surface query plans inline in the editor.

**Implementation**: Add `async explain_query(tenant_id, sql_text, datasource_id, format="text")` that runs `EXPLAIN (FORMAT JSON, ANALYZE false)` via the datasource adapter, parses the plan tree, and returns structured cost, rows, and width estimates per node. Guard with `guard_tenant_id`. Expose `POST /api/bia/anl/queries/explain`.

**Competitor Reference**: pgAdmin's "Explain" tab; Redash query editor explain panel.

---

## 3. Column-Level Lineage Tracking

**Category**: Governance / Data Catalog

**Justification**: GDPR and SOC2 require knowing which columns flow into which reports. Without lineage, impact analysis for schema changes is manual. Atlan, Alation, and dbt Core all maintain column-level lineage graphs.

**Implementation**: Add `async track_lineage(tenant_id, query_id, lineage_graph)` where `lineage_graph` is `{"source_columns": [...], "target_columns": [...], "transformation": str}`. Store in `_lineage: dict[tuple, list[dict]]`. Add `async get_lineage(tenant_id, column_fqn)` to fetch upstream/downstream chains. Emit `lineage_tracked` audit event.

**Competitor Reference**: dbt's `column_lineage` manifest; Atlan's lineage graph API.

---

## 4. Semantic Layer with Reusable Dimension Joins

**Category**: Modelling / Reuse

**Justification**: Analysts repeatedly write the same JOIN boilerplate. A semantic layer (as in Cube.js or dbt Semantic Layer) lets you define dimensions once and reference them by name across queries. Current service has no join abstraction.

**Implementation**: Add `async define_dimension(tenant_id, name, sql_expression, datasource_id, owner_id)` and `async resolve_semantic_query(tenant_id, metrics, dimensions, filters)` that expands named dimensions into full SQL. Store dimension definitions in `_dimensions: dict[tuple, dict]`. Resolution engine assembles valid SELECT with GROUP BY from semantic names.

**Competitor Reference**: Cube.js semantic layer; dbt Semantic Layer (MetricFlow).

---

## 5. Percentile and Statistical Aggregation Functions

**Category**: Computation

**Justification**: Business analysts routinely need P50/P95/P99 latency, revenue percentiles, and Gini coefficients. The current `calculated_metric` eval only supports basic arithmetic. Tableau and Power BI ship rich statistical functions out of the box.

**Implementation**: Extend `calculated_metric` safe_globals with `statistics` stdlib functions: `median`, `stdev`, `quantiles`, `harmonic_mean`. Add `async compute_percentiles(tenant_id, dataset_id, column, percentiles)` that returns named percentile values using the `statistics.quantiles` call path with explicit interpolation. Return `Decimal`-typed values for money columns.

**Competitor Reference**: Tableau's `PERCENTILE()` and `WINDOW_PERCENTILE()` table calcs; Power BI's `PERCENTILEINC`.

---

## 6. Incremental Cube Refresh with Watermark

**Category**: Performance / Correctness

**Justification**: Full cube rebuilds are expensive. Systems like Apache Druid and Kylin support incremental segment ingestion keyed on a time watermark column. Without this, cube.refresh_cube triggers a full rebuild regardless of how much data changed.

**Implementation**: Add `watermark_column` and `last_watermark_value` fields to cube records. `async incremental_refresh_cube(tenant_id, cube_id, new_watermark)` updates only rows where `watermark_column > last_watermark_value`, stores the new watermark, and transitions state to `active`. Track `rows_added` vs `rows_scanned` in the response.

**Competitor Reference**: Apache Kylin incremental build; Druid's segment granularity and compaction.

---

## 7. Multi-Dimensional Slice / Dice API

**Category**: OLAP

**Justification**: Drill-down is one operation; slice (fix one dimension value) and dice (fix multiple dimensions to a sub-cube) are equally important OLAP operations. The current API only exposes `olap_drill_down`. Analysts using tools like Mondrian or Microsoft SSAS expect the full OLAP operation set.

**Implementation**: Add `async olap_slice(tenant_id, cube_id, dimension, member, measures)` and `async olap_dice(tenant_id, cube_id, dimension_members: dict[str, list], measures)`. Both validate that the cube is in `active` state, apply dimension restrictions to the cell-set generation, and return structured responses with `cell_count`, `execution_time_ms`, and `filtered_dimensions`.

**Competitor Reference**: Microsoft SSAS MDX Slice/Dice; Apache Kylin REST `query/tables` API.

---

## 8. Anomaly Detection on Metric Time-Series

**Category**: AI / Alerting

**Justification**: Manual threshold alerting breaks when seasonality changes. Time-series anomaly detection (as in Datadog Monitors or Grafana's ML-based alerting) catches real anomalies automatically. APG already has Ollama-backed ML in `ml_analysis_narrate` — the same infrastructure can run local anomaly scoring.

**Implementation**: Add `async detect_metric_anomalies(tenant_id, metric_id, time_series, sensitivity)` that applies a simple IQR-fence (Q1 - k*IQR, Q3 + k*IQR) plus optional local Ollama-backed scoring. Return `anomaly_points: list[{ts, value, score, reason}]`, severity (`low/medium/high`), and recommended alert thresholds. Use `Decimal` for all numeric comparisons.

**Competitor Reference**: Datadog metric anomaly detection; Grafana ML-based Spikedetect.

---

## 9. Query Version Control and Diff

**Category**: Governance / Collaboration

**Justification**: Analysts iterate on queries; without versioning, deleted or overwritten SQL is unrecoverable. GitHub Copilot for data (Hex, Deepnote) and Metabase both version queries. Audit log stores events but not SQL snapshots.

**Implementation**: Add `_query_versions: dict[tuple, list[dict]]` storing immutable version snapshots on every `update_query` call (version_number, sql_text, updated_by, updated_at). Add `async get_query_versions(tenant_id, query_id)` and `async diff_query_versions(tenant_id, query_id, v1, v2)` that returns a unified diff string using `difflib.unified_diff`.

**Competitor Reference**: Hex query versioning; Metabase query revision history.

---

## 10. Cross-Datasource Federated Query

**Category**: Multi-Datasource

**Justification**: Analysts frequently need to JOIN data across two datasources (e.g., PostgreSQL transactions with S3 event logs). The current service treats each datasource independently. Trino and Presto are purpose-built for this; lightweight federation can be simulated at the service layer.

**Implementation**: Add `async federated_query(tenant_id, sources: list[{datasource_id, alias, sql}], join_expression, output_columns)` that validates all referenced datasource IDs belong to the tenant, assembles a virtual plan, returns a synthetic result set with full cross-source attribution, and records all source datasource IDs in the audit event.

**Competitor Reference**: Trino federated catalog; AWS Athena federation with Lambda connectors.

---

## 11. Metric Goal Tracking and Variance Analysis

**Category**: Business Metrics

**Justification**: Metrics in isolation are meaningless without targets. CFOs compare actuals vs. budget. Looker's "Goals" and Salesforce Einstein Analytics both support target-vs-actual KPI tracking with variance decomposition.

**Implementation**: Add `async set_metric_goal(tenant_id, metric_id, target_value, period, owner_id)` storing goals in `_goals`. Add `async compute_metric_variance(tenant_id, metric_id, actual_value, period)` that returns `{actual, target, abs_variance, pct_variance, status: "on_track|at_risk|off_track"}`. Use `Decimal` throughout for financial accuracy.

**Competitor Reference**: Looker KPI Goals; Salesforce Einstein Analytics target tracking.

---

## 12. Query Parameterisation with Type-Safe Bindings

**Category**: Security / Correctness

**Justification**: The current `parameters` field is a freeform dict. SQL injection is possible if parameter values are interpolated naively. All production databases use parameterised queries (SQLAlchemy bindparams, psycopg3 `%s` bindings). Type-safe bindings also enable UI form generation.

**Implementation**: Add `async validate_query_parameters(tenant_id, query_id, parameters)` that parses `{{param_name:type}}` placeholders in `sql_text`, validates each supplied parameter against its declared type (`int`, `str`, `date`, `Decimal`), and returns `{valid: bool, errors: list, bound_sql: str}`. Raise `ValueError` on type mismatch before any execution.

**Competitor Reference**: Metabase variable binding; Redash `{{parameter}}` syntax with type declarations.

---

## 13. Execution Plan Queue with Priority Lanes

**Category**: Resource Management

**Justification**: Long-running cube refreshes should not block short interactive queries. Snowflake uses multi-cluster virtual warehouses; Databricks SQL uses queue priorities. Without queueing, a rogue full-table scan starves all other tenants.

**Implementation**: Add `async enqueue_query(tenant_id, query_id, priority: Literal["interactive","batch","background"])` that places the query in a priority-sorted `_queue: list[dict]`. Add `async get_queue_status(tenant_id)` returning queue depth per lane and estimated wait times. Emit `query_enqueued` / `query_dequeued` audit events.

**Competitor Reference**: Snowflake multi-cluster warehouse queuing; Databricks SQL serverless queue.

---

## 14. Natural-Language Query Generation (NL2SQL)

**Category**: AI / Accessibility

**Justification**: Most business users cannot write SQL. Tableau's Ask Data, Microsoft Copilot for Power BI, and ThoughtSpot all ship NL-to-SQL. APG's existing Ollama integration (`ml_analysis_narrate`) provides the scaffolding.

**Implementation**: Add `async nl_to_sql(tenant_id, question: str, datasource_id, schema_context: dict, actor_id)` that constructs a schema-aware prompt, calls the local Ollama endpoint (`/api/generate`), extracts the SQL block, validates it with `sqlparse`, and returns `{sql, confidence, explanation}`. Fall back gracefully when `OLLAMA_BASE_URL` is absent.

**Competitor Reference**: Tableau Ask Data; ThoughtSpot SpotIQ; Microsoft Power BI Copilot.

---

## 15. Result Pivot and Transpose API

**Category**: UX / Presentation

**Justification**: Columnar query results often need to be pivoted before rendering in dashboards (rows become columns, one dimension becomes header). Excel pivot tables, Superset's pivot chart, and Looker's table calculations all provide this. It is trivially done at the service layer, sparing clients from implementing it.

**Implementation**: Add `async pivot_result(tenant_id, result_id, pivot_column, value_column, agg_function: Literal["sum","avg","count","max","min"])` that takes a stored result (keyed by `result_id`), groups rows by all non-pivot columns, and builds a sparse matrix with pivot column values as headers. Return `{pivot_columns, rows, row_count}`. Use `Decimal` for numeric aggregation.

**Competitor Reference**: Apache Superset pivot table; Excel PivotTable; Looker pivot table calcs.
