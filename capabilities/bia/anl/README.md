# Analytics Engine

## Overview
The Analytics Engine (`bia_anl`) provides the core analytical computation runtime for the BIA domain. It delivers ad-hoc SQL/MDX query execution, OLAP cube management, semantic dimension layer, metric definition and calculation, multi-datasource connectivity, TTL result caching, query scheduling, column-level lineage tracking, IQR anomaly detection, A/B test analysis, and governed analytical data access — all tenant-scoped.

## Capability ID
`bia_anl`

## Provides
- `ad_hoc_query_execution` — Parameterised SQL/MDX against registered datasources
- `olap_cube_management` — Create, refresh, archive, drill-down, slice, dice
- `metric_definition_registry` — Define, version, govern calculated business metrics with goal tracking
- `analytical_data_access` — Governed access layer with access-level enforcement
- `query_result_cache` — TTL-keyed result cache (session/hourly/daily/weekly) with per-query invalidation
- `datasource_connectivity` — Register and test connections to 9 datasource types
- `saved_query_library` — Store, share, version-control, and diff reusable queries
- `query_scheduling` — Schedule recurring query execution via the `schd` capability
- `result_export` — Export to JSON, CSV, Parquet, Arrow, XLSX, HTML
- `semantic_layer` — Define reusable named dimensions; resolve semantic queries to SQL
- `column_lineage` — Track upstream/downstream column-level data lineage per query
- `anomaly_detection` — IQR-fence anomaly scoring on metric time-series with severity classification
- `result_pivot` — Server-side cross-tab of columnar results with Decimal aggregation
- `percentile_statistics` — Compute P10/P25/P50/P75/P90/P95/P99 with Decimal precision
- `metric_goal_tracking` — Period targets for metrics; on_track/at_risk/off_track variance status
- `execution_queue` — Priority-lane query queue (interactive/batch/background) with SLA estimates
- `cohort_analysis` — Retention matrices indexed by cohort x period
- `funnel_analysis` — Multi-step conversion funnel with drop-off rates
- `attribution_modelling` — First/last/linear/time-decay/data-driven touchpoint attribution
- `segmentation` — Filter-criteria audience segments with estimated size
- `ab_test_analysis` — Two-tailed Z-test with lift, p-value, and Bonferroni correction

## Requires
| Capability | Reason |
|------------|--------|
| auth | User identity and permission checks |
| audl | Audit trail for all query executions |
| mten | Tenant context enforcement |
| conf | Runtime configuration management |
| schd | Scheduled query execution |
| mqeb | Streaming query lifecycle events |
| moni | Operational monitoring of query performance |
| nlpc | Natural-language query parsing (future) |

## Configuration
| Option | Default | Description |
|--------|---------|-------------|
| max_rows_per_query | 100,000 | Hard row limit per query |
| timeout_seconds | 300 | Query execution timeout |
| default_cache_policy | session | Cache lifetime for results |
| require_approval_for_public | true | Public queries require approval |
| credentials_vault_required | true | Datasource credentials must be in vault |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/anl/queries | GET | List saved queries | bia_anl:query |
| /api/bia/anl/queries | POST | Save a new query | bia_anl:query |
| /api/bia/anl/queries/\<id\> | GET | Get query detail | bia_anl:query |
| /api/bia/anl/queries/\<id\> | PUT | Update query (auto-snapshots version) | bia_anl:query |
| /api/bia/anl/queries/\<id\> | DELETE | Delete query | bia_anl:query |
| /api/bia/anl/queries/\<id\>/execute | POST | Execute query (cached) | bia_anl:query |
| /api/bia/anl/queries/\<id\>/versions | GET | List SQL versions | bia_anl:query |
| /api/bia/anl/queries/\<id\>/diff | POST | Diff two SQL versions | bia_anl:query |
| /api/bia/anl/cubes | GET | List OLAP cubes | bia_anl:cubes |
| /api/bia/anl/cubes | POST | Create cube | bia_anl:cubes |
| /api/bia/anl/cubes/\<id\>/slice | POST | OLAP slice | bia_anl:cubes |
| /api/bia/anl/cubes/\<id\>/dice | POST | OLAP dice | bia_anl:cubes |
| /api/bia/anl/metrics | GET | List metrics | bia_anl:metrics |
| /api/bia/anl/metrics | POST | Define metric | bia_anl:metrics |
| /api/bia/anl/metrics/\<id\>/goal | POST | Set metric goal | bia_anl:metrics |
| /api/bia/anl/metrics/\<id\>/variance | POST | Compute variance vs goal | bia_anl:metrics |
| /api/bia/anl/metrics/\<id\>/anomalies | POST | Detect time-series anomalies | bia_anl:metrics |
| /api/bia/anl/dimensions | GET | List semantic dimensions | bia_anl:query |
| /api/bia/anl/dimensions | POST | Define semantic dimension | bia_anl:query |
| /api/bia/anl/semantic/resolve | POST | Resolve semantic query to SQL | bia_anl:query |
| /api/bia/anl/lineage | POST | Track column lineage | bia_anl:admin |
| /api/bia/anl/lineage/\<col\> | GET | Get column lineage chain | bia_anl:query |
| /api/bia/anl/pivot | POST | Pivot result set | bia_anl:query |
| /api/bia/anl/percentiles | POST | Compute percentiles | bia_anl:query |
| /api/bia/anl/queue | POST | Enqueue query | bia_anl:query |
| /api/bia/anl/queue/status | GET | Queue depth per lane | bia_anl:query |
| /api/bia/anl/datasources | GET | List datasources | bia_anl:admin |
| /api/bia/anl/datasources | POST | Register datasource | bia_anl:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| cross_tenant_query_denied | Cross-tenant access | deny |
| query_timeout_enforced | Timeout exceeded | deny |
| max_rows_enforced | Row limit exceeded | deny |
| public_access_requires_approval | access_level=public, not approved | deny |
| datasource_credentials_vault_required | Credentials not in vault | deny |
| stale_cube_read_allowed_with_warning | Cube state=stale | allow with metadata |

## New Features in v2.0

### Query Version Control
Every `update_query` call snapshots the pre-update SQL as an immutable version. Use `get_query_versions` to list and `diff_query_versions` for unified diffs between any two.

### Semantic Dimension Layer
Define SQL fragments as named dimensions once via `define_dimension`, reference them by name in `resolve_semantic_query`. The service assembles a valid SELECT + GROUP BY without requiring analysts to repeat JOIN boilerplate.

### Result Cache with TTL
`execute_query_cached` uses SHA-256-keyed result cache. TTL from `cache_policy` field on the query. `invalidate_query_cache` evicts all variants for a query.

### Metric Goals and Variance
`set_metric_goal` attaches a Decimal target to a metric for a named period. `compute_metric_variance` returns absolute and percentage variance, plus `on_track/at_risk/off_track` status. All arithmetic uses Decimal.

### IQR-Fence Anomaly Detection
`detect_metric_anomalies` applies Tukey's IQR fence to a time-series, classifies each anomalous point with a distance score, and returns `low/medium/high/none` severity.

### OLAP Slice and Dice
`olap_slice` fixes one dimension to a single member. `olap_dice` restricts multiple dimensions to specified member sets. Both return structured cell sets.

### Column-Level Lineage
`track_lineage` records source-to-target column mappings per query. `get_lineage` retrieves upstream/downstream chains for any fully-qualified column.

### Result Pivot
`pivot_result` cross-tabs row dicts: pivot-column values become headers, values aggregated using sum/avg/count/max/min with Decimal arithmetic.

### Percentile Statistics
`compute_percentiles` returns P10-P99 for any numeric value list using linear interpolation with Decimal-typed results.

### Priority Execution Queue
`enqueue_query` places queries into interactive/batch/background lanes (SLA 5s/60s/900s). `get_queue_status` returns depth and estimated wait per lane.

## Quick Usage

```python
import asyncio
from decimal import Decimal
from capabilities.bia.anl.service import AnalyticsEngineService

svc = AnalyticsEngineService(tenant_id="acme", actor_id="analyst_1")

async def demo():
    ds = await svc.register_datasource(
        "acme", "Sales DB", "postgresql",
        {"host": "db.internal", "port": 5432, "database": "sales"},
        "vault/acme/salesdb", "analyst_1",
    )
    q = await svc.save_query(
        "acme", "Revenue", "adhoc_sql",
        "SELECT region, SUM(amount) FROM orders WHERE month = {{month}} GROUP BY 1",
        ds["id"], "analyst_1", cache_policy="hourly",
    )
    r1 = await svc.execute_query_cached("acme", q["id"], {"month": "2026-05"})
    r2 = await svc.execute_query_cached("acme", q["id"], {"month": "2026-05"})
    assert r2["cached"] is True

    # Metric goal + variance
    metric = await svc.define_metric(
        "acme", "Gross Revenue", "financial", "SUM(amount)", "cube-id", "analyst_1", unit="USD",
    )
    await svc.set_metric_goal("acme", metric["id"], Decimal("1000000"), "2026-Q2", "analyst_1")
    variance = await svc.compute_metric_variance("acme", metric["id"], Decimal("940000"), "2026-Q2")
    print(variance["status"])  # at_risk

asyncio.run(demo())
```

## Data Models
- `DatasourceResponse`: id, tenant_id, name, datasource_type, connection_config, credentials_vault_ref, owner_id
- `QueryResponse`: id, tenant_id, name, query_type, sql_text, datasource_id, access_level, cache_policy, owner_id
- `QueryVersion`: version_number, sql_text, updated_by, updated_at
- `CubeResponse`: id, tenant_id, name, dimensions, measures, grain_sql, state, owner_id, last_refreshed_at
- `MetricResponse`: id, tenant_id, name, metric_type, formula, cube_id, unit, owner_id
- `MetricGoal`: id, tenant_id, metric_id, target_value (Decimal str), period, tolerance_pct
- `MetricVariance`: actual_value, target_value, abs_variance, pct_variance, status
- `DimensionResponse`: id, tenant_id, name, sql_expression, datasource_id, data_type
- `SemanticQueryResult`: generated_sql, resolved_dimensions, metrics
- `LineageEntry`: id, query_id, source_columns, target_columns, transformation
- `PivotResult`: pivot_column, value_column, agg_function, pivot_values, rows
- `PercentileResult`: column, n, min, max, mean, stdev, percentiles (dict)
- `AnomalyResult`: anomaly_points, lower_fence, upper_fence, severity
- `QueueEntry`: id, priority, status, estimated_wait_seconds, sla_seconds
- `QueryResultResponse`: query_id, columns, rows, row_count, execution_time_ms, cached, cache_age_seconds

## Streaming Events
- query_executed, query_saved, query_scheduled, query_enqueued, query_cache_invalidated
- cube_created, cube_refreshed, cube_archived, olap_slice, olap_dice
- metric_defined, metric_updated, metric_goal_set, metric_variance_computed
- anomaly_detection_run, lineage_tracked, lineage_queried
- dimension_defined, semantic_query_resolved
- result_pivoted, percentiles_computed, query_versions_diffed

## Edge Cases Handled
- Stale cube reads return data with staleness metadata rather than blocking
- Archived cubes reject refresh requests with explicit restore instruction
- Shared queries can only be deleted by the owner
- Row and timeout limits enforced per-query regardless of datasource type
- Credentials are never stored inline; vault reference mandatory
- Result cache keys are SHA-256 hashed; raw SQL never used as cache key
- Anomaly detection requires at least 4 data points for IQR calculation
- Metric variance raises ValueError if no goal exists for the given period

## Composability Notes
- Feeds results into `dsh` (Dashboard Management) for widget data binding
- Metrics feed into `rpt` (Report Builder) for parameterised reports
- Cube refresh orchestrated by `wflo` (Workflow) with approval gates
- `nlpc` translates natural-language questions into SQL
- `moni` tracks query latency and cube staleness for SLA alerting
- Lineage data consumed by `catl` (Data Catalog) for governance dashboards
- Goal variance feeds `alets` (Alerts) for threshold-based notifications
