# Analytics Engine — User Guide

**Capability ID**: `bia_anl` | **Domain**: `bia` | **Version**: `2.0.0`

## Description

The Analytics Engine (`bia_anl`) provides the core analytical computation runtime for the BIA domain. It delivers ad-hoc SQL/MDX query execution, OLAP cube management, semantic dimension layer, metric definition and calculation with goal tracking, multi-datasource connectivity, TTL result caching, query scheduling, column-level lineage tracking, IQR anomaly detection, A/B test analysis, cohort/funnel/attribution analysis, audience segmentation, and a priority execution queue — all tenant-scoped.

## Installation

```bash
pip install apg-bia-anl
```

## Quick Start

```python
import asyncio
from decimal import Decimal
from capabilities.bia.anl.service import AnalyticsEngineService

svc = AnalyticsEngineService(tenant_id="acme", actor_id="analyst_1")

async def main():
    ds = await svc.register_datasource(
        tenant_id="acme",
        name="Sales DB",
        datasource_type="postgresql",
        connection_config={"host": "db.acme.internal", "port": 5432, "database": "sales"},
        credentials_vault_ref="vault/acme/salesdb",
        owner_id="analyst_1",
    )
    q = await svc.save_query(
        tenant_id="acme",
        name="Monthly Revenue",
        query_type="adhoc_sql",
        sql_text="SELECT region, SUM(amount) FROM orders WHERE month = {{month}} GROUP BY 1",
        datasource_id=ds["id"],
        owner_id="analyst_1",
        cache_policy="hourly",
    )
    r1 = await svc.execute_query_cached("acme", q["id"], {"month": "2026-05"})
    r2 = await svc.execute_query_cached("acme", q["id"], {"month": "2026-05"})
    assert r2["cached"] is True   # second call is served from cache

asyncio.run(main())
```

---

## Core Concepts

### Tenant Isolation

Every method requires a `tenant_id`. All storage is keyed on `(tenant_id, entity_id)`. Cross-tenant access raises a `ValueError` via the capability rule engine.

### Audit Trail

Every mutation appends a structured audit event to `self._audit`. Pass an `audit` adapter to the constructor to forward events to an external system.

### Rule Enforcement

`_enforce(context)` calls the capability rule engine before every operation. Violations raise `ValueError` with the matched rule and required action.

---

## Feature Reference

### 1. Datasource Management

```python
ds = await svc.register_datasource(
    tenant_id="acme",
    name="Warehouse",
    datasource_type="bigquery",
    connection_config={"project": "acme-prod"},
    credentials_vault_ref="vault/acme/bq",
    owner_id="admin_1",
)
status = await svc.test_datasource("acme", ds["id"])
sources = await svc.list_datasources("acme")
```

---

### 2. Saved Query Library

```python
q = await svc.save_query(
    tenant_id="acme",
    name="Top Products",
    query_type="adhoc_sql",
    sql_text="SELECT sku, SUM(qty) FROM sales GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
    datasource_id=ds["id"],
    owner_id="analyst_1",
    cache_policy="daily",
)
result = await svc.execute_query("acme", q["id"], {})
```

---

### 3. Query Version Control

Every `update_query` call snapshots the pre-mutation SQL.

```python
await svc.update_query("acme", q["id"], {"sql_text": "SELECT sku, AVG(qty) FROM sales GROUP BY 1"})
versions = await svc.get_query_versions("acme", q["id"])
# [{"version_number": 1, "sql_text": "...", "updated_by": "analyst_1", "updated_at": "..."}]

diff = await svc.diff_query_versions("acme", q["id"], version_a=1, version_b=2)
print(diff["diff"])           # unified diff string
print(diff["lines_added"])    # int
print(diff["lines_removed"])  # int
```

---

### 4. Result Cache with TTL

```python
# Cache miss: executes and stores
r1 = await svc.execute_query_cached("acme", q["id"], {"month": "2026-05"})
assert r1["cached"] is False

# Cache hit within TTL
r2 = await svc.execute_query_cached("acme", q["id"], {"month": "2026-05"})
assert r2["cached"] is True
print(r2["cache_age_seconds"])  # float

# Force refresh
r3 = await svc.execute_query_cached("acme", q["id"], {"month": "2026-05"}, force_refresh=True)

# Evict cache for this query
evict = await svc.invalidate_query_cache("acme", q["id"])
print(evict["evicted_entries"])
```

TTL map: session=1800s, hourly=3600s, daily=86400s, weekly=604800s, none=0s.

---

### 5. OLAP Cube Management

```python
cube = await svc.create_cube(
    tenant_id="acme",
    name="Sales Cube",
    datasource_id=ds["id"],
    dimensions=["region", "product_category", "order_month"],
    measures=["revenue", "units"],
    grain_sql="SELECT * FROM fact_sales",
    owner_id="analyst_1",
)
await svc.refresh_cube("acme", cube["id"])

# Drill down
drill = await svc.olap_drill_down("acme", cube["id"], dimension="region", level="country")

# Slice: fix one dimension to a single member
sliced = await svc.olap_slice("acme", cube["id"], dimension="region", member="APAC")

# Dice: restrict multiple dimensions simultaneously
diced = await svc.olap_dice(
    "acme", cube["id"],
    dimension_members={"region": ["APAC", "EMEA"], "product_category": ["SaaS"]},
)
```

Cube states: building -> active -> stale (after update) / archived.

---

### 6. Metric Goals and Variance

```python
metric = await svc.define_metric(
    "acme", "Gross Revenue", "financial", "SUM(amount)", cube["id"], "analyst_1", unit="USD",
)

# Attach a target
await svc.set_metric_goal(
    tenant_id="acme",
    metric_id=metric["id"],
    target_value=Decimal("1_000_000"),
    period="2026-Q2",
    owner_id="analyst_1",
    tolerance_pct=5.0,
)

# Compare actual vs target
variance = await svc.compute_metric_variance(
    tenant_id="acme",
    metric_id=metric["id"],
    actual_value=Decimal("940_000"),
    period="2026-Q2",
)
print(variance["status"])        # at_risk  (6% shortfall, 5% tolerance)
print(variance["pct_variance"])  # "-6.0000"
```

Status thresholds:
- `on_track`: |pct_variance| <= tolerance_pct
- `at_risk`: tolerance_pct < |pct_variance| <= 2 * tolerance_pct
- `off_track`: |pct_variance| > 2 * tolerance_pct

---

### 7. Anomaly Detection (IQR Fence)

```python
ts = [
    {"ts": "2026-01", "value": 1000},
    {"ts": "2026-02", "value": 1050},
    {"ts": "2026-03", "value": 980},
    {"ts": "2026-04", "value": 1020},
    {"ts": "2026-05", "value": 9999},   # spike
    {"ts": "2026-06", "value": 1010},
    {"ts": "2026-07", "value": 995},
    {"ts": "2026-08", "value": 1030},
]
anomalies = await svc.detect_metric_anomalies("acme", metric["id"], ts, sensitivity=1.5)
print(anomalies["severity"])          # high
print(anomalies["anomaly_count"])     # 1
print(anomalies["anomaly_points"][0]["score"])  # distance from fence in IQR units
```

Requires >= 4 data points. sensitivity=3.0 detects only extreme outliers.

---

### 8. Semantic Dimension Layer

```python
await svc.define_dimension(
    tenant_id="acme",
    name="order_month",
    sql_expression="DATE_TRUNC('month', order_date)",
    datasource_id=ds["id"],
    owner_id="analyst_1",
)
sem = await svc.resolve_semantic_query(
    tenant_id="acme",
    metrics=["SUM(amount) AS revenue"],
    dimensions=["order_month", "region"],
    filters={"status": "completed"},
)
print(sem["generated_sql"])
# SELECT DATE_TRUNC('month', order_date), region, SUM(amount) AS revenue
# FROM __semantic_table__
# WHERE status = 'completed'
# GROUP BY DATE_TRUNC('month', order_date), region
```

---

### 9. Column-Level Lineage

```python
await svc.track_lineage(
    tenant_id="acme",
    query_id=q["id"],
    source_columns=["sales.order_date", "sales.amount"],
    target_columns=["report.revenue_month"],
    transformation="DATE_TRUNC month + SUM aggregation",
)
lineage = await svc.get_lineage("acme", "report.revenue_month", direction="upstream")
print(lineage["upstream_count"])  # 1
```

---

### 10. Result Pivot

```python
rows = [
    {"region": "APAC", "month": "2026-04", "revenue": "120000"},
    {"region": "EMEA", "month": "2026-04", "revenue": "95000"},
    {"region": "APAC", "month": "2026-05", "revenue": "130000"},
    {"region": "EMEA", "month": "2026-05", "revenue": "100000"},
]
pivoted = await svc.pivot_result(
    tenant_id="acme",
    rows=rows,
    pivot_column="region",
    value_column="revenue",
    row_key_columns=["month"],
    agg_function="sum",
)
# pivoted["pivot_values"] == ["APAC", "EMEA"]
# pivoted["rows"] == [
#   {"month": "2026-04", "APAC": "120000", "EMEA": "95000"},
#   {"month": "2026-05", "APAC": "130000", "EMEA": "100000"},
# ]
```

---

### 11. Percentile Statistics

```python
pcts = await svc.compute_percentiles(
    tenant_id="acme",
    dataset_id=ds["id"],
    column="order_value",
    values=[12.5, 34.0, 55.0, 88.0, 102.0, 250.0, 1200.0],
    percentiles=[0.5, 0.75, 0.95, 0.99],
)
print(pcts["percentiles"])  # {"p50": "88.0000", "p75": "...", ...}
print(pcts["mean"])         # Decimal string
```

---

### 12. Priority Execution Queue

```python
entry = await svc.enqueue_query("acme", q["id"], priority="interactive")
bg = await svc.enqueue_query("acme", q["id"], priority="background", parameters={"region": "ALL"})
status = await svc.get_queue_status("acme")
print(status["lanes"]["interactive"]["depth"])          # 1
print(status["lanes"]["background"]["sla_seconds"])     # 900
```

Priority order: interactive (SLA 5s) > batch (SLA 60s) > background (SLA 900s).

---

### 13. Cohort Analysis

```python
cohort = await svc.cohort_analysis(
    tenant_id="acme",
    cohort_definition={"segment_by": "signup_month", "entity": "user_id"},
    metrics=["retention", "revenue"],
    periods=["2026-01", "2026-02", "2026-03", "2026-04"],
)
```

---

### 14. Funnel Analysis

```python
funnel = await svc.funnel_analysis(
    tenant_id="acme",
    steps=[
        {"name": "Landing Page", "event": "page_view"},
        {"name": "Sign Up", "event": "signup"},
        {"name": "First Purchase", "event": "purchase"},
    ],
    window_hours=72,
)
print(funnel["overall_conversion_rate"])
```

---

### 15. Attribution Modelling

```python
attr = await svc.attribution_modelling(
    tenant_id="acme",
    touchpoints=[
        {"channel": "organic_search", "timestamp": "2026-05-01"},
        {"channel": "email", "timestamp": "2026-05-03"},
        {"channel": "paid_search", "timestamp": "2026-05-05"},
    ],
    conversion_event="purchase",
    model="time_decay",
)
```

---

### 16. A/B Test Analysis

```python
analysis = await svc.ab_test_analysis(
    tenant_id="acme",
    experiment_id="exp_homepage_v2",
    metric="conversion_rate",
    confidence_level=0.95,
)
print(analysis["statistically_significant"])  # True/False
print(analysis["recommendation"])             # "ship_variant" | "retain_control"
```

---

## Flask-AppBuilder Blueprint Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bia/anl/dashboard` | `bia_anl:view` | Overview |
| `/bia/anl/query-builder` | `bia_anl:query` | Querying |
| `/bia/anl/saved-queries` | `bia_anl:query` | Querying |
| `/bia/anl/cubes` | `bia_anl:cubes` | OLAP |
| `/bia/anl/metrics` | `bia_anl:metrics` | Metrics |
| `/bia/anl/dimensions` | `bia_anl:query` | Semantic Layer |
| `/bia/anl/lineage` | `bia_anl:admin` | Governance |
| `/bia/anl/queue` | `bia_anl:query` | Execution |

---

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `max_rows_per_query` | 100000 | Hard row limit per execution |
| `timeout_seconds` | 300 | Query execution timeout |
| `default_cache_policy` | `session` | Default result TTL policy |
| `require_approval_for_public` | `true` | Public queries require approval |
| `credentials_vault_required` | `true` | Credentials must reference vault |
| `anomaly_default_sensitivity` | `1.5` | Default IQR multiplier |
| `queue_interactive_sla_seconds` | `5` | SLA for interactive priority lane |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_BASE_URL` | No | Enables ML-backed analytics narration |

## Interoperability

```apg
use bia_anl;
```

- Results feed `dsh` (Dashboard Management) for widget data binding
- Metrics feed `rpt` (Report Builder) for parameterised reports
- Cube refresh orchestrated by `wflo` (Workflow) with approval gates
- Lineage data consumed by `catl` (Data Catalog) for governance dashboards
- Goal variance feeds `alets` (Alerts) for threshold-based notifications

## Further Reading

- `service.py` — Full business logic (1900+ lines, 50+ async methods)
- `models.py` — SQLAlchemy and Pydantic data models
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder blueprint views
- `capability_contract.py` — Rule engine and contract definition
- `README.md` — Quick reference and API route table
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural improvement proposals with competitor references
