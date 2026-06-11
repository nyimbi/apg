# obs_mtx User Guide — Metrics & SLO (v2.0.0)

## Overview

`obs_mtx` is a production-grade observability capability providing:

- **RED Metrics**: Rate, Error rate, Duration percentiles per service.
- **SLO Management**: Multi-type SLO targets with error budget tracking.
- **Burn Rate Alerting**: Single-window and Google SRE dual-window modes.
- **Histogram Buckets**: Prometheus-native `_bucket/_sum/_count` counters.
- **EWMA Anomaly Detection**: Early-warning trend detection on RED metrics.
- **SLO Forecasting**: Linear regression compliance forecast with budget depletion ETA.
- **Composite SLOs**: User-journey-level SLOs aggregated from downstream services.
- **Cardinality Guards**: Block label-explosion before Prometheus OOM.
- **Downsampling**: Fast time-bucketed queries for long retention windows.
- **Error Budget Policies**: Automated actions when budget crosses thresholds.
- **Grafana JSON Export**: Zero-friction Grafana dashboard import.
- **SLO Impact Analysis**: Data-driven target negotiation.
- **Tenant Quotas**: Per-tenant ingestion rate and definition caps.

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| Metric Definition | Named metric with type, unit, and label schema. |
| Data Point | A recorded value at a timestamp with labels. |
| RED Metrics | Aggregate view: request rate, error rate, duration percentiles. |
| SLO | Service Level Objective: target compliance % over a rolling window with an error budget. |
| Error Budget | Allowed non-compliance time = (100 − target) % of the window. |
| Burn Rate | How fast error budget is consumed relative to sustainable rate (1.0 = on track). |
| Histogram Bucket | Cumulative count of observations ≤ bucket boundary (`le` label). |
| EWMA | Exponential Weighted Moving Average — smoothed signal for anomaly detection. |
| Composite SLO | Aggregation of child SLOs into a single user-journey compliance number. |

---

## Quick Start

### 1. Create a metric definition

```http
POST /api/obs/mtx/metrics
X-Tenant-ID: my-org

{
  "name": "payment_requests_total",
  "service_name": "payment-svc",
  "metric_type": "counter",
  "unit": "requests",
  "labels": ["status_code", "method"]
}
```

### 2. Record data points

```http
POST /api/obs/mtx/data-points
X-Tenant-ID: my-org

{
  "metric_name": "payment_requests_total",
  "value": 1,
  "service_name": "payment-svc",
  "labels": {"status_code": "200", "method": "POST"}
}
```

### 3. Record a histogram observation (latency)

Use this instead of raw data points for duration metrics — enables `histogram_quantile` semantics.

```http
POST /api/obs/mtx/data-points/histogram
X-Tenant-ID: my-org

{
  "metric_name": "payment_duration_ms",
  "value": 123.4,
  "service_name": "payment-svc",
  "bucket_boundaries": [5, 10, 25, 50, 100, 250, 500, 1000, 2500]
}
```

Then query quantiles:

```http
GET /api/obs/mtx/data-points/histogram/payment_duration_ms/payment-svc/quantile?q=0.99
```

### 4. Check RED metrics

```http
GET /api/obs/mtx/red/payment-svc?window_minutes=5
X-Tenant-ID: my-org
```

Returns: `request_rate`, `error_rate`, `p50/p95/p99_duration_ms`, `total_requests`, `total_errors`.

### 5. Define an SLO

```http
POST /api/obs/mtx/slos
X-Tenant-ID: my-org

{
  "name": "payment-availability",
  "service_name": "payment-svc",
  "slo_type": "availability",
  "target": 99.9,
  "window_days": 30
}
```

### 6. Create a burn rate alert (dual-window)

```http
POST /api/obs/mtx/burn-rate-alerts
X-Tenant-ID: my-org

{
  "slo_id": "<slo_id>",
  "name": "fast-burn",
  "burn_rate_threshold": 14.4,
  "severity": "critical",
  "short_window_minutes": 60,
  "long_window_minutes": 360
}
```

Evaluate with dual-window semantics (reduces false positives ~60%):

```http
GET /api/obs/mtx/burn-rate-alerts/<alert_id>/evaluate-multiwindow
```

---

## SLO Types

| Type | Measurement Method |
|------|--------------------|
| `availability` | Fraction of `_up` data points equal to 1 |
| `error_rate` | 1 − (errors / requests) using `_errors_total` / `_requests_total` |
| `latency` | Fraction of `_duration_ms` points ≤ `latency_threshold_ms` |
| `throughput` | Request rate relative to target |

---

## Burn Rate Interpretation

| Burn Rate | Meaning | Budget depletes in |
|-----------|---------|-------------------|
| 1.0 | Sustainable | End of window (30 days) |
| 6.0 | Elevated | 5 days |
| 14.4 | Fast burn | 2 hours |
| 36.0 | Critical | 50 minutes |

Google SRE dual-window alerting fires only when BOTH of these are true simultaneously:
- Short window (e.g., 1h) burn rate ≥ threshold
- Long window (e.g., 6h) burn rate ≥ threshold / 2.4

---

## EWMA Anomaly Detection

EWMA maintains smoothed estimates of rate, error rate, and p99 duration. When the current value deviates more than N standard deviations from the EWMA, an anomaly is flagged.

```http
GET /api/obs/mtx/red/payment-svc/anomaly?alpha=0.1&z_score_threshold=3.0
```

Response includes `is_anomalous`, per-dimension `z_score`, and the current EWMA state.

**Tuning**:
- `alpha=0.05` — heavy smoothing, slow response, fewer false positives.
- `alpha=0.3` — lighter smoothing, faster response to genuine spikes.
- `z_score_threshold=2.0` — more sensitive; `3.0` — fewer false positives.

---

## SLO Forecasting

Fit a linear trend to compliance snapshots and project forward:

```http
GET /api/obs/mtx/slos/<slo_id>/forecast?lookahead_hours=24
```

Response:

```json
{
  "predicted_compliance": 99.72,
  "budget_depletion_eta": "2026-06-15T14:30:00+00:00",
  "slope_per_hour": -0.012,
  "confidence": "high",
  "snapshots_used": 48
}
```

`budget_depletion_eta` is the ISO timestamp when compliance is forecast to fall below the SLO target. `null` means no depletion projected.

---

## Composite SLOs

A checkout SLO depends on payment, inventory, and auth services:

```http
POST /api/obs/mtx/composite-slos
{
  "name": "checkout-journey",
  "child_slo_ids": ["<payment_slo>", "<inventory_slo>", "<auth_slo>"],
  "aggregation": "min",
  "description": "Weakest-link checkout composite SLO"
}
```

**Aggregation modes**:

| Mode | Formula | Use Case |
|------|---------|---------|
| `min` | `min(child compliances)` | User experience reflects worst dependency |
| `product` | `P(A) × P(B) × ...` | Independent failure model |
| `weighted_average` | `Σ(weight × compliance)` | Tiered dependency importance |

---

## Cardinality Guard

Before accepting metrics with unknown label sets, check cardinality:

```http
GET /api/obs/mtx/cardinality/payment_requests_total?max_cardinality=10000
```

Response:

```json
{
  "cardinality": 342,
  "over_limit": false,
  "top_labels": [
    {"label": "user_id", "distinct_values": 320},
    {"label": "status_code", "distinct_values": 5}
  ]
}
```

**Recommended practice**: Reject `user_id`, `session_id`, `request_id` as label dimensions. Use trace IDs via exemplars (I2) instead.

---

## Downsampling

Query long time ranges efficiently with bucketed aggregates:

```http
GET /api/obs/mtx/red/payment-svc/downsample?resolution_minutes=60&start_time=2026-06-01T00:00:00Z
```

Returns per-bucket `{min, max, avg, count, p50, p99}`. Results are TTL-cached (default 60s) to avoid rescanning 100k+ points on repeated dashboard refreshes.

---

## Error Budget Policies

Define automated responses when error budget falls below thresholds:

```http
POST /api/obs/mtx/error-budget-policies
{
  "slo_id": "<slo_id>",
  "name": "payment-budget-policy",
  "thresholds": [
    {
      "budget_remaining_pct": 50,
      "action": "freeze_deployments",
      "severity": "warning",
      "message": "Budget below 50% — freeze non-critical deploys"
    },
    {
      "budget_remaining_pct": 10,
      "action": "incident_page",
      "severity": "critical",
      "message": "Budget below 10% — page on-call"
    }
  ]
}
```

Evaluate (with optional async callbacks):

```python
async def freeze_deploys(policy, threshold):
    # Call your deployment system API
    ...

await svc.evaluate_error_budget_policy(
    policy_id,
    action_callbacks={"freeze_deployments": freeze_deploys}
)
```

---

## Grafana JSON Export

Export any dashboard for zero-friction Grafana import:

```http
GET /api/obs/mtx/dashboards/<dash_id>/export/grafana
```

The response is valid Grafana dashboard JSON (schema v36+). Import via:
`Grafana → Dashboards → Import → Paste JSON`.

All panel queries, grid positions, units, and thresholds are mapped from APG panel definitions.

---

## SLO Target Change Analysis

Before tightening an SLO target, simulate the historical impact:

```http
POST /api/obs/mtx/slos/<slo_id>/analyze-target-change
{"proposed_target": 99.95}
```

Response:

```json
{
  "current_target": 99.9,
  "proposed_target": 99.95,
  "would_have_breached_n_times": 3,
  "historical_min_compliance": 99.87,
  "error_budget_delta_minutes": -21.9,
  "feasible": false
}
```

`feasible: false` means the historical minimum compliance never reached the proposed target — the tighter target would have been breached.

---

## Tenant Quotas

Configure per-tenant limits to prevent noisy-neighbour incidents:

```http
POST /api/obs/mtx/quota
{
  "max_points_per_minute": 10000,
  "max_metric_definitions": 500,
  "max_slos": 200
}
```

Check current usage:

```http
GET /api/obs/mtx/quota/usage
```

---

## Prometheus Integration

Configure and expose a Prometheus scrape endpoint:

```http
POST /api/obs/mtx/prometheus/config
{
  "endpoint": "/metrics",
  "port": 9090,
  "scrape_interval_seconds": 15,
  "include_namespaces": ["apg"]
}
```

Scrape:
```
GET /api/obs/mtx/prometheus/metrics
Content-Type: text/plain; version=0.0.4
```

Histogram metrics are emitted with `le` bucket labels, `_sum`, and `_count` suffixes — compatible with Prometheus `histogram_quantile()`.

---

## NATS JetStream Ingestion (Streaming)

For high-throughput ingestion, publish metric CloudEvents to NATS JetStream and let the service consume them asynchronously via `start_nats_ingestion()` (requires `nats-py`):

```python
await svc.start_nats_ingestion(
    nats_url="nats://localhost:4222",
    subject="obs.metrics.ingest",
)
```

This decouples ingestion from evaluation, enables horizontal fanout, and provides durable replay for late-arriving telemetry — using NATS JetStream as the APG-standard streaming transport.

---

## Audit Log

All mutating operations emit audit events:

```http
GET /api/obs/mtx/audit?event_type=slo_created&limit=50
```

Event types include: `metric_definition_created`, `slo_created`, `slo_updated`, `burn_rate_alert_fired`, `burn_rate_alert_fired_multiwindow`, `composite_slo_created`, `error_budget_policy_evaluated`, `dashboard_grafana_exported`, `tenant_quota_set`.
