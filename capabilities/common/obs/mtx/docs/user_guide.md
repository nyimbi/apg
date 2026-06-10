# obs_mtx User Guide — Metrics & SLO

## Overview

`obs_mtx` provides RED metrics collection (Rate, Error rate, Duration), SLO (Service Level Objective) management, burn rate alerting, Prometheus text exposition, and auto-generated dashboards.

## Core Concepts

- **Metric Definition**: declares a named metric with type (counter/gauge/histogram/summary), unit, and label schema.
- **Data Point**: a recorded value for a named metric at a given timestamp with labels.
- **RED Metrics**: aggregate view — request rate, error rate, and duration percentiles for a service.
- **SLO**: a target compliance level (e.g., 99.9% availability over 30 days) with an error budget.
- **Burn Rate Alert**: fires when the error budget is being consumed faster than it can recover.
- **Dashboard**: a collection of panels with PromQL-style queries.

## Use Cases

1. **Service health monitoring**: record `_requests_total`, `_errors_total`, `_duration_ms` counters and use the RED endpoint for live summaries.
2. **SLO tracking**: define a 99.9% availability SLO, run `/slos/<id>/evaluate` to check compliance, see remaining error budget.
3. **Burn rate alerting**: a burn rate of 14.4× depletes the monthly error budget in 2 hours — configure alerts to catch this before the window closes.
4. **Prometheus integration**: expose `/prometheus/metrics` as a scrape endpoint for Prometheus or Grafana Agent.
5. **Auto dashboards**: POST to `/dashboards/generate/red/<service>` to create a Grafana-compatible RED dashboard.

## Quick Start

### 1. Define a metric

```http
POST /api/obs/mtx/metrics
X-Tenant-ID: my-org
Content-Type: application/json

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

### 3. Define an SLO

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

### 4. Create a burn rate alert

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

### 5. Prometheus scrape endpoint

```
GET /api/obs/mtx/prometheus/metrics
```

Returns metrics in Prometheus text format (content-type `text/plain; version=0.0.4`).

## SLO Types

| Type | Measures |
|------|----------|
| `availability` | Fraction of time service is up (`_up` metric = 1) |
| `error_rate` | 1 - (errors / requests) |
| `latency` | Fraction of requests below `latency_threshold_ms` |
| `throughput` | Request rate relative to target |

## Burn Rate Interpretation

A burn rate of 1.0 means the budget is being consumed at exactly the sustainable rate. 14.4× over 1 hour exhausts the monthly budget in 2 hours. Google SRE recommends multi-window (1h/6h) alerting to avoid false positives.
