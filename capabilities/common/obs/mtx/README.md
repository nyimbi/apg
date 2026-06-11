# obs_mtx — Metrics & SLO

RED metrics (Rate/Error/Duration), SLO definition, burn rate alerts, Prometheus export, dashboard generation, EWMA anomaly detection, SLO forecasting, composite SLOs, histogram buckets, cardinality guards, downsampling, error budget policies, Grafana JSON export.

**Capability ID:** `obs_mtx` | **Domain:** observability | **Version:** 2.0.0

## Core Features

| Feature | Description |
|---------|-------------|
| RED Metrics | Rate / Error rate / Duration percentiles per service |
| SLO Management | Availability, latency, error rate, throughput targets with error budget |
| Multi-Window Burn Rate | Google SRE dual-window alerting (fast + slow burn) |
| Histogram Buckets | Prometheus-compatible bucket aggregation with `histogram_quantile` |
| EWMA Anomaly Detection | Exponential weighted moving average anomaly scoring on RED metrics |
| SLO Forecasting | Linear regression forecast of compliance + budget depletion ETA |
| Composite SLOs | Aggregate child SLOs with min / product / weighted_average semantics |
| Cardinality Guard | Detect and block high-cardinality label explosions |
| Downsampling | Time-bucketed min/max/avg/p50/p99 with TTL cache |
| Error Budget Policies | Automated actions (freeze deploys, throttle flags) on budget thresholds |
| Grafana JSON Export | Native Grafana dashboard JSON (schema v36+) |
| SLO Impact Analysis | Simulate effect of target change on historical compliance |
| Tenant Quotas | Per-tenant ingestion rate limits and definition caps |
| Prometheus Export | Full text exposition format v0.0.4 |

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/mtx/health` | Health check |
| GET | `/api/obs/mtx/describe` | Capability descriptor |
| GET | `/api/obs/mtx/audit` | Audit event log |

### Metric Definitions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/mtx/metrics` | List metric definitions |
| POST | `/api/obs/mtx/metrics` | Create metric definition |
| GET | `/api/obs/mtx/metrics/<id>` | Get metric definition |
| PUT | `/api/obs/mtx/metrics/<id>` | Update metric definition |
| DELETE | `/api/obs/mtx/metrics/<id>` | Delete metric definition |

### Data Ingestion

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/obs/mtx/data-points` | Record a metric data point |
| POST | `/api/obs/mtx/data-points/bulk` | Bulk record data points |
| GET | `/api/obs/mtx/data-points/query` | Query data points |
| POST | `/api/obs/mtx/data-points/histogram` | Record histogram observation |
| GET | `/api/obs/mtx/data-points/histogram/<name>/<service>/quantile` | Compute quantile from buckets |

### RED Metrics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/mtx/red/<service_name>` | RED metrics for one service |
| GET | `/api/obs/mtx/red` | RED metrics for all services |
| GET | `/api/obs/mtx/red/<service_name>/anomaly` | EWMA anomaly detection |
| GET | `/api/obs/mtx/red/<service_name>/downsample` | Downsampled time series |

### SLOs

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/mtx/slos` | List SLOs |
| POST | `/api/obs/mtx/slos` | Create SLO |
| GET | `/api/obs/mtx/slos/<id>` | Get SLO |
| PUT | `/api/obs/mtx/slos/<id>` | Update SLO |
| DELETE | `/api/obs/mtx/slos/<id>` | Delete SLO |
| GET | `/api/obs/mtx/slos/<id>/evaluate` | Evaluate SLO compliance |
| GET | `/api/obs/mtx/slos/evaluate-all` | Evaluate all SLOs |
| GET | `/api/obs/mtx/slos/<id>/forecast` | Forecast compliance + depletion ETA |
| POST | `/api/obs/mtx/slos/<id>/analyze-target-change` | Simulate target change impact |

### Composite SLOs

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/obs/mtx/composite-slos` | Create composite SLO |
| GET | `/api/obs/mtx/composite-slos/<id>/evaluate` | Evaluate composite SLO |

### Burn Rate Alerts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/mtx/burn-rate-alerts` | List burn rate alerts |
| POST | `/api/obs/mtx/burn-rate-alerts` | Create burn rate alert |
| GET | `/api/obs/mtx/burn-rate-alerts/<id>` | Get alert |
| PUT | `/api/obs/mtx/burn-rate-alerts/<id>` | Update alert |
| DELETE | `/api/obs/mtx/burn-rate-alerts/<id>` | Delete alert |
| GET | `/api/obs/mtx/burn-rate-alerts/<id>/evaluate` | Single-window burn rate |
| GET | `/api/obs/mtx/burn-rate-alerts/<id>/evaluate-multiwindow` | Dual-window burn rate (Google SRE) |

### Error Budget Policies

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/obs/mtx/error-budget-policies` | Create error budget policy |
| GET | `/api/obs/mtx/error-budget-policies/<id>/evaluate` | Evaluate policy + trigger actions |

### Prometheus & Export

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/obs/mtx/prometheus/config` | Configure Prometheus export |
| GET | `/api/obs/mtx/prometheus/config` | Get Prometheus config |
| GET | `/api/obs/mtx/prometheus/metrics` | Prometheus text exposition |

### Dashboards

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/mtx/dashboards` | List dashboards |
| POST | `/api/obs/mtx/dashboards` | Create dashboard |
| GET | `/api/obs/mtx/dashboards/<id>` | Get dashboard |
| PUT | `/api/obs/mtx/dashboards/<id>` | Update dashboard |
| DELETE | `/api/obs/mtx/dashboards/<id>` | Delete dashboard |
| POST | `/api/obs/mtx/dashboards/generate/red/<service>` | Auto-generate RED dashboard |
| GET | `/api/obs/mtx/dashboards/<id>/export/grafana` | Export as Grafana JSON |

### Cardinality & Quotas

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/mtx/cardinality/<metric_name>` | Cardinality check for a metric |
| POST | `/api/obs/mtx/quota` | Set tenant quota |
| GET | `/api/obs/mtx/quota/usage` | Get quota usage |

## Headers

Pass `X-Tenant-ID: <tenant>` on every request for multi-tenant isolation.

## New Methods (v2.0.0)

| Method | Description |
|--------|-------------|
| `evaluate_burn_rate_multiwindow()` | Dual-window burn rate (Google SRE Chapter 5) |
| `record_histogram_observation()` | Prometheus-compatible histogram bucket ingestion |
| `get_histogram_quantile()` | Quantile estimation via bucket interpolation |
| `compute_ewma_anomaly()` | EWMA anomaly detection on RED metrics |
| `forecast_slo_compliance()` | Linear regression SLO compliance forecast |
| `create_composite_slo()` | Aggregate multiple SLOs (min/product/weighted_average) |
| `evaluate_composite_slo()` | Evaluate composite SLO compliance |
| `check_metric_cardinality()` | Detect high-cardinality label explosions |
| `compute_downsampled_series()` | Bucketed time series with TTL cache |
| `create_error_budget_policy()` | Automated actions on budget threshold breach |
| `evaluate_error_budget_policy()` | Evaluate policy + fire async action callbacks |
| `export_grafana_dashboard()` | Export APG dashboard as Grafana JSON v36+ |
| `analyze_slo_target_change()` | Simulate target change on historical data |
| `set_tenant_quota()` | Configure per-tenant ingestion limits |
| `get_quota_usage()` | Report current quota consumption |
