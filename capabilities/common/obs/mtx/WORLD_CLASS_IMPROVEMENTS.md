# obs_mtx — World-Class Improvements

15 targeted enhancements to elevate this capability from solid baseline to production-grade, Google-SRE-level observability.

---

### I1. Multi-Window Burn Rate Alerting (Google SRE Chapter 5 Compliant)

**Category**: Alerting | **Justification**: Single-window burn rate has a 14% false-positive rate at low traffic. Dual-window (fast + slow) alerting reduces noise by requiring both windows to fire simultaneously, matching the Google SRE Workbook recommendation that cuts page fatigue by ~60%. | **Implementation**: Add `evaluate_burn_rate_multiwindow()` that computes independent short/long burn rates and only fires when *both* exceed their respective thresholds (e.g., 14.4× over 1h AND 6× over 6h). Store per-window states independently. | **Competitor**: Google Cloud Monitoring multi-condition burn rate policies; Sloth SLO framework.

---

### I2. OpenMetrics 1.0 + Exemplar Support

**Category**: Export | **Justification**: Prometheus text format v0.0.4 lacks trace context linkage. OpenMetrics 1.0 + exemplars allows Grafana Tempo/Jaeger to jump directly from a spike on a graph to the causative trace, cutting MTTR by 40-70% in practice. | **Implementation**: Add `render_openmetrics()` that emits `# EOF`, exemplar `{ ... } value timestamp traceID=<id>` lines, and `_created` suffixes for counters. Accept `exemplar` dict in `record_metric()`. | **Competitor**: Prometheus native histograms + exemplar support; DataDog APM.

---

### I3. Error Budget Policy Automation

**Category**: SLO | **Justification**: Static SLO targets don't adapt to release cadence. Error budget policies define *automatic* actions when budget drops below thresholds (freeze deployments, throttle feature flags), turning SLOs from dashboards into control loops. | **Implementation**: Add `create_error_budget_policy()` with configurable thresholds and action callbacks; `evaluate_error_budget_policy()` fires registered async callables when budget crosses a tier. | **Competitor**: Nobl9 error budget policies; Chronosphere SLO automation.

---

### I4. Histogram Bucket Aggregation with Native Percentiles

**Category**: Metrics | **Justification**: Storing raw duration data points and computing percentiles ad-hoc via `_percentile()` is O(n log n) per query and loses bucket resolution. Prometheus-compatible histogram buckets enable `histogram_quantile()` without materialising all samples. | **Implementation**: Add `record_histogram_observation()` that buckets values into configurable bucket boundaries (e.g., `[5, 10, 25, 50, 100, 250, 500, 1000, 2500]` ms) and maintains `_bucket`, `_sum`, `_count` counters in-memory. `render_prometheus_metrics()` emits proper `le` labels. | **Competitor**: Prometheus native histograms; VictoriaMetrics; Grafana Mimir.

---

### I5. NATS JetStream Metric Ingestion Pipeline

**Category**: Streaming | **Justification**: The current in-process `record_metric()` creates backpressure when >100k points flood in. A NATS JetStream consumer decouples ingestion from evaluation, enables horizontal fanout, and provides durable replay for late-arriving telemetry — matching bytewax+NATS as the APG streaming platform. | **Implementation**: Add `start_nats_ingestion(nats_url, subject)` that subscribes to a JetStream subject, deserialises CloudEvents payloads, and pipes to `bulk_record_metrics()`. Uses `nats-py` async client with backpressure-aware batch flushing. | **Competitor**: (reference only) Apache Kafka Streams; APG recommended: NATS JetStream + bytewax.

---

### I6. Adaptive Anomaly Detection on RED Metrics

**Category**: Intelligence | **Justification**: Static burn rate thresholds miss gradual degradation (e.g., p99 drifting from 200ms to 800ms over 6 hours). EWMA-based anomaly detection on RED metrics catches these trends 3-5× earlier than threshold alerting. | **Implementation**: Add `compute_ewma_anomaly()` that maintains exponential weighted moving averages (α=0.1) for rate/error/duration per service. Returns z-score and `is_anomalous` flag when deviation exceeds 3σ. State persists in `_ewma_state` dict. | **Competitor**: AWS CloudWatch Anomaly Detection; Dynatrace Davis AI; Grafana Machine Learning.

---

### I7. SLO Forecasting with Error Budget Depletion ETA

**Category**: SLO | **Justification**: Current compliance is backward-looking. Engineering managers need to answer "will we breach SLO before the window closes?" Linear regression on error budget consumption rate gives a depletion ETA with confidence intervals, enabling proactive intervention. | **Implementation**: Add `forecast_slo_compliance(slo_id, lookahead_hours)` using `statistics.linear_regression()` (Python 3.11+) on time-series compliance snapshots. Returns `predicted_compliance_at_window_end`, `budget_depletion_eta`, `confidence`. | **Competitor**: Nobl9 SLO forecasting; Lightstep (ServiceNow) predictive SLOs.

---

### I8. Metric Cardinality Guard

**Category**: Reliability | **Justification**: Label cardinality explosions (e.g., `user_id` as a label) are the #1 cause of Prometheus OOM. A cardinality guard that warns/rejects high-cardinality label combinations before they're recorded prevents production incidents. | **Implementation**: Add `check_metric_cardinality(metric_name)` that counts distinct label value combinations. Enforce a configurable `max_cardinality` (default 10k) per metric at `record_metric()` time, raising `CardinalityLimitError` with offending label suggestions. | **Competitor**: Grafana Mimir per-tenant cardinality limits; VictoriaMetrics `-storage.maxUniqueTimeseries`.

---

### I9. SLO Composite (Dependency-Aware) Views

**Category**: SLO | **Justification**: A user-facing SLO (e.g., checkout flow) depends on multiple downstream SLOs (payment, inventory, auth). A composite SLO aggregates child SLO compliance using weakest-link semantics, giving a single reliability number that reflects real user experience. | **Implementation**: Add `create_composite_slo(name, child_slo_ids, aggregation)` where `aggregation` is `min` (weakest link), `product` (independent failures), or `weighted_average`. `evaluate_composite_slo()` recursively evaluates children and applies the formula. | **Competitor**: Grafana SLO composite; Dynatrace business SLOs; Nobl9 composite objectives.

---

### I10. Grafana Dashboard JSON Export

**Category**: Export | **Justification**: The current dashboard model is an internal representation. Exporting to native Grafana JSON means zero-friction import into existing Grafana instances, eliminating manual panel recreation and configuration drift between code and UI. | **Implementation**: Add `export_grafana_dashboard(dash_id)` that transforms APG panel definitions into Grafana dashboard JSON schema v36+ with correct `gridPos`, `targets`, `fieldConfig`, `thresholds`, and `datasource` references. | **Competitor**: Grafana as-code (Grafonnet/Jsonnet); Perses dashboard-as-code.

---

### I11. PromQL Query Validation

**Category**: Developer Experience | **Justification**: Invalid PromQL silently produces empty panels, wasting debugging time. Inline PromQL syntax validation at dashboard creation time surfaces errors immediately, matching the developer experience of Grafana's built-in query inspector. | **Implementation**: Add `validate_promql(query)` using a lightweight regex + AST parser (or remote call to Prometheus `/api/v1/query` dry-run endpoint). Integrate into `create_dashboard()` to warn on malformed queries without blocking creation. | **Competitor**: Prometheus API `/api/v1/query?time=0&query=...`; Grafana query inspector.

---

### I12. Time-Window Downsampling for Long-Retention Queries

**Category**: Performance | **Justification**: Querying 100k raw data points for a 7-day trend is O(n) per query. Automatic downsampling into 1m/5m/1h resolution tiers (inspired by Prometheus recording rules) reduces query latency from O(n) to O(resolution_steps), typically 100-1000× faster for wide time ranges. | **Implementation**: Add `compute_downsampled_series(metric_name, service_name, resolution_minutes, start_time, end_time)` that groups raw points into time buckets and emits `{min, max, avg, count, p50, p99}` per bucket. Cache results in `_downsample_cache` with TTL. | **Competitor**: Prometheus recording rules; Thanos downsampling; VictoriaMetrics rollup functions.

---

### I13. OTLP (OpenTelemetry) Ingest Endpoint

**Category**: Interoperability | **Justification**: 80%+ of modern instrumentation libraries (OpenTelemetry SDKs) emit OTLP protobuf. Accepting OTLP metrics directly eliminates the need for a separate OTel Collector sidecar, reducing operational complexity and latency by one network hop. | **Implementation**: Add `ingest_otlp_metrics(otlp_payload)` that parses OTLP `ExportMetricsServiceRequest` JSON (or proto-JSON) and maps `ResourceMetrics → ScopeMetrics → Metric` onto APG metric definitions and data points. | **Competitor**: Grafana Alloy OTLP receiver; VictoriaMetrics OTLP endpoint; DataDog OTLP ingest.

---

### I14. SLO Change Impact Analysis

**Category**: Engineering | **Justification**: Changing an SLO target from 99.9% to 99.95% doubles the allowed downtime from 43.8 min/month to 21.9 min/month. An impact analysis that shows historical compliance against the *proposed* target helps teams negotiate realistic targets backed by data. | **Implementation**: Add `analyze_slo_target_change(slo_id, proposed_target)` that replays historical data points against the new target, computes what compliance would have been, and returns `would_have_breached_n_times`, `historical_min_compliance`, `error_budget_delta_minutes`. | **Competitor**: Nobl9 SLO simulation; Blameless SLO negotiation tools.

---

### I15. Tenant-Scoped Metric Quotas and Rate Limiting

**Category**: Multi-Tenancy | **Justification**: In a shared APG deployment, a single misconfigured service can generate millions of data points per minute, starving other tenants. Per-tenant metric ingestion quotas with automatic throttling and quota usage reporting are required for SLA-backed multi-tenant operation. | **Implementation**: Add `set_tenant_quota(max_points_per_minute, max_metric_definitions, max_slos)`. Enforce at `record_metric()` and `create_metric_definition()` using a token bucket per tenant stored in `_quota_state`. Expose `get_quota_usage()` endpoint. | **Competitor**: Grafana Mimir per-tenant limits; Thanos per-tenant object store limits; VictoriaMetrics per-tenant rate limiting.
