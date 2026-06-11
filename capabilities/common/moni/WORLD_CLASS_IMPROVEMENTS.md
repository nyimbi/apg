# MONI - World Class Improvements

## 15 High-Impact Improvements for APG Monitoring & Observability

---

### 1. Distributed Tracing Context Propagation

The current trace ingestion only stores a `trace_id` string on `SignalRecord` with no propagation context. A world-class implementation propagates W3C `traceparent`/`tracestate` headers across tenant boundaries, builds a directed acyclic graph of spans, and surfaces flame-chart latency breakdowns per tenant service. This closes the gap between MONI being a governance record store and being a first-class distributed tracing backend.

**Impact**: Critical path visibility for multi-service SLO burn investigations.

---

### 2. Adaptive Anomaly Detection with Concept Drift Handling

`_update_metric_baseline` accumulates a rolling deque of 1000 values and recomputes mean/std, but it does not handle concept drift — sustained regime changes that make historical baselines stale. A world-class system uses exponentially weighted moving statistics (EWMA) with a configurable half-life, detects drift using the CUSUM algorithm, and auto-resets baselines when drift is confirmed rather than silently absorbing the shift.

**Impact**: Eliminates chronic alert fatigue caused by stale baselines after deployments or traffic shifts.

---

### 3. SLO Burn Rate Alerting with Error Budget Tracking

`SloRecord` stores a `threshold` and a `window_minutes` but has no burn-rate logic. Google SRE's error budget model requires alerting when the burn rate exceeds a multiple of the budget consumption rate that would exhaust the monthly budget if sustained. MONI should compute `error_budget_remaining_percent`, `burn_rate_1h`, `burn_rate_6h`, and emit `slo.burn_rate_critical` events when the fast-window and slow-window burn rates both breach their thresholds.

**Impact**: Actionable SLO alerting that is neither too early nor too late, matching production SRE workflows.

---

### 4. On-Call Schedule Integration and Escalation Routing

`AlertRecord` has a `notification_route` string but no on-call schedule model. A world-class system manages rotation schedules (primary, secondary, manager), computes who is currently on-call at alert-creation time, and escalates automatically after configurable acknowledgement windows. The schedule should be tenant-scoped, support override windows, and produce an audit event for every routing decision.

**Impact**: Eliminates missed alerts due to stale routing configuration and manual on-call handoffs.

---

### 5. Cardinality Budget Enforcement per Tenant

`ingest_signal` accepts a `cardinality` integer but only checks that it is non-negative. High-cardinality metrics are the primary cause of observability system overload. MONI should maintain a per-tenant cardinality budget (`max_series_per_tenant`), count active series per metric name, reject ingestion that would breach the budget with a `cardinality_budget_exceeded` policy decision, and surface the budget utilization in `dashboard_summary`.

**Impact**: Prevents runaway cardinality from degrading the monitoring plane for other tenants.

---

### 6. Metric Rollup and Downsampling Pipeline

`_metrics_store` uses an in-memory `deque(maxlen=100000)` with no downsampling. After the raw retention window expires, high-resolution data is simply discarded. A world-class implementation produces rollup tiers (1m → 5m → 1h → 1d) using configurable aggregation functions (min, max, sum, count, p50, p95, p99). Rollup tasks run in the background processing loop, and `query_metrics` selects the appropriate resolution tier based on the requested time range.

**Impact**: Extends effective retention by 100x without proportional storage growth.

---

### 7. Multi-Dimensional Metric Correlation Engine

`_update_correlation_graph` builds correlations by temporal proximity over the last 10 metric keys — a near-trivial heuristic. A world-class correlation engine uses Pearson and Spearman cross-correlation across configurable lag windows, segments correlation by tenant service topology, and surfaces the top correlated metrics at incident creation time to accelerate root-cause analysis.

**Impact**: Cuts mean time to diagnosis by surfacing relevant co-varying signals automatically.

---

### 8. Structured Runbook Execution with Step Tracking

`RemediationRequestRecord` tracks approval state but has no model for the runbook steps themselves. A world-class runbook engine stores ordered steps with preconditions, postcondition checks, rollback steps, and estimated duration. Execution emits a `RunbookStepRecord` per step with stdout/stderr capture and outcome. The approval workflow gates execution at step boundaries for production environments, not just at request creation.

**Impact**: Provides complete remediation audit trails and enables safe semi-automated incident response.

---

### 9. Real-Time WebSocket Push for Dashboard Updates

`get_dashboard_data` polls cached data on demand. Every dashboard refresh is a separate query round-trip. A world-class implementation maintains a lightweight publish-subscribe channel (asyncio `Queue` per connected dashboard session) that pushes delta updates when metrics or alert states change. This eliminates polling overhead and delivers sub-second dashboard freshness without long-polling.

**Impact**: Dramatically improves operator situational awareness during active incidents.

---

### 10. Composite Health Scoring with Weighted Signals

`_calculate_tenant_health_score` uses simple subtraction based on alert counts. It does not weight signals by SLO criticality, service tier, or historical reliability. A world-class health score uses a weighted sum over error budget consumption, P99 latency percentiles, saturation metrics, and dependency health — following the four golden signals model (latency, traffic, errors, saturation). Weights are configurable per tenant service tier.

**Impact**: Makes health scores actionable and comparable across services with different traffic patterns.

---

### 11. Metric Pipeline Backpressure and Rejection Feedback

`track_metric` silently drops metrics when exceptions occur and always returns `False` on failure with no feedback about why. A world-class pipeline implements explicit backpressure: when the in-memory queue exceeds a high-water mark, new metrics are rejected with a `MONI_BACKPRESSURE` status code that the producer must handle. Rejection counters are surfaced as first-class metrics so operator dashboards show ingestion pressure in real time.

**Impact**: Prevents silent data loss during ingestion spikes and makes capacity planning observable.

---

### 12. Per-Tenant Retention Policy Enforcement

`MonitoringServiceConfig` has global `metric_retention_hours` and `alert_retention_hours` but no per-tenant overrides. A world-class system enforces a `DataRetentionPolicy` per tenant (already modeled in `models.py` but unused in the control plane). Cleanup tasks in `_cleanup_old_metrics` should respect per-tenant TTLs, emit `data.purged` audit events, and expose retention budget utilization in `dashboard_summary`.

**Impact**: Enables compliance-driven data governance without reimplementing retention logic per customer.

---

### 13. Chaos Injection Hooks for Resilience Testing

The service has no mechanism for synthetic fault injection. A world-class observability system includes a `chaos` mode that can inject artificial latency into metric ingestion, simulate alert delivery failures, and introduce artificial anomaly signals. This is essential for testing alert routing, escalation logic, and on-call workflows without waiting for production incidents. Chaos hooks must be guarded by a `chaos_enabled` config flag defaulting to `False`.

**Impact**: Enables continuous resilience validation of the monitoring plane itself.

---

### 14. OpenTelemetry Semantic Convention Validation

`ingest_signal` accepts arbitrary `name` strings with no naming convention enforcement. OTel semantic conventions define canonical names like `http.server.request.duration`, `db.client.operation.duration`, and `system.cpu.utilization`. A world-class MONI validates signal names against a configurable convention registry, emits `signal.convention_violation` audit events for non-conforming names, and provides a migration helper that suggests canonical alternatives.

**Impact**: Prevents signal naming entropy that makes cross-service correlation and dashboard reuse impossible.

---

### 15. AI-Assisted Alert Triage with Explanation Generation

`create_alert` creates an `AlertRecord` but provides no diagnostic context beyond the triggering signal. A world-class alert triage engine queries correlated metrics, recent deployment events, and historical resolution patterns at alert creation time, then appends a structured `triage_context` block to the alert record. When an Ollama-hosted model is available (honoring the APG local-AI strategy), it generates a natural-language explanation of likely root causes ranked by confidence.

**Impact**: Reduces mean time to acknowledge from minutes to seconds by eliminating the initial diagnostic search phase.
