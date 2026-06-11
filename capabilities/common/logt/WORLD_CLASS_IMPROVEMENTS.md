# LOGT — World-Class Improvement Proposals

### I1. Structured Correlation ID Propagation via Context Vars
**Category**: Observability Architecture
**Justification**: Every log/trace emitted inside a request context should carry the same correlation ID without explicit threading. Python's `contextvars.ContextVar` allows zero-overhead, async-safe propagation that survives `asyncio.create_task` — the pattern used by Datadog's ddtrace and OpenTelemetry's SDK. Without it, distributed traces fragment across log lines and correlation requires expensive post-hoc joins.
**Implementation**: Introduce `CorrelationContext` (a `ContextVar[dict]`) holding `trace_id`, `span_id`, `tenant_id`, `request_id`. `ingest_log` reads from it when caller omits explicit ids. Expose `logt_context()` async context manager to set/clear on entry.
**Competitor**: OpenTelemetry Python SDK (`opentelemetry-sdk`), Datadog ddtrace context propagators.

---

### I2. Adaptive Head-Based + Tail-Based Sampling Decision Engine
**Category**: Sampling / Cost Control
**Justification**: Head-based sampling (decide at trace root) blindly drops slow/error traces. Tail-based sampling (decide after all spans arrive) retains exactly the interesting ones. Jaeger, Honeycomb, and Google Cloud Trace offer tail sampling; most OSS stacks don't. At scale, adaptive rate decisions cut storage 10x while keeping 100 % of anomalous traces.
**Implementation**: Add `SamplingEngine` to `observability_runtime.py` with `should_sample(trace_id, root_service, policy)` returning a `SamplingDecision(keep, reason, rate)`. Support policy types: `head_fixed_rate`, `tail_error`, `tail_latency_p99`, `adaptive_token_bucket`. Store decisions in `_sampling_decisions` dict keyed by `trace_id`.
**Competitor**: Honeycomb Refinery, Jaeger adaptive sampling, Google Cloud Trace adaptive sampling.

---

### I3. Semantic Log Level Budget with Token-Bucket Rate Limiting
**Category**: Performance / Cardinality Control
**Justification**: Uncontrolled debug/info log bursts during incidents saturate pipelines, masking signal with noise. Datadog Agent and Vector both ship per-service log rate limiters. A token-bucket per `(tenant_id, service_name, severity)` triplet gives smooth throughput contracts without blanket sampling.
**Implementation**: Add `LogBudgetLimiter` with `Decimal`-precise token counts per bucket. `ingest_log` checks bucket before accepting — returns `{"decision": "rate_limited", "retry_after_ms": N}` on exhaustion. Admin API to inspect and reset buckets. Audit event on every rate-limit hit.
**Competitor**: Vector `throttle` transform, Fluent Bit rate-limit filter, Datadog per-service ingestion controls.

---

### I4. Immutable Append-Only Log Segments with Merkle-Root Integrity Proof
**Category**: Compliance / Tamper Evidence
**Justification**: GDPR Art. 5(2) and SOC2 CC7.2 require demonstrable log integrity. Hash-chained log segments (like Kafka segment files or certificate transparency logs) prove no log was silently deleted. Splunk's immutable archive and AWS CloudTrail log file validation use this pattern.
**Implementation**: Group `LogEvent` records into 100-event segments. Each segment carries a `merkle_root` (SHA-256 of sorted log IDs + hashes). `verify_log_integrity(tenant_id, segment_id)` recomputes root and returns `{"valid": bool, "segment_id": ..., "root": ...}`. Store segments in `_log_segments` dict.
**Competitor**: AWS CloudTrail log file validation, Splunk immutable archive, Hyperledger Fabric transaction log.

---

### I5. OpenTelemetry OTLP Ingest Adapter with W3C TraceContext Parsing
**Category**: Interoperability / Standards
**Justification**: Every modern observability tool (Grafana, Tempo, Jaeger, Honeycomb) emits OTLP. A native OTLP ingest path means LOGT becomes a drop-in backend for any instrumented service without custom SDKs. W3C TraceContext (`traceparent` / `tracestate`) is the IETF standard for distributed context propagation.
**Implementation**: Add `async ingest_otlp_batch(tenant_id, otlp_payload_json, pipeline_id, actor)` that parses OTLP `ResourceSpans`/`ScopeLogs` JSON, extracts W3C `traceparent`, maps to internal `SpanRecord`/`LogEvent`, and bulk-ingests. Return `{"accepted": N, "rejected": M, "errors": [...]}`.
**Competitor**: Grafana Tempo OTLP receiver, OpenTelemetry Collector, Jaeger OTLP endpoint.

---

### I6. Real-Time Anomaly Detection with Exponential Weighted Moving Average
**Category**: Intelligence / Incident Detection
**Justification**: Static error-rate thresholds (current `log_anomaly_detect`) fire too late and generate false positives during deploy ramps. EWMA (exponential weighted moving average) models baseline seasonality, firing only when deviation exceeds N standard deviations. Datadog APM anomaly detection and AWS DevOps Guru use EWMA variants.
**Implementation**: Add `EWMADetector` per `(tenant_id, service_name)` in `observability_runtime.py`. `update(value)` updates μ and σ². `is_anomaly(value, threshold_sigma=3.0)` returns bool. `log_anomaly_detect_ewma` service method exposes this with per-service state stored in `_ewma_state` dict.
**Competitor**: Datadog APM anomaly detection, Prometheus `predict_linear`, AWS DevOps Guru.

---

### I7. Distributed Trace Waterfall Critical-Path Analysis
**Category**: Performance Engineering
**Justification**: Knowing a trace took 2 s is useless without knowing which span is on the critical path. Critical-path analysis (finding the longest dependency chain through span parent-child relationships) is the core of Jaeger's trace comparison view, Honeycomb's `HEATMAP`, and Google Dapper. It reduces MTTR from hours to minutes.
**Implementation**: Add `async trace_critical_path(tenant_id, trace_id)` that builds a DAG from `parent_span_id` links, runs topological sort + longest-path DP, returns `{"critical_path": [span_ids], "total_critical_ms": float, "bottleneck_span": span_dict}`.
**Competitor**: Jaeger UI critical-path overlay, Honeycomb trace waterfall, Google Dapper.

---

### I8. Multi-Tenant Log Cardinality Budget with Decimal Accounting
**Category**: Multi-Tenancy / Cost Fairness
**Justification**: High-cardinality tenants (many unique attribute key-value pairs) cause exponential index bloat in Elasticsearch/Loki, degrading all tenants. Enforcing a per-tenant cardinality budget — tracked with `Decimal` for precision across billing periods — enables fair cost allocation and SLA guarantees. Datadog's custom metric limits and Grafana Cloud's cardinality enforcement use this model.
**Implementation**: Add `async cardinality_budget_check(tenant_id, attribute_keys)` tracking unique attribute key counts per tenant using `Decimal`. `async cardinality_usage_report(tenant_id)` returns `{"budget": Decimal, "used": Decimal, "pct_used": Decimal, "top_keys": [...]}`.
**Competitor**: Datadog custom metric limits, Grafana Cloud cardinality enforcement, Honeycomb column limits.

---

### I9. Log Parsing Pipeline with Named-Pattern Registry
**Category**: Ingestion Quality
**Justification**: Raw logs from nginx, postgres, Java stack traces, and syslog need different parsers. Maintaining a registry of named patterns (grok-style) allows zero-code ingestion of new log formats. Fluentd/Fluent Bit, Logstash, and Vector all expose named-pattern registries as their primary extension mechanism.
**Implementation**: Add `_parsers: dict[str, Callable[[str], dict]]` registry. `async register_log_parser(tenant_id, name, pattern_type, pattern, actor)` installs parser. `async parse_with_named_pattern(tenant_id, raw_text, parser_name, ...)` dispatches to registered parser then calls `ingest_log`. Support pattern types: `kv`, `json`, `regex`, `csv`.
**Competitor**: Logstash grok patterns, Vector `remap` transforms, Fluent Bit parsers.

---

### I10. Span-Level Cost Attribution with Decimal Money Tracking
**Category**: FinOps / Observability Economics
**Justification**: Platform teams need to attribute infrastructure cost to individual services, operations, and tenants. Tagging spans with `cost_usd: Decimal` (computed from duration × resource rate) enables per-operation cost visibility — the model used by AWS X-Ray cost analysis and Datadog Cost Management.
**Implementation**: Add `CostAttribution` model with `span_id`, `cost_usd: Decimal`, `resource_class`, `rate_per_ms`. `async attribute_span_cost(tenant_id, span_id, resource_class, rate_per_ms_decimal)` computes and stores attribution. `async cost_report(tenant_id, group_by)` aggregates with `Decimal` sum.
**Competitor**: AWS X-Ray cost analysis, Datadog Cost Management, Honeycomb usage-based pricing introspection.

---

### I11. Log Sampling with Consistent Hashing (Deterministic Replay)
**Category**: Sampling / Reproducibility
**Justification**: Random sampling breaks reproducibility — replaying events yields different samples. Consistent-hash sampling (hash `log_id % N == 0`) guarantees the same log is always sampled or always dropped across replays, restoring incident reproducibility. This is the approach used by Datadog's probabilistic sampler and OpenTelemetry's `ParentBasedSampler`.
**Implementation**: Add `ConsistentHashSampler` to `observability_runtime.py`. `should_keep(log_id, rate)` returns `bool` via `int(sha256(log_id)[:8], 16) % 10000 < rate * 10000`. `ingest_log` passes through sampler before storage when pipeline sampling_policy is `consistent_hash`.
**Competitor**: Datadog probabilistic sampler, OpenTelemetry `TraceIdRatioBased` sampler, Jaeger remote sampling.

---

### I12. Tenant-Scoped Log Masking Rules with Regex Compilation Cache
**Category**: Privacy / PII Protection
**Justification**: GDPR and HIPAA require PII (emails, phone numbers, NINs) to never appear in plaintext logs. Static `[redacted]` replacement (current approach) requires manual field enumeration. A regex-based masking rule engine with pre-compiled patterns reduces PII exposure to zero without requiring callers to enumerate fields. AWS Macie and Google DLP use this model.
**Implementation**: Add `MaskingRuleSet` with compiled regex cache (`_compiled: dict[str, re.Pattern]`). `async register_masking_rule(tenant_id, name, pattern, replacement, actor)` stores rule. `async mask_log_message(tenant_id, message)` applies all rules. `redact_message` in runtime delegates to active ruleset.
**Competitor**: Google DLP, AWS Macie, Vector `redact` transform, Presidio (Microsoft).

---

### I13. Structured Audit Query Language (mini-AQL) with Predicate Pushdown
**Category**: Diagnostic / Compliance
**Justification**: `search_logs` supports only substring matching. Production on-call engineers need expressions like `severity=error AND service=payments AND NOT redaction_applied`. A minimal predicate DSL (parsed to an AST, evaluated against log dicts) gives self-service incident investigation without full-text indexing infrastructure. Splunk SPL, Grafana LogQL, and Loki are all built on this model.
**Implementation**: Add `LogQueryParser` parsing expressions `field=value`, `AND`, `OR`, `NOT`, parentheses into an AST. `async query_logs_aql(tenant_id, aql_expr, requested_by, ...)` uses the AST evaluator instead of substring match. Return `{"parsed_ast": ..., "results": [...], "plan": [...]}`.
**Competitor**: Grafana LogQL, Splunk SPL, Loki label filter expressions, AWS CloudWatch Insights query language.

---

### I14. Pipeline Health Scoring with SLO Budget Burn Rate
**Category**: Reliability / SRE
**Justification**: SLO burn rate alerting (Google SRE Book, ch. 5) is the gold standard for catching degradation before user impact. Scoring each ingestion pipeline on error rate, latency p99, and throughput against declared SLOs — and computing multiwindow burn rate (1 h / 6 h) — gives SREs actionable alerts 10x earlier than threshold alerting.
**Implementation**: Add `PipelineHealthScore` model with `error_rate`, `p99_ms`, `throughput_rps`, `slo_burn_1h`, `slo_burn_6h`, `health_score` (0-100). `async pipeline_health_score(tenant_id, pipeline_id)` computes from span/log data. `async slo_burn_report(tenant_id)` covers all pipelines.
**Competitor**: Google SRE Workbook burn-rate alerts, Nobl9, Sloth (prometheus-based SLO), Datadog SLO monitoring.

---

### I15. Columnar Log Compaction with Parquet-Compatible Schema Export
**Category**: Storage Efficiency / Analytics
**Justification**: Row-oriented in-memory storage (current approach) wastes 10–50x space compared to columnar storage for analytic workloads. Exporting compacted logs in Parquet-compatible columnar JSON (column arrays instead of row dicts) enables direct ingestion by DuckDB, Apache Arrow, BigQuery, and Redshift Spectrum — eliminating ETL pipelines for compliance reporting.
**Implementation**: `async compact_logs_columnar(tenant_id, pipeline_id, chunk_size)` groups logs into column-oriented chunks: `{"id": [...], "severity": [...], "message": [...], ...}`. Applies run-length encoding on low-cardinality fields (`severity`, `service_name`). Returns `{"chunks": N, "row_count": N, "schema": ..., "compression_ratio": Decimal}`.
**Competitor**: Apache Parquet, DuckDB columnar store, ClickHouse MergeTree, BigQuery columnar storage.
