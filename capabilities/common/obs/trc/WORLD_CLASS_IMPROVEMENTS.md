# obs_trc — World Class Improvements

15 high-leverage improvements benchmarked against industry-leading tracing systems.

---

### I1. Adaptive Head-Based Sampling with Feedback Loop

**Category**: Sampling | **Justification**: Static probabilistic rules waste budget on boring fast paths and under-sample slow/error paths. An adaptive sampler that re-weights based on observed latency distributions captures 10× more signal per stored span. **Implementation**: Maintain per-service exponential moving averages (EMA) of latency and error rate. Dynamically adjust `sample_rate` so high-p99 or high-error-rate operations receive `always_on` treatment while low-risk operations are dropped to 1–5%. Re-evaluate every 30 s via a background asyncio task. **Competitor**: Google Dapper's adaptive sampling; AWS X-Ray reservoir sampling with target rates.

---

### I2. Tail-Based Sampling via NATS JetStream Buffer

**Category**: Sampling | **Justification**: Head-based sampling discards traces that turn out to be errors or slow; tail-based sampling makes the decision *after* the trace is complete, achieving near-perfect recall for interesting traces. **Implementation**: Publish all spans to a NATS JetStream subject `obs.trc.spans.raw` with a short retention window (30 s). A `TailSamplerWorker` subscribes, buffers spans per `trace_id`, then applies tail rules (error? slow? anomalous service sequence?) on trace completion and either commits to permanent store or discards. **Competitor**: Jaeger's `stratified` sampler; Honeycomb's dynamic sampling.

---

### I3. Exemplar Linking to Metrics (OpenMetrics Compatibility)

**Category**: Integration | **Justification**: Bridging traces to metrics closes the observability loop — engineers jump directly from a Prometheus spike to the causal trace. **Implementation**: When a span finishes with `duration_ms > p99_threshold`, attach a metrics exemplar record (`{trace_id, span_id, value, timestamp}`) to an in-memory exemplar store keyed by metric name and label set. Expose a `GET /exemplars?metric=<name>&labels=<json>` endpoint returning OpenMetrics exemplar format. **Competitor**: Prometheus 2.43+ native exemplars; Grafana Mimir exemplar storage.

---

### I4. Automatic Anomaly Detection on Span Latency (z-score + IQR)

**Category**: Analytics | **Justification**: Manual threshold tuning is fragile. Statistical anomaly detection flags unexpected latency regressions within minutes of a deployment, without hardcoded thresholds. **Implementation**: Maintain a rolling 5-minute histogram (1 s buckets) per `(service_name, operation_name)` pair. On each `finish_span`, compute z-score against the rolling mean/stddev and flag spans with `|z| > 3` as anomalous. Also apply IQR fencing. Expose `GET /analytics/anomalies` returning ranked anomalous spans. **Competitor**: Lightstep anomaly detection; Datadog APM latency anomaly monitors.

---

### I5. Distributed Context Propagation via W3C TraceContext + Baggage

**Category**: Standards Compliance | **Justification**: W3C TraceContext (RFC 7230) is now the interoperability standard; supporting it eliminates custom propagation glue code across polyglot microservices. **Implementation**: Add `parse_traceparent(header: str)` and `build_traceparent(trace_id, span_id, sampled)` helpers. Accept `traceparent` and `tracestate` HTTP headers in `create_span` alongside the existing API fields. Encode baggage per W3C Baggage spec. **Competitor**: OpenTelemetry SDK propagation API; AWS X-Ray `X-Amzn-Trace-Id` header.

---

### I6. Span Compression and Deduplication for High-Cardinality Operations

**Category**: Storage Efficiency | **Justification**: Loop-intensive services emit thousands of near-identical spans (e.g., DB row fetches). Storing raw copies blows up storage 100×. Compression reduces storage 90%+ while preserving statistical accuracy. **Implementation**: After `finish_span`, check if an identical `(service_name, operation_name, tags hash)` span already exists within the last 5 s. If so, increment a `repetition_count` field on the prototype span and discard the duplicate. Expose the aggregated span with min/max/avg/p99 duration. **Competitor**: Honeycomb's column-oriented storage compression; Jaeger adaptive sampling + span deduplication.

---

### I7. Real-Time Critical Path Analysis

**Category**: Analytics | **Justification**: The critical path determines end-to-end trace latency. Identifying it cuts MTTR from hours to minutes by focusing engineer attention on the one sequential bottleneck rather than all slow spans. **Implementation**: For a completed trace, build a DAG from `parent_span_id` links. Run a longest-path algorithm (topological sort + DP on `duration_ms`) to identify the critical path. Expose `GET /traces/<id>/critical-path` returning the ordered list of spans on the critical path with their contribution percentages. **Competitor**: Netflix's critical-path tracing (Edgar); Uber Jaeger critical-path analysis.

---

### I8. NATS-Based Live Span Streaming (Push to Subscribers)

**Category**: Integration | **Justification**: Polling for new spans introduces seconds of latency during incidents. Push-based streaming lets dashboards and alerting systems react in <100 ms. **Implementation**: On every `create_span` and `finish_span`, publish a CloudEvent to NATS subject `obs.trc.spans.live.<tenant_id>`. Add `GET /streams/spans` SSE endpoint that subscribes to the NATS subject and fans out to HTTP clients. Also support WebSocket upgrade. **Competitor**: Grafana Tempo's live tail; Datadog APM live tail (uses internal Kafka — we use NATS).

---

### I9. Per-Tenant Retention Policies with Automated TTL Eviction

**Category**: Multi-Tenancy | **Justification**: Without retention limits, in-memory stores grow without bound. Per-tenant TTL lets SRE teams enforce SLAs on trace storage independently per tenant. **Implementation**: Add a `RetentionPolicy` per tenant (`max_age_seconds`, `max_span_count`, `max_trace_count`). A background asyncio task runs every 60 s, evicting spans older than `max_age_seconds` and pruning to `max_span_count` using LRU order. **Competitor**: Elastic APM retention management; Jaeger Cassandra TTL configuration.

---

### I10. Flamegraph-Ready Span Tree Serialisation

**Category**: Visualisation | **Justification**: Flamegraphs reduce trace comprehension time from minutes to seconds. Providing a ready-to-render format eliminates bespoke frontend transformation code. **Implementation**: Add `GET /traces/<id>/flamegraph` returning spans serialised to the Flamescope/Inferno JSON format: `{name, value, children:[...]}` where `value` is `duration_ms` and children are sorted child spans. **Competitor**: Datadog flame graphs; Pyroscope continuous profiling with trace linking; Jaeger UI trace tree.

---

### I11. OpenTelemetry Resource Attribute Enrichment

**Category**: Standards Compliance | **Justification**: Raw spans lack deployment context (k8s pod, region, image tag). Automatic resource enrichment adds this at ingestion time without SDK changes, giving instant fleet-wide filtering. **Implementation**: Add a `ResourceEnricher` that, on `create_span`, merges a configured resource attribute set (`service.version`, `deployment.environment`, `k8s.pod.name`, `cloud.region`) into the span's tags. Resource attributes are configured per `service_name` via a `POST /resource-attrs` endpoint. **Competitor**: OpenTelemetry SDK resource detection; AWS Distro for OpenTelemetry auto-instrumentation.

---

### I12. Trace Comparison and Regression Detection

**Category**: Analytics | **Justification**: Comparing traces from before/after a deployment instantly surfaces regressions without needing full APM baselines. **Implementation**: Add `POST /traces/compare` accepting two `trace_id` values. Returns a diff: spans present in one but not the other, latency delta per operation, new errors introduced. Sort by impact (latency delta × call frequency). **Competitor**: Lightstep regression detection; Honeycomb BubbleUp analysis.

---

### I13. Multi-Hop Correlation: Traces + Logs + Metrics

**Category**: Integration | **Justification**: Correlating all three pillars of observability in one API call halves the number of context switches during incident response. **Implementation**: Extend `correlate_trace_with_logs` to also accept a metric query window. Return a unified `ObservabilityCorrelation` payload: `{trace_summary, log_query_hints, metric_query_hints, anomaly_flags}`. Log hints include `trace_id` and `span_id` fields formatted as Loki label matchers; metric hints include Prometheus label selectors derived from span tags. **Competitor**: Grafana's "Explore" correlations; Elastic Observability unified view.

---

### I14. Intelligent Rate-Limiting Sampler with Token Bucket

**Category**: Sampling | **Justification**: The current `rate_limiting` strategy is a stub. A real token-bucket implementation prevents sampling budget exhaustion under traffic spikes while guaranteeing minimum throughput during quiet periods. **Implementation**: For each `(tenant_id, service_name)` pair, maintain a `TokenBucket(capacity, refill_rate)`. `_apply_sampling` consumes one token per span attempt; returns `True` only if a token is available. Refill asynchronously at `refill_rate` tokens/second via a periodic background coroutine. **Competitor**: OpenTelemetry `LeakySampler`; Jaeger remote sampler with rate limiting.

---

### I15. Trace-to-Profile Linking via Continuous Profiling Integration

**Category**: Integration | **Justification**: Slow traces without profiling data leave root cause as "DB is slow" or "CPU". Linking spans to pprof/pyspy profiles narrows root cause to the exact function. **Implementation**: Add `profile_url` and `profile_type` fields to the span schema (`cpu`, `memory`, `goroutine`). When a span is flagged as anomalous (see I4), emit a `profile_requested` event to NATS subject `obs.trc.profile.request.<service_name>` for a sidecar profiler to handle. Provide `POST /spans/<id>/attach-profile` to link a profile artifact URL. **Competitor**: Pyroscope span profiling integration; Datadog Continuous Profiler trace linking.
