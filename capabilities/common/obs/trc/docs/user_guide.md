# obs_trc User Guide — Distributed Tracing (v1.1.0)

## Overview

`obs_trc` provides OpenTelemetry-compatible distributed tracing for the APG platform. It collects
spans, groups them into traces, builds a service dependency graph, and exports to Jaeger, Tempo, or
any OTLP-compatible backend.

New in v1.1.0: critical-path analysis, flamegraph serialisation, trace comparison, resource attribute
enrichment, retention policies, statistical anomaly detection, token-bucket sampling, W3C TraceContext
parsing, and multi-pillar observability correlation.

## Core Concepts

- **Span**: a single unit of work within one service (has start/end time, status, tags, logs).
- **Trace**: a directed graph of causally-related spans sharing a `trace_id`.
- **Critical Path**: the longest sequential chain of spans that determines trace latency.
- **Service Dependency Map**: auto-derived from parent→child span relationships across service boundaries.
- **Sampling Rule**: controls which spans are recorded (probabilistic, rate-limiting, always-on/off).
- **Token Bucket**: rate-limiting sampler guaranteeing at most N spans/second per service.
- **Resource Attributes**: static deployment metadata (version, environment, pod) auto-attached to spans.
- **Retention Policy**: per-tenant TTL + count limits with async eviction.
- **Export Config**: push configuration for Jaeger/Tempo/Zipkin/OTLP.

## Quick Start

### 1. Create a root span

```http
POST /api/obs/trc/spans
X-Tenant-ID: my-org
Content-Type: application/json

{
  "operation_name": "handle_payment",
  "service_name": "payment-svc",
  "kind": "server",
  "tags": {"http.method": "POST", "http.path": "/payments"}
}
```

Response includes `id` (span_id) and `trace_id`.

### 2. Create a child span

```http
POST /api/obs/trc/spans
X-Tenant-ID: my-org

{
  "operation_name": "db_insert",
  "service_name": "payment-db",
  "trace_id": "<trace_id from step 1>",
  "parent_span_id": "<span_id from step 1>",
  "kind": "client"
}
```

### 3. Finish spans

```http
PUT /api/obs/trc/spans/<span_id>/finish
X-Tenant-ID: my-org

{"status": "ok"}
```

### 4. View the trace

```http
GET /api/obs/trc/traces/<trace_id>
```

### 5. Configure Jaeger export

```http
POST /api/obs/trc/export-configs
X-Tenant-ID: my-org

{
  "name": "jaeger-prod",
  "exporter_type": "jaeger",
  "endpoint": "http://jaeger:14268/api/traces",
  "batch_size": 512,
  "flush_interval_ms": 5000
}
```

---

## Critical Path Analysis

Find the single span sequence responsible for the most latency in a trace:

```http
GET /api/obs/trc/traces/<trace_id>/critical-path
```

Response:

```json
{
  "trace_id": "abc123",
  "total_duration_ms": 843.2,
  "critical_path": [
    {"span_id": "s1", "operation_name": "handle_payment", "service_name": "payment-svc",
     "duration_ms": 843.2, "contribution_pct": 100.0},
    {"span_id": "s2", "operation_name": "db_insert", "service_name": "payment-db",
     "duration_ms": 720.1, "contribution_pct": 85.4}
  ]
}
```

Use this to focus optimisation effort on the one sequential bottleneck.

---

## Flamegraph

Get a Flamescope/Inferno-compatible JSON tree for visual span analysis:

```http
GET /api/obs/trc/traces/<trace_id>/flamegraph
```

The `flamegraph` field is ready to pass to any Inferno-compatible renderer.  Each node:
`{"name": "service:operation", "value": <duration_ms>, "children": [...], "error": false}`.

---

## Trace Comparison

Compare a before/after trace to detect regressions after a deployment:

```http
POST /api/obs/trc/traces/compare

{
  "trace_id_a": "baseline_trace_id",
  "trace_id_b": "canary_trace_id"
}
```

Response includes `regressions`, `improvements`, `added_operations`, `removed_operations`,
and `new_errors_in_b`, sorted by absolute latency delta.

---

## Resource Attribute Enrichment

Configure static deployment metadata to auto-attach to every span from a service:

```http
PUT /api/obs/trc/resource-attrs/payment-svc

{
  "service.version": "2.4.1",
  "deployment.environment": "production",
  "k8s.pod.name": "payment-svc-84f9b-xvz2"
}
```

All subsequent spans from `payment-svc` will include these tags automatically, enabling
fleet-wide filtering without SDK changes.

---

## Retention Policies

Set a per-tenant retention policy to prevent unbounded memory growth:

```http
PUT /api/obs/trc/retention

{
  "max_age_seconds": 3600,
  "max_span_count": 100000,
  "max_trace_count": 10000
}
```

Trigger eviction manually (or schedule it via NATS timer):

```http
POST /api/obs/trc/retention/evict
```

Eviction removes spans older than `max_age_seconds`, caps to `max_span_count` (LRU), and
cleans up orphaned traces.

---

## Anomaly Detection

Detect statistically anomalous spans across all services or a specific one:

```http
GET /api/obs/trc/analytics/anomalies?service_name=payment-svc&z_threshold=3.0
```

Uses Welford online mean/variance (z-score) and IQR fencing.  Anomalies are ranked by
`|z_score|` descending.  Response includes `op_mean_ms` and `op_stdev_ms` for context.

---

## Token-Bucket Rate-Limiting Sampler

Consume a sampling token before creating high-volume spans:

```http
POST /api/obs/trc/sampling/token-bucket/consume

{"service_name": "metrics-collector"}
```

Response: `{"allowed": true, "tokens_remaining": 87.3, ...}`.
If `allowed` is `false`, skip `create_span` to stay within the rate budget.
Default: 100-token bucket refilling at 10 tokens/second per service.

---

## W3C TraceContext Propagation

Parse an incoming `traceparent` header from an upstream service:

```http
POST /api/obs/trc/traceparent/parse

{"header": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
```

Build a `traceparent` header to forward to a downstream service:

```http
POST /api/obs/trc/traceparent/build

{"trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "span_id": "00f067aa0ba902b7", "sampled": true}
```

---

## Multi-Pillar Observability Correlation

Get a unified correlation payload combining trace data with Loki log and Prometheus metric hints:

```http
GET /api/obs/trc/traces/<trace_id>/correlation
```

Response includes:
- `log_query_hints.loki_selector` — Loki query string pre-populated with `trace_id`
- `metric_query_hints.prometheus_queries` — PromQL `rate()` queries per service
- `anomaly_flags` — any anomalies detected for spans in this trace

Paste the Loki selector directly into Grafana Explore to jump from trace to logs in one click.

---

## Sampling Strategies

| Strategy | Behaviour |
|----------|-----------|
| `probabilistic` | Sample at `sample_rate` fraction (0.0–1.0) |
| `rate_limiting` | Token-bucket: allow up to N spans/second |
| `always_on` | Record all matching spans |
| `always_off` | Drop all matching spans |

Rules are evaluated in `priority` order (lowest first). First matching rule wins.

### Recommended Configuration

```python
# Drop noisy health checks entirely
await svc.create_sampling_rule("drop-healthchecks", sample_rate=0.0,
    operation_pattern=r"health.*", strategy="always_off", priority=10)

# Sample 10% of routine read operations
await svc.create_sampling_rule("low-rate-reads", sample_rate=0.1,
    operation_pattern=r"GET.*", strategy="probabilistic", priority=50)

# Always capture writes and errors
await svc.create_sampling_rule("always-writes", sample_rate=1.0,
    operation_pattern=r"(POST|PUT|DELETE|INSERT|UPDATE).*", strategy="always_on", priority=20)
```

---

## Supported Exporters

| Exporter | Protocol | Default Port |
|----------|----------|-------------|
| `jaeger` | Jaeger HTTP collector | 14268 |
| `tempo` | Grafana Tempo OTLP/HTTP | 4318 |
| `otlp` | OpenTelemetry Collector gRPC or HTTP | 4317 / 4318 |
| `zipkin` | Zipkin HTTP API v2 | 9411 |

---

## NATS Integration

Spans publish CloudEvents to NATS subjects on key lifecycle transitions:

| Event | NATS Subject |
|-------|-------------|
| span created | `obs.trc.spans.live.<tenant_id>` |
| span finished | `obs.trc.spans.live.<tenant_id>` |
| anomaly detected | `obs.trc.anomalies.<tenant_id>` |
| eviction completed | `obs.trc.eviction.<tenant_id>` |

Subscribe with a bytewax dataflow for real-time trace analytics or alert routing.

---

## API Reference

See `README.md` for the complete endpoint table. All endpoints accept `X-Tenant-ID` header.
