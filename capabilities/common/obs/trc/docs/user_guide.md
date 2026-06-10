# obs_trc User Guide — Distributed Tracing

## Overview

`obs_trc` provides OpenTelemetry-compatible distributed tracing for the APG platform. It collects spans, groups them into traces, builds a service dependency graph, and exports to Jaeger, Tempo, or any OTLP-compatible backend.

## Core Concepts

- **Span**: a single unit of work within one service (has start/end time, status, tags, logs).
- **Trace**: a directed graph of causally-related spans sharing a `trace_id`.
- **Service Dependency Map**: automatically derived from parent→child span relationships across service boundaries.
- **Sampling Rule**: controls which spans are recorded (probabilistic, rate-limiting, always-on/off).
- **Export Config**: configures push to Jaeger/Tempo/Zipkin/OTLP.

## Use Cases

1. **Request tracing**: create a root span at API ingress, propagate `trace_id` and `span_id` downstream, finish spans as each service completes its work.
2. **Latency diagnosis**: use `GET /analytics/slow-spans?threshold_ms=500` to surface p99 outliers.
3. **Error attribution**: `GET /analytics/statistics?service_name=payment-svc` shows error rates per service.
4. **Service topology**: `GET /service-map` returns all services and their call dependencies with p99 latencies.
5. **Sampling control**: create `always_off` rules for high-volume health-check endpoints, `probabilistic` at 10% for routine requests, `always_on` for error spans.

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

## API Reference

See `README.md` for the full endpoint table. All endpoints accept `X-Tenant-ID` header.

## Supported Exporters

- `jaeger` — Jaeger HTTP collector
- `tempo` — Grafana Tempo OTLP/HTTP
- `otlp` — OpenTelemetry Collector (gRPC or HTTP)
- `zipkin` — Zipkin HTTP API

## Supported Sampling Strategies

| Strategy | Behaviour |
|----------|-----------|
| `probabilistic` | Sample at `sample_rate` fraction (0.0–1.0) |
| `rate_limiting` | Allow up to N spans/second |
| `always_on` | Record all matching spans |
| `always_off` | Drop all matching spans |

Rules are evaluated in `priority` order (lowest first). First matching rule wins.
