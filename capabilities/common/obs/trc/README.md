# obs_trc — Distributed Tracing

OpenTelemetry trace collection, span correlation, service dependency map, Jaeger/Tempo export, trace sampling.

**Capability ID:** `obs_trc` | **Domain:** observability | **Version:** 1.1.0

## What's New in v1.1.0

- Critical-path analysis (`GET /traces/<id>/critical-path`)
- Flamegraph-ready span tree serialisation (`GET /traces/<id>/flamegraph`)
- Trace comparison / regression detection (`POST /traces/compare`)
- Resource attribute enrichment (auto-tags spans with deployment metadata)
- Per-tenant retention policies with TTL eviction
- Statistical anomaly detection on span latency (z-score + IQR)
- Token-bucket rate-limiting sampler (`POST /sampling/token-bucket/consume`)
- W3C TraceContext `traceparent` header parse/build
- Multi-pillar observability correlation (traces + Loki + Prometheus hints)

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/trc/health` | Health check |
| GET | `/api/obs/trc/describe` | Capability descriptor |
| GET | `/api/obs/trc/audit` | Audit event log |

### Spans

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/trc/spans` | List spans (filter: trace_id, service_name, status, error_only, min_duration_ms) |
| GET | `/api/obs/trc/spans/<span_id>` | Get span by ID |
| POST | `/api/obs/trc/spans` | Create a new span |
| PUT | `/api/obs/trc/spans/<span_id>/finish` | Finish a span |
| DELETE | `/api/obs/trc/spans/<span_id>` | Delete a span |
| POST | `/api/obs/trc/spans/bulk` | Bulk ingest spans (OTLP batch) |
| POST | `/api/obs/trc/spans/<span_id>/logs` | Attach a log entry to a span |
| PUT | `/api/obs/trc/spans/<span_id>/tags/<key>` | Set a single span tag |

### Traces

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/trc/traces` | List traces |
| GET | `/api/obs/trc/traces/<trace_id>` | Get trace with computed duration |
| DELETE | `/api/obs/trc/traces/<trace_id>` | Delete trace and all spans |
| GET | `/api/obs/trc/traces/<trace_id>/otlp` | Export trace in OTLP format |
| GET | `/api/obs/trc/traces/<trace_id>/critical-path` | **NEW** Critical-path DAG analysis |
| GET | `/api/obs/trc/traces/<trace_id>/flamegraph` | **NEW** Flamegraph-ready span tree |
| GET | `/api/obs/trc/traces/<trace_id>/correlation` | **NEW** Multi-pillar observability correlation |
| POST | `/api/obs/trc/traces/compare` | **NEW** Compare two traces (body: `{trace_id_a, trace_id_b}`) |

### Service Map

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/trc/service-map` | Full service dependency topology |
| GET | `/api/obs/trc/services/<service>/dependencies` | Per-service upstream/downstream |

### Sampling

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/trc/sampling-rules` | List sampling rules |
| POST | `/api/obs/trc/sampling-rules` | Create sampling rule |
| GET | `/api/obs/trc/sampling-rules/<rule_id>` | Get sampling rule |
| PUT | `/api/obs/trc/sampling-rules/<rule_id>` | Update sampling rule |
| DELETE | `/api/obs/trc/sampling-rules/<rule_id>` | Delete sampling rule |
| POST | `/api/obs/trc/sampling/token-bucket/consume` | **NEW** Consume one token-bucket token for rate-limiting |

### Export Configs

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/trc/export-configs` | List export configs |
| POST | `/api/obs/trc/export-configs` | Create export config (jaeger/tempo/otlp/zipkin) |
| GET | `/api/obs/trc/export-configs/<id>` | Get export config |
| PUT | `/api/obs/trc/export-configs/<id>` | Update export config |
| DELETE | `/api/obs/trc/export-configs/<id>` | Delete export config |
| POST | `/api/obs/trc/export-configs/<id>/test` | Test export connectivity |

### Resource Attributes (NEW)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/trc/resource-attrs` | List all resource attribute sets |
| GET | `/api/obs/trc/resource-attrs/<service>` | Get resource attributes for a service |
| PUT | `/api/obs/trc/resource-attrs/<service>` | Set resource attributes for a service |

### Retention Policy (NEW)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/trc/retention` | Get active retention policy |
| PUT | `/api/obs/trc/retention` | Set retention policy (max_age_seconds, max_span_count, max_trace_count) |
| POST | `/api/obs/trc/retention/evict` | Trigger eviction cycle manually |

### Analytics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/trc/analytics/statistics` | Trace statistics (error rate, p99) |
| GET | `/api/obs/trc/analytics/slow-spans` | Find slowest spans above threshold |
| GET | `/api/obs/trc/analytics/anomalies` | **NEW** Detect anomalous spans by z-score + IQR |

### W3C TraceContext (NEW)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/obs/trc/traceparent/parse` | Parse a W3C traceparent header |
| POST | `/api/obs/trc/traceparent/build` | Build a W3C traceparent header |

## Headers

Pass `X-Tenant-ID: <tenant>` on every request for multi-tenant isolation.

## Streaming Integration

Spans are published to NATS subject `obs.trc.spans.live.<tenant_id>` as CloudEvents on
`create_span` and `finish_span`.  Connect a NATS subscriber or use the SSE endpoint to
receive live span updates with sub-100 ms latency.  (Uses NATS+bytewax — not NATS JetStream.)
