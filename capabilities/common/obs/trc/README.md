# obs_trc — Distributed Tracing

OpenTelemetry trace collection, span correlation, service dependency map, Jaeger/Tempo export, trace sampling.

**Capability ID:** `obs_trc` | **Domain:** observability | **Version:** 1.0.0

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/trc/health` | Health check |
| GET | `/api/obs/trc/describe` | Capability descriptor |
| GET | `/api/obs/trc/spans` | List spans (filter by trace_id, service_name, status, error_only) |
| GET | `/api/obs/trc/spans/<span_id>` | Get span by ID |
| POST | `/api/obs/trc/spans` | Create a new span |
| PUT | `/api/obs/trc/spans/<span_id>/finish` | Finish a span (set end_time, status, error) |
| DELETE | `/api/obs/trc/spans/<span_id>` | Delete a span |
| POST | `/api/obs/trc/spans/bulk` | Bulk ingest spans (OTLP batch) |
| GET | `/api/obs/trc/traces` | List traces |
| GET | `/api/obs/trc/traces/<trace_id>` | Get trace with computed duration |
| DELETE | `/api/obs/trc/traces/<trace_id>` | Delete trace and all spans |
| GET | `/api/obs/trc/traces/<trace_id>/otlp` | Export trace in OTLP format |
| GET | `/api/obs/trc/service-map` | Full service dependency topology |
| GET | `/api/obs/trc/services/<service>/dependencies` | Per-service upstream/downstream |
| GET | `/api/obs/trc/sampling-rules` | List sampling rules |
| POST | `/api/obs/trc/sampling-rules` | Create sampling rule |
| GET | `/api/obs/trc/sampling-rules/<rule_id>` | Get sampling rule |
| PUT | `/api/obs/trc/sampling-rules/<rule_id>` | Update sampling rule |
| DELETE | `/api/obs/trc/sampling-rules/<rule_id>` | Delete sampling rule |
| GET | `/api/obs/trc/export-configs` | List export configs |
| POST | `/api/obs/trc/export-configs` | Create export config (jaeger/tempo/otlp/zipkin) |
| GET | `/api/obs/trc/export-configs/<id>` | Get export config |
| PUT | `/api/obs/trc/export-configs/<id>` | Update export config |
| DELETE | `/api/obs/trc/export-configs/<id>` | Delete export config |
| POST | `/api/obs/trc/export-configs/<id>/test` | Test export connectivity |
| GET | `/api/obs/trc/analytics/statistics` | Trace statistics (error rate, p99) |
| GET | `/api/obs/trc/analytics/slow-spans` | Find slowest spans above threshold |
| GET | `/api/obs/trc/audit` | Audit event log |

## Headers

Pass `X-Tenant-ID: <tenant>` on every request for multi-tenant isolation.
