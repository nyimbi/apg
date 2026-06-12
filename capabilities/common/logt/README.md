# LOGT — Logging and Tracing Capability

LOGT provides APG applications with a tenant-scoped observability runtime:
structured log ingestion, distributed trace roots, span recording, diagnostic
search, approved diagnostic exports, retention policy, audit evidence,
observability agents, UI metadata, theme tokens, and Bytewax-backed lifecycle
events.

The package stays dependency-light. Production collectors, search indexes,
monitoring backends, compliance exporters, audit stores, and Bytewax workers
are represented as APG adapters in the executable contract and are bound by the
host application.

## What It Provides

- Structured log ingestion with tenant, pipeline, service, severity, trace,
  span, attribute, redaction, and privacy metadata.
- Distributed trace and span records with trace context, service ownership,
  duration validation, slow-span detection, and service-map summaries.
- Pipeline lifecycle with accountable owners, schema references, sampling
  policy, retention policy, and Bytewax stream enforcement.
- Diagnostic search with requester identity, large-query review, result
  counting, and audit evidence.
- Approved diagnostic exports for incident, compliance, and review bundles.
- First-class AI observability agents with runtime, role, scope,
  registration, and contribution-disclosure guardrails.
- Log pattern alert rules, compliance reports, and forwarding audit trails.
- Async aggregation, anomaly detection, trace correlation, and archival.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

## Main Files

| File | Purpose |
|---|---|
| `SPECIFICATION.md` | Normative capability behavior |
| `PLAN.md` | Implementation packet plan |
| `capability_contract.py` | Executable config, rules, routes, theme, adapters, provides/requires, Bytewax metadata |
| `models.py` | Tenant-scoped pipelines, logs, traces, spans, queries, exports, retention policies, audit events, agents |
| `observability_runtime.py` | Deterministic IDs, redaction, severity, query, span, service-map, sampling helpers |
| `service.py` | Runtime facade — sync core + async extended API |
| `api.py` | Package-safe helper functions |
| `views.py` | UI view models |
| `test_capability_contract.py` | Lifecycle behavior and generated evidence |

## Quick Start

```python
from capabilities.common.logt import LogtService

service = LogtService()

# 1. Retention policy
service.create_retention_policy(
    policy_id="retention-main",
    tenant_id="tenant-demo",
    name="Main diagnostics retention",
    log_retention_days=30,
)

# 2. Ingestion pipeline
service.create_pipeline(
    pipeline_id="pipeline-main",
    tenant_id="tenant-demo",
    name="Main diagnostics pipeline",
    owner="sre-team",
    schema_ref="schema://logs/v1",
    event_bus_ref="bytewax://diagnostics",
    sampling_policy="head-based-10pct",
    retention_policy_id="retention-main",
)

# 3. Ingest a log event
service.ingest_log(
    log_id="log-1",
    tenant_id="tenant-demo",
    pipeline_id="pipeline-main",
    service_name="orders-api",
    severity="info",
    message="order created",
)
```

## Core API

### Sync Methods

| Method | Description |
|---|---|
| `create_retention_policy(...)` | Define log/span retention with redaction and export-approval flags |
| `create_pipeline(...)` | Register a named ingestion pipeline bound to an owner, schema, event bus, and retention policy |
| `ingest_log(...)` | Store a structured log event with severity normalization and redaction enforcement |
| `ingest_trace(...)` | Record a distributed trace root with W3C-compatible trace context |
| `record_span(...)` | Attach a child span to a trace, compute slow-span status automatically |
| `search_logs(...)` | Substring-match log events with query audit trail and reviewer gating for large windows |
| `export_logs(export_id, ...)` | Create an approved diagnostic export bundle (incident, compliance, review) |
| `register_logt_agent(...)` | Register an AI observability agent with runtime, role, scope, and disclosure |
| `service_map(tenant_id)` | Derive a service dependency map from recorded spans |
| `dashboard_summary(tenant_id)` | Single-call health snapshot: counts, error logs, slow spans, streaming state |
| `list_pipelines/logs/traces/spans/queries/exports/retention_policies/audit_events/logt_agents(tenant_id)` | Paginate-ready list accessors, all tenant-scoped |

### Async Methods

| Method | Description |
|---|---|
| `query_logs(...)` | Async wrapper around `search_logs` with auto-generated query ID |
| `aggregate_logs(tenant_id, group_by, service_filter)` | Group and count log events by field (severity, service_name, pipeline_id) |
| `create_alert_on_log(...)` | Register a log-pattern alert rule, stored as an auditable pipeline tag |
| `log_retention_set(...)` | Upsert a retention policy (overwrites existing) |
| `structured_log_parse(tenant_id, raw_text, ...)` | Parse `key=value` log lines into structured log events and ingest automatically |
| `log_anonymize(tenant_id, log_id, fields_to_redact, actor)` | In-place attribute redaction with audit evidence |
| `compliance_log_report(...)` | Compliance-oriented summary: sensitive/redacted log counts, approved exports |
| `log_correlation(tenant_id, trace_id)` | Return all logs, spans, and traces correlated to a single `trace_id` |
| `trace_query(tenant_id, root_service, operation, status)` | Filter trace records by service, operation, or status |
| `span_export(tenant_id, trace_id, format, requested_by)` | Export all spans for a trace in Jaeger JSON or other formats |
| `log_anomaly_detect(tenant_id, service_name, error_rate_threshold)` | Flag services whose error rate exceeds threshold |
| `dashboard_create(tenant_id, name, panels, created_by)` | Register a named observability dashboard configuration |
| `log_forward(tenant_id, destination, log_ids, forwarded_by)` | Audit-trail log forwarding to external destinations |
| `log_archive(tenant_id, pipeline_id, older_than_days, actor)` | Mark aged log events as archived with audit evidence |

## AI Observability Agents

Register AI agents before they assist with diagnostic operations:

```python
agent = service.register_logt_agent(
    tenant_id="tenant-demo",
    name="Incident reviewer",
    runtime="codex",
    role="incident_reviewer",
    scope="Review slow spans and error logs before incident export",
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
Supported roles: pipeline, log, trace, incident, privacy, retention review.

## New Methods — Usage Examples

### 1. Structured Log Parsing

Parse nginx / Logfmt / k=v lines without writing custom ingestion code:

```python
record = await service.structured_log_parse(
    tenant_id="tenant-demo",
    raw_text='level=error msg="connection refused" host=db-01 latency_ms=1204',
    pipeline_id="pipeline-main",
    service_name="gateway",
)
# record["severity"] == "error", record["attributes"]["host"] == "db-01"
```

### 2. PII / Attribute Anonymization

Redact sensitive fields after ingestion (GDPR right-to-erasure pattern):

```python
await service.log_anonymize(
    tenant_id="tenant-demo",
    log_id="log-42",
    fields_to_redact=["email", "phone", "ssn"],
    actor="privacy-officer",
)
```

### 3. Trace Correlation

Collect all observability signals tied to a single distributed trace ID:

```python
context = await service.log_correlation(
    tenant_id="tenant-demo",
    trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
)
# context["logs"], context["spans"], context["traces"], context["total_items"]
```

### 4. Log Anomaly Detection

Detect services exceeding an error-rate SLO:

```python
result = await service.log_anomaly_detect(
    tenant_id="tenant-demo",
    error_rate_threshold=0.05,  # 5 %
)
# result["anomalies"] == [{"service": "payments", "error_rate": 0.12, "total_logs": 940}]
```

### 5. Compliance Report

Instant audit-ready summary for a compliance review:

```python
report = await service.compliance_log_report(
    tenant_id="tenant-demo",
    requested_by="compliance-bot",
)
# report keys: total_logs, sensitive_logs, redacted_logs,
#              exports_approved, queries_executed, pipelines, retention_policies
```

## World-Class Enhancements (v2.0)

The following 15 enhancements bring LOGT to production observability platform
standards. Each is architecturally justified against a category leader.

| # | Enhancement | Category | Competitor Benchmark |
|---|---|---|---|
| I1 | **Correlation ID propagation via `contextvars`** — `CorrelationContext` ContextVar carrying `trace_id`, `span_id`, `tenant_id`, `request_id`; zero-overhead async-safe propagation through `asyncio.create_task` | Observability Architecture | OpenTelemetry Python SDK, Datadog ddtrace |
| I2 | **Adaptive head + tail sampling engine** — `SamplingEngine.should_sample()` with `head_fixed_rate`, `tail_error`, `tail_latency_p99`, `adaptive_token_bucket` policies; stores decisions per `trace_id` | Sampling / Cost Control | Honeycomb Refinery, Jaeger adaptive sampling |
| I3 | **Semantic log-level budget with token-bucket rate limiting** — `LogBudgetLimiter` per `(tenant_id, service_name, severity)` triplet; `Decimal`-precise token counts; returns `retry_after_ms` on exhaustion | Performance / Cardinality | Vector throttle transform, Datadog per-service ingestion |
| I4 | **Immutable append-only log segments with Merkle-root integrity proof** — 100-event segments; SHA-256 Merkle root; `verify_log_integrity(tenant_id, segment_id)` returns `{valid, root}` | Compliance / Tamper Evidence | AWS CloudTrail log file validation, Splunk immutable archive |
| I5 | **OpenTelemetry OTLP ingest adapter with W3C TraceContext parsing** — `ingest_otlp_batch()` parses `ResourceSpans`/`ScopeLogs` JSON; extracts `traceparent`/`tracestate`; bulk-ingests to internal models | Interoperability / Standards | Grafana Tempo, OpenTelemetry Collector |
| I6 | **EWMA anomaly detection** — `EWMADetector` per `(tenant_id, service_name)`; `is_anomaly(value, threshold_sigma=3.0)`; replaces static threshold with adaptive baseline; exposed via `log_anomaly_detect_ewma` | Intelligence / Incident Detection | Datadog APM anomaly detection, AWS DevOps Guru |
| I7 | **Distributed trace critical-path analysis** — `trace_critical_path(tenant_id, trace_id)` builds span DAG, runs topological sort + longest-path DP; returns `{critical_path, total_critical_ms, bottleneck_span}` | Performance Engineering | Jaeger UI, Honeycomb trace waterfall |
| I8 | **Multi-tenant log cardinality budget with Decimal accounting** — `cardinality_budget_check()` tracks unique attribute key counts per tenant; `cardinality_usage_report()` returns `{budget, used, pct_used, top_keys}` | Multi-Tenancy / Cost Fairness | Datadog custom metric limits, Grafana Cloud |
| I9 | **Log parsing pipeline with named-pattern registry** — `register_log_parser(name, pattern_type, pattern)` for `kv`, `json`, `regex`, `csv` patterns; `parse_with_named_pattern()` dispatches to registered parser then ingests | Ingestion Quality | Logstash grok, Vector remap, Fluent Bit parsers |
| I10 | **Span-level cost attribution with Decimal money tracking** — `CostAttribution` model; `attribute_span_cost(span_id, resource_class, rate_per_ms_decimal)`; `cost_report(group_by)` with Decimal-sum aggregation | FinOps / Observability Economics | AWS X-Ray cost analysis, Datadog Cost Management |
| I11 | **Consistent-hash log sampling for deterministic replay** — `ConsistentHashSampler.should_keep(log_id, rate)` via SHA-256 hash modulo; same log always sampled or always dropped across replays | Sampling / Reproducibility | Datadog probabilistic sampler, OTel TraceIdRatioBased |
| I12 | **Tenant-scoped PII masking rules with regex compilation cache** — `MaskingRuleSet` with pre-compiled `_compiled: dict[str, re.Pattern]`; `register_masking_rule(name, pattern, replacement)`; `mask_log_message()` applies all rules | Privacy / PII Protection | Google DLP, AWS Macie, Vector redact, Presidio |
| I13 | **Structured Audit Query Language (mini-AQL) with predicate pushdown** — `LogQueryParser` for `field=value AND/OR/NOT` expressions parsed to AST; `query_logs_aql()` evaluates predicates over log dicts; returns `{parsed_ast, results, plan}` | Diagnostic / Compliance | Grafana LogQL, Splunk SPL, AWS CloudWatch Insights |
| I14 | **Pipeline health scoring with SLO budget burn rate** — `PipelineHealthScore` model with `error_rate`, `p99_ms`, `throughput_rps`, `slo_burn_1h`, `slo_burn_6h`, `health_score` (0–100); `slo_burn_report()` covers all pipelines | Reliability / SRE | Google SRE burn-rate alerts, Nobl9, Datadog SLO |
| I15 | **Columnar log compaction with Parquet-compatible schema export** — `compact_logs_columnar(chunk_size)` produces column-array chunks `{id: [...], severity: [...], ...}`; run-length encoding on low-cardinality fields; returns `{chunks, row_count, schema, compression_ratio: Decimal}` | Storage Efficiency / Analytics | Apache Parquet, DuckDB, ClickHouse, BigQuery |

## Composition

LOGT composes with:

- `moni` — monitoring integration and operational dashboards.
- `conf` — tenant configuration and policy.
- `audl` — durable audit evidence.
- `srch` — persistent diagnostic search indexes.
- `anom` — anomaly detection over spans, traces, and logs.
- `comp` — compliance export and retention attestations.

Batch diagnostic mutation and ingestion pipelines must use the `bytewax`
event-stream adapter.

## Verification

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/logt/__init__.py \
    capabilities/common/logt/capability_contract.py \
    capabilities/common/logt/models.py \
    capabilities/common/logt/observability_runtime.py \
    capabilities/common/logt/service.py \
    capabilities/common/logt/api.py \
    capabilities/common/logt/views.py \
    capabilities/common/logt/app.py \
    capabilities/common/logt/test_capability_contract.py

./.venv/bin/pytest -q capabilities/common/logt/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/logt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/logt --json
```

Live collectors, search engines, monitoring backends, durable audit stores,
rendered UI, and Bytewax workers are integration concerns outside the package
proof.
