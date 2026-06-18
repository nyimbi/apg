# obs_log — Log Aggregation

Structured log ingestion, correlation ID injection, retention policies, log level management, Loki export.

**Capability ID:** `obs_log` | **Domain:** observability | **Version:** 1.0.0

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/log/health` | Health check |
| GET | `/api/obs/log/describe` | Capability descriptor |
| GET | `/api/obs/log/entries` | List/filter log entries |
| GET | `/api/obs/log/entries/<id>` | Get log entry |
| POST | `/api/obs/log/entries` | Ingest a single log entry |
| POST | `/api/obs/log/entries/bulk` | Bulk ingest log entries |
| DELETE | `/api/obs/log/entries/<id>` | Delete a log entry |
| DELETE | `/api/obs/log/entries/purge` | Purge entries by service/timestamp |
| GET | `/api/obs/log/search` | Full-text search over messages |
| POST | `/api/obs/log/correlation` | Create correlation context |
| GET | `/api/obs/log/correlation` | List correlation contexts |
| GET | `/api/obs/log/correlation/<id>` | Get correlation context |
| DELETE | `/api/obs/log/correlation/<id>` | Delete correlation context |
| GET | `/api/obs/log/by-correlation/<correlation_id>` | Logs by correlation ID |
| GET | `/api/obs/log/by-trace/<trace_id>` | Logs by trace ID |
| GET | `/api/obs/log/retention` | List retention policies |
| POST | `/api/obs/log/retention` | Create retention policy |
| GET | `/api/obs/log/retention/<id>` | Get retention policy |
| PUT | `/api/obs/log/retention/<id>` | Update retention policy |
| DELETE | `/api/obs/log/retention/<id>` | Delete retention policy |
| POST | `/api/obs/log/retention/apply` | Enforce all retention policies |
| GET | `/api/obs/log/levels` | List log level overrides |
| POST | `/api/obs/log/levels` | Create level override (with optional expiry) |
| GET | `/api/obs/log/levels/<id>` | Get level override |
| PUT | `/api/obs/log/levels/<id>` | Update level override |
| DELETE | `/api/obs/log/levels/<id>` | Delete level override |
| GET | `/api/obs/log/levels/effective` | Get effective log level for service |
| GET | `/api/obs/log/loki` | List Loki export configs |
| POST | `/api/obs/log/loki` | Create Loki export config |
| GET | `/api/obs/log/loki/<id>` | Get Loki config |
| PUT | `/api/obs/log/loki/<id>` | Update Loki config |
| DELETE | `/api/obs/log/loki/<id>` | Delete Loki config |
| GET | `/api/obs/log/loki/export` | Render Loki push API payload |
| GET | `/api/obs/log/stats` | Log statistics (level distribution) |
| GET | `/api/obs/log/errors` | Error/critical summary |
| GET | `/api/obs/log/audit` | Audit event log |

## Headers

Pass `X-Tenant-ID: <tenant>` on every request for multi-tenant isolation.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Sampling-Based Log Ingestion | Performance | High-volume services generate millions of logs per hour; sampling at ingestion reduces storage costs by 60–90% while preserving statistical accuracy for non-error traffic | Implement per-service configurable sampling rates (head-based probabilistic, tail-based error-biasing); always capture ERROR/CRITICAL at 100%; store sampling metadata on each entry so analytics remain unbiased | Datadog Agent tail-based sampling, OpenTelemetry Collector probabilistic_sampler processor** [Enhancement]
- **I2. Structured Log Pattern Detection | Analytics | Ad-hoc message strings carry no machine-readable semantics; pattern extraction converts noisy free-text into queryable event types automatically | Regex + frequency clustering to auto-detect recurring message templates; assign a `pattern_id` to each entry; expose a `GET /patterns` endpoint that groups by template with count/rate statistics | Elasticsearch ECS field normalization, Splunk event type extraction** [Enhancement]
- **I3. Real-Time Log Streaming via SSE | UX/Integration | Developers need a live tail for debugging without polling; SSE streams entries matching a filter as they are ingested, enabling Grafana-like real-time dashboards | Maintain per-filter async queues fed by `ingest_log`; stream via `GET /stream?service_name=X&min_level=ERROR`; auto-expire stale connections after configurable idle timeout | Grafana Loki `tail` WebSocket, Splunk streaming API** [Enhancement]
- **I4. Log Anomaly Scoring | Intelligence | Static level thresholds miss rate spikes; anomaly scoring detects when error rate, message velocity, or field-value distribution deviates from the service's historical baseline | Per-service rolling mean/stddev of log rate; compute z-score at ingestion; flag entries with `anomaly_score > threshold` in `get_error_summary`; expose `GET /anomalies` | Dynatrace Davis AI, New Relic Lookout** [Enhancement]
- **I5. Field-Level Redaction and PII Masking | Compliance | GDPR/CCPA mandate PII removal from logs; a redaction pipeline prevents sensitive fields from persisting at rest | Per-tenant field redaction rules (regex on field names/values); apply at ingestion before storage; record which fields were redacted in `fields.__redacted`; support `MASK`, `HASH`, `DROP` strategies | Cribl Stream PII redaction, Datadog Sensitive Data Scanner** [Enhancement]
- **I6. Dead-Letter Queue for Failed Ingestions | Reliability | Malformed entries or validation errors are silently dropped in `bulk_ingest_logs`; a DLQ enables replay and forensic debugging | Append failed entries + error detail to `_dlq` list; expose `GET /dlq` and `POST /dlq/replay`; cap DLQ at 10k entries with FIFO eviction; emit `dlq_entry_added` audit events | Bytewax Dead Letter Topic, AWS SQS DLQ pattern** [Enhancement]
- **I7. Log Aggregation Windows and Rate Metrics | Analytics | Operations teams need time-bucketed log rates to build SLO dashboards; raw entry lists are insufficient for trend detection | Compute entry counts in configurable time buckets (1m/5m/1h) on demand; return `rate_per_minute`, `p99_latency_between_entries`, and `spike_factor` per service | Prometheus `rate()` function, Grafana time-series panels** [Enhancement]
- **I8. Cross-Service Log Join and Trace Reconstruction | Debugging | Distributed request traces span multiple services; reconstructing the causal chain from `correlation_id` + `trace_id` requires a join operation not provided today | `reconstruct_trace(trace_id)` fetches all entries sharing the trace, sorts by timestamp, groups by span_id, returns a waterfall-ordered span tree with timing gaps highlighted | Jaeger trace timeline, Honeycomb trace waterfall** [Enhancement]
- **I9. Log Export Formats: NDJSON, CSV, Parquet | Integration | Loki is one export target; data pipelines require NDJSON for streaming, CSV for analysts, and Parquet for Spark/DuckDB analytics | `export_logs(format, service_name, start_time, end_time)` serialises entries to requested format; Parquet via `pyarrow`; NDJSON via stdlib json with `\n`-delimiters; return filename + byte count | Elasticsearch `_export`, Splunk REST export API** [Enhancement]
- **I10. Intelligent Alert Rules on Log Patterns | Intelligence | Log-based alerting (e.g., "alert if >5 ERROR logs in 60s for payment-svc") requires a rule engine; today only raw query is available | `AlertRule` model: service_name, min_level, pattern_regex, threshold_count, window_seconds, cooldown_seconds; `evaluate_alert_rules()` scans recent entries, fires `alert_triggered` events with matched entry IDs | PagerDuty log-based alerting, Grafana Loki alert rules** [Enhancement]
- **I11. Log Entry Tagging and Labelling | Organisation | Ad-hoc investigation requires grouping entries by incident, feature flag, or deployment; fields are unindexed and hard to query at scale | `tag_log_entries(query_filter, tags)` applies a tag set to matching entries; `GET /entries?tag=incident-42` filters by tag; tags stored as `list[str]` in `entry["tags"]`; `GET /tags` lists all known tags with counts | Datadog log facets, Elastic tags field** [Enhancement]
- **I12. Adaptive Retention Based on Error Content | Operations | Blindly deleting old INFO logs while discarding ERROR logs at the same TTL wastes forensic evidence; level-aware retention keeps errors longer | Retention policy gains `error_retention_days` field (default 3x `retention_days`); `apply_retention_policies()` runs two sweeps: standard sweep for INFO/DEBUG, extended sweep for ERROR/CRITICAL | Splunk index-time field extraction for tiered retention, Datadog Log Indexes** [Enhancement]
- **I13. Webhook Notifications for Level-Breach Events | Integration | External incident management tools (PagerDuty, Slack) need push notifications when critical logs arrive; polling is infeasible at low latency | `WebhookConfig` model per tenant: URL, secret (HMAC-SHA256), min_level, service_filter; `_dispatch_webhooks(entry)` called post-ingestion for qualifying entries; exponential backoff on failure, max 3 retries | Grafana alerting webhooks, Datadog Event API** [Enhancement]
- **I14. Log Volume Budget Enforcement | Cost Control | Runaway services can fill storage and degrade search; volume budgets enforce a per-service daily log count cap | `LogBudget` model: service_name, max_entries_per_hour, max_bytes_per_day (estimated from message length); `ingest_log` checks budget before writing; returns `budget_exceeded` sentinel; exposes `GET /budgets` for monitoring | AWS CloudWatch log data protection, Datadog daily log volume caps** [Enhancement]
- **I15. Log Context Propagation Middleware Helpers | Developer Experience | Injecting correlation IDs manually in every service is error-prone; helper methods reduce boilerplate and standardise propagation | `propagate_context(headers, service_name)` extracts W3C `traceparent`, `X-Correlation-ID`, `X-Request-ID` from headers, creates or resolves a correlation context, and returns enriched headers to forward downstream | OpenTelemetry context propagation API, AWS X-Ray header propagation** [Enhancement]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
