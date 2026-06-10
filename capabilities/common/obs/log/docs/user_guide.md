# obs_log User Guide — Log Aggregation

## Overview

`obs_log` provides structured log ingestion, correlation ID injection and propagation, retention policy enforcement, dynamic log level management, and Loki-compatible export.

## Core Concepts

- **Log Entry**: a structured log record with level, message, timestamp, service_name, and optional correlation/trace IDs and arbitrary key-value fields.
- **Correlation Context**: tracks a `correlation_id` bound to a request, user, or session for cross-service log correlation.
- **Retention Policy**: defines minimum log level, retention window, and archival/deletion rules per service or globally.
- **Level Override**: dynamically adjusts the minimum log level for a service/logger at runtime (with optional TTL).
- **Loki Export Config**: configures push to Grafana Loki.

## Use Cases

1. **Structured ingestion**: POST individual or bulk log entries with arbitrary `fields` for context.
2. **Cross-service correlation**: assign a `correlation_id` at API gateway ingress, propagate it downstream, query `GET /by-correlation/<id>` to see the complete request log trail.
3. **Trace-log linking**: store `trace_id` and `span_id` on log entries to cross-reference with `obs_trc` spans.
4. **Dynamic debug logging**: create a 30-minute DEBUG override for a noisy service without redeploying — expires automatically.
5. **Retention enforcement**: run `POST /retention/apply` (or schedule it) to delete entries older than the policy window.
6. **Loki export**: configure a Loki endpoint and call `GET /loki/export` to get a push-API payload ready for forwarding.

## Quick Start

### 1. Ingest a log entry

```http
POST /api/obs/log/entries
X-Tenant-ID: my-org
Content-Type: application/json

{
  "service_name": "payment-svc",
  "level": "ERROR",
  "message": "Charge failed: insufficient funds",
  "correlation_id": "req-abc123",
  "trace_id": "abcdef1234567890",
  "fields": {"user_id": "u42", "amount": 9900, "currency": "KES"}
}
```

### 2. Create a correlation context

```http
POST /api/obs/log/correlation
X-Tenant-ID: my-org

{
  "service_name": "api-gateway",
  "user_id": "u42",
  "session_id": "sess-xyz"
}
```

Response provides a `correlation_id` to propagate in downstream calls.

### 3. Retrieve logs by correlation

```http
GET /api/obs/log/by-correlation/req-abc123
X-Tenant-ID: my-org
```

### 4. Set a DEBUG override for 30 minutes

```http
POST /api/obs/log/levels
X-Tenant-ID: my-org

{
  "service_name": "payment-svc",
  "level": "DEBUG",
  "duration_minutes": 30,
  "reason": "Investigating charge failure"
}
```

### 5. Define a retention policy

```http
POST /api/obs/log/retention
X-Tenant-ID: my-org

{
  "name": "default-30d",
  "retention_days": 30,
  "min_level": "INFO",
  "archive_after_days": 7,
  "delete_after_days": 30
}
```

### 6. Configure Loki export

```http
POST /api/obs/log/loki
X-Tenant-ID: my-org

{
  "name": "loki-prod",
  "endpoint": "http://loki:3100/loki/api/v1/push",
  "extra_labels": {"env": "production"},
  "batch_size": 1000
}
```

## Log Levels

Ordered from lowest to highest verbosity: `TRACE < DEBUG < INFO < WARNING < ERROR < CRITICAL`

Level filtering is applied at ingestion time based on the effective minimum level (override > retention policy > DEBUG default).

## Full-Text Search

```http
GET /api/obs/log/search?query=charge+failed&service_name=payment-svc
```

Uses Python `re.search` with `re.IGNORECASE`. Supports regex patterns.

## API Reference

See `README.md` for the full endpoint table.
