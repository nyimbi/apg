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
