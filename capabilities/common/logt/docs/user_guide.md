# Logging and Tracing

**Capability ID**: `logt` | **Domain**: `common` | **Version**: `1.0.0`

## Description

LOGT provides APG applications with a tenant-scoped observability runtime: structured log ingestion, distributed trace roots, span recording, diagnostic search, approved diagnostic exports, retention policy, audit evidence,

## Installation

```bash
pip install apg-common-logt
```

## Provides

- `structured_logging`
- `distributed_tracing`
- `trace_correlation`
- `log_search`
- `diagnostic_retention`

## Requires

- `moni`
- `conf`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/logt/dashboard` | `logt:view` | Overview |
| `/logt/logs` | `logt:query` | Diagnostics |
| `/logt/traces` | `logt:query` | Diagnostics |
| `/logt/spans` | `logt:query` | Diagnostics |
| `/logt/pipelines` | `logt:manage_pipelines` | Pipelines |
| `/logt/retention` | `logt:manage_retention` | Governance |
| `/logt/agents` | `logt:admin` | Operations |
| `/logt/analytics` | `logt:view` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_retention_policy()`
- `create_pipeline()`
- `ingest_log()`
- `ingest_trace()`
- `record_span()`
- `search_logs()`
- `export_logs()`
- `create_record()`

_(See `service.py` for complete API.)_

## Interoperability

`logt` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use logt;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `LOGT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
