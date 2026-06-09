# Monitoring and Observability

**Capability ID**: `moni` | **Domain**: `common` | **Version**: `1.0.0`

## Description

MONI is APG's tenant-scoped monitoring and observability capability. It gives generated applications a dependency-light control plane for registering signal sources, governing metrics/logs/traces, managing SLOs, routing alerts,

## Installation

```bash
pip install apg-common-moni
```

## Provides

- `observability_governance`
- `metrics_lifecycle`
- `monitoring_agent_composition`
- `review_evidence`

## Requires

- `conf`
- `audl`
- `mqeb`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/moni/dashboard` | `moni:view` | Overview |
| `/moni/sources` | `moni:manage_sources` | Signals |
| `/moni/metrics` | `moni:view_metrics` | Signals |
| `/moni/logs` | `moni:view_logs` | Signals |
| `/moni/alerts` | `moni:manage_alerts` | Signals |
| `/moni/traces` | `moni:view_traces` | Signals |
| `/moni/slos` | `moni:manage_slos` | Reliability |
| `/moni/incidents` | `moni:manage_incidents` | Reliability |

## Key Service Methods

- `initialize()`
- `shutdown()`
- `track_metric()`
- `query_metrics()`
- `create_alert_rule()`
- `get_health_status()`
- `detect_anomalies()`
- `predict_resource_usage()`
- `analyze_performance()`
- `get_active_alerts()`

_(See `service.py` for complete API.)_

## Interoperability

`moni` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use moni;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MONI_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
