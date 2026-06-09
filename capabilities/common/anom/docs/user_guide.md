# Anomaly Detection

**Capability ID**: `anom` | **Domain**: `common` | **Version**: `1.0.0`

## Description

ANOM is the APG capability for governed anomaly detection across monitored metrics, events, traces, forecast residuals, and security signals. It lets generated applications register monitoring sources, build statistical

## Installation

```bash
pip install apg-common-anom
```

## Provides

- `anomaly_detection`
- `signal_intelligence`
- `anomaly_agent_composition`

## Requires

- `pred`
- `aicr`
- `moni`
- `conf`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/anom/dashboard` | `anom:view` | Overview |
| `/anom/sources` | `anom:tune` | Sources |
| `/anom/baselines` | `anom:tune` | Baselines |
| `/anom/detector` | `anom:detect` | Detection |
| `/anom/signals` | `anom:detect` | Signals |
| `/anom/investigations` | `anom:investigate` | Investigations |
| `/anom/alerts` | `anom:investigate` | Investigations |
| `/anom/rules` | `anom:manage_rules` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_source()`
- `list_sources()`
- `create_baseline()`
- `list_baselines()`
- `reset_baseline()`
- `detect()`
- `list_signals()`
- `list_records()`

_(See `service.py` for complete API.)_

## Interoperability

`anom` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use anom;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ANOM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
