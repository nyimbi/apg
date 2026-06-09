# Performance Management

**Capability ID**: `telecom_per` | **Domain**: `telecom` | **Version**: `1.0.0`

## Description

Telecom network performance management covering KPI monitoring across all network layers, SLA compliance tracking with breach alerting, capacity utilisation forecasting, trend analysis with ML-based predictions, configurable threshold management, benchmark gap analysis, and scheduled performance reporting.

## Installation

```bash
pip install apg-telecom-per
```

## Provides

- `kpi_monitoring_workflow`
- `sla_compliance_workflow`
- `capacity_utilisation_workflow`
- `trend_reporting_workflow`
- `performance_reporting_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-per/dashboard` | `telecom_per:view` | Overview |
| `/telecom-per/kpis` | `telecom_per:kpis` | KPIs |
| `/telecom-per/kpis/<id>` | `telecom_per:kpis` | KPIs |
| `/telecom-per/sla` | `telecom_per:sla` | SLA |
| `/telecom-per/capacity` | `telecom_per:capacity` | Capacity |
| `/telecom-per/trends` | `telecom_per:trends` | Analytics |
| `/telecom-per/thresholds` | `telecom_per:thresholds` | Configuration |
| `/telecom-per/benchmarks` | `telecom_per:benchmarks` | Analytics |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_kpi()`
- `update_kpi_status()`
- `record_sla_compliance()`
- `record_capacity()`
- `record_trend()`
- `set_threshold()`
- `record_benchmark()`
- `generate_report()`

_(See `service.py` for complete API.)_

## Interoperability

`telecom_per` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use telecom_per;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_PER_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
