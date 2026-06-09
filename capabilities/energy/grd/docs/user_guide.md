# Grid Operations

**Capability ID**: `energy_grd` | **Domain**: `energy` | **Version**: `1.0.0`

## Description

Grid Operations provides the real-time operational intelligence layer for power system management. It covers state estimation with convergence tracking, N-1/N-2 contingency analysis with automatic system status classification, voltage control via multiple methods (tap changers, SVCs, STATCOMs), frequency control including AGC and UFLS, market interval settlement with imbalance calculation, a full grid alarm management system with severity-gated acknowledgement, and EMS function execution in real-time and study modes.

## Installation

```bash
pip install apg-energy-grd
```

## Provides

- `real_time_state_estimation`
- `contingency_analysis`
- `voltage_control`
- `frequency_control`
- `market_settlement`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/energy-grd/dashboard` | `energy_grd:view` | Overview |
| `/energy-grd/state-estimation` | `energy_grd:state_estimation` | Real-Time |
| `/energy-grd/contingency` | `energy_grd:contingency` | Analysis |
| `/energy-grd/contingency/<id>` | `energy_grd:contingency` | Analysis |
| `/energy-grd/voltage-control` | `energy_grd:voltage_control` | Control |
| `/energy-grd/frequency-control` | `energy_grd:frequency_control` | Control |
| `/energy-grd/market-settlement` | `energy_grd:market_settlement` | Market |
| `/energy-grd/market-settlement/<id>` | `energy_grd:market_settlement` | Market |

## Key Service Methods

- `describe()`
- `evaluate()`
- `run_state_estimation()`
- `get_latest_se_run()`
- `list_se_runs()`
- `run_contingency()`
- `list_contingency_cases()`
- `apply_voltage_control()`
- `list_voltage_control_actions()`
- `apply_frequency_control()`

_(See `service.py` for complete API.)_

## Interoperability

`energy_grd` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use energy_grd;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ENERGY_GRD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
