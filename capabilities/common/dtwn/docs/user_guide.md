# Digital Twin Framework

**Capability ID**: `dtwn` | **Domain**: `common` | **Version**: `1.0.0`

## Description

DTWN is the APG capability for governed digital twins, simulation models, authenticated telemetry fusion, topology mapping, prediction review, AI twin-agent governance, audit, and lifecycle stream metadata. It gives generated

## Installation

```bash
pip install apg-common-dtwn
```

## Provides

- `twin_registry`
- `simulation_models`
- `telemetry_fusion`
- `prediction_workflows`
- `asset_topology`

## Requires

- `pred`
- `iotd`
- `geos`
- `cvsn`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/dtwn/dashboard` | `dtwn:view` | Overview |
| `/dtwn/twins` | `dtwn:manage_twins` | Twins |
| `/dtwn/models` | `dtwn:model` | Models |
| `/dtwn/telemetry` | `dtwn:view` | Signals |
| `/dtwn/simulations` | `dtwn:simulate` | Simulations |
| `/dtwn/predictions` | `dtwn:view` | Intelligence |
| `/dtwn/topology` | `dtwn:view` | Twins |
| `/dtwn/agents` | `dtwn:model` | Agents |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_twin()`
- `register_simulation_model()`
- `ingest_telemetry()`
- `link_topology()`
- `run_simulation()`
- `record_prediction()`
- `review_prediction()`
- `register_twin_agent()`

_(See `service.py` for complete API.)_

## Interoperability

`dtwn` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use dtwn;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `DTWN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
