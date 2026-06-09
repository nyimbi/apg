# Predictive Analytics

**Capability ID**: `bia_pda` | **Domain**: `bia` | **Version**: `1.0.0`

## Description

The Predictive Analytics capability (bia_pda) provides ML-based model training and deployment, demand and time-series forecasting, trend analysis, regression modelling, scenario simulation, and prediction serving — all tenant-scoped with full versioning, governance, and audit trails.

## Installation

```bash
pip install apg-bia-pda
```

## Provides

- `ml_model_training`
- `demand_forecasting`
- `trend_analysis`
- `regression_modelling`
- `scenario_simulation`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `schd`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bia/pda/dashboard` | `bia_pda:view` | Overview |
| `/bia/pda/models` | `bia_pda:models` | Models |
| `/bia/pda/models/<id>` | `bia_pda:models` | Models |
| `/bia/pda/models/train` | `bia_pda:train` | Models |
| `/bia/pda/forecasts` | `bia_pda:forecasts` | Forecasting |
| `/bia/pda/forecasts/<id>` | `bia_pda:forecasts` | Forecasting |
| `/bia/pda/trends` | `bia_pda:trends` | Analysis |
| `/bia/pda/scenarios` | `bia_pda:scenarios` | Simulation |

## Key Service Methods

- `describe()`
- `create_model()`
- `train_model()`
- `evaluate_model()`
- `get_model()`
- `list_models()`
- `deploy_model()`
- `deprecate_model()`
- `delete_model()`
- `run_prediction()`

_(See `service.py` for complete API.)_

## Interoperability

`bia_pda` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bia_pda;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BIA_PDA_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
