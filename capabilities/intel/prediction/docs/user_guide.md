# Predictive Intelligence

**Capability ID**: `intel_prediction` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_prediction` is an executable APG capability package for building governed predictive-intelligence applications. It gives generated APG apps a concrete runtime for lawful authority, analytical workspaces, scenarios,

## Installation

```bash
pip install apg-intel-prediction
```

## Provides

- `prediction_authority_workflow`
- `prediction_workspace_workflow`
- `prediction_scenario_workflow`
- `prediction_indicator_workflow`
- `prediction_model_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-prediction/dashboard` | `intel_prediction:view` | Overview |
| `/intel-prediction/authorities` | `intel_prediction:authorities` | Governance |
| `/intel-prediction/workspaces` | `intel_prediction:workspaces` | Planning |
| `/intel-prediction/scenarios` | `intel_prediction:scenarios` | Planning |
| `/intel-prediction/indicators` | `intel_prediction:indicators` | Signals |
| `/intel-prediction/models` | `intel_prediction:models` | Models |
| `/intel-prediction/forecasts` | `intel_prediction:forecasts` | Forecasts |
| `/intel-prediction/projections` | `intel_prediction:projections` | Forecasts |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_workspace()`
- `record_scenario()`
- `record_indicator()`
- `record_model()`
- `record_forecast()`
- `record_projection()`
- `record_warning()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_prediction` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_prediction;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_PREDICTION_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
