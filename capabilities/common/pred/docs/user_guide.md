# Predictive Analytics

**Capability ID**: `pred` | **Domain**: `common` | **Version**: `1.0.0`

## Description

PRED is the APG capability for governed forecasting, scoring, scenario simulation, and predictive model operations. It lets generated applications register predictive models and feature sets, create forecasts, score entities,

## Installation

```bash
pip install apg-common-pred
```

## Provides

- `predictive_analytics`
- `forecasting`
- `prediction_agent_composition`

## Requires

- `aicr`
- `mlcm`
- `etlp`
- `conf`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pred/dashboard` | `pred:view` | Overview |
| `/pred/forecasts` | `pred:forecast` | Forecasts |
| `/pred/scores` | `pred:score` | Scoring |
| `/pred/features` | `pred:manage_models` | Scoring |
| `/pred/scenarios` | `pred:simulate` | Simulation |
| `/pred/models` | `pred:manage_models` | Models |
| `/pred/drift` | `pred:govern` | Models |
| `/pred/batch` | `pred:score` | Scoring |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_model()`
- `approve_model()`
- `register_feature_set()`
- `create_forecast()`
- `score_entity()`
- `simulate_scenario()`
- `record_drift()`
- `register_prediction_agent()`

_(See `service.py` for complete API.)_

## Interoperability

`pred` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pred;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PRED_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
