# Predictive Analytics

## Overview
The Predictive Analytics capability (bia_pda) provides ML-based model training and deployment, demand and time-series forecasting, trend analysis, regression modelling, scenario simulation, and prediction serving — all tenant-scoped with full versioning, governance, and audit trails.

## Capability ID
`bia_pda`

## Provides
- ml_model_training: Train 11 model types with auto-versioning
- demand_forecasting: Generate point, interval, and distribution forecasts
- trend_analysis: Decompose and characterise linear/seasonal/cyclical trends
- regression_modelling: Linear, logistic, gradient boosting, and neural network regression
- scenario_simulation: Optimistic/pessimistic/stress-test scenario comparison
- anomaly_prediction: Isolation forest and LSTM autoencoder anomaly scoring
- model_versioning: Automatic version tagging on each training run
- prediction_serving: Low-latency synchronous prediction endpoint

## Requires
| Capability | Reason |
|------------|--------|
| auth | User identity and permission checks |
| audl | Audit trail for model training and serving |
| mten | Tenant context enforcement |
| conf | Runtime configuration |
| schd | Scheduled retraining jobs |
| mqeb | Streaming model lifecycle events |
| moni | Model drift and accuracy monitoring |
| bia_anl | Training data sourced from analytical queries |

## Configuration
| Option | Default | Description |
|--------|---------|-------------|
| min_training_samples | 100 | Minimum samples required for training |
| max_scenarios_per_model | 10 | Scenario limit per model |
| max_features | 500 | Feature store size limit |
| confidence_interval_default | 0.95 | Default CI for forecasts |
| auto_versioning | true | Automatic version increment on retrain |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/pda/models | GET/POST | List/train models | bia_pda:models |
| /api/bia/pda/models/<id>/deploy | POST | Deploy model | bia_pda:train |
| /api/bia/pda/forecasts | GET/POST | List/generate forecasts | bia_pda:forecasts |
| /api/bia/pda/scenarios | GET/POST | List/simulate scenarios | bia_pda:scenarios |
| /api/bia/pda/features | GET/POST | Feature store | bia_pda:features |
| /api/bia/pda/predict | POST | Serve prediction | bia_pda:models |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| min_samples_enforced | <100 training samples | deny |
| forecast_requires_deployed_model | model_state=training | deny |
| deprecated_model_cannot_be_deployed | state=deprecated | deny |
| failed_model_cannot_serve | state=failed | deny |
| scenario_limit_enforced | >10 scenarios | deny |

## Data Models
- MLModelResponse: id, tenant_id, name, model_type, state, version, accuracy_score, trained_at
- ForecastResponse: id, model_id, horizon, output_type, confidence_interval, forecast_data
- ScenarioResponse: id, model_id, name, scenario_type, parameters, results
- FeatureResponse: id, name, feature_type, source_column, datasource_id
- PredictionResponse: id, model_id, input_data, output, confidence, served_at

## Streaming Events
- model_training_started, model_trained, model_deployed, model_deprecated
- forecast_generated, trend_analysed, scenario_simulated, prediction_served
- feature_registered, anomaly_predicted

## Edge Cases Handled
- Deprecated models cannot be redeployed — require training a new version
- Failed models reject serving requests — explicit retrain or rollback required
- Scenario limit per model enforced — prevents unbounded scenario proliferation
- Auto-versioning is mandatory — cannot be disabled at the tenant level
- Forecasts validate that the backing model is in deployed state, not just trained

## Composability Notes
- bia_psa consumes bia_pda model outputs as baseline for optimisation
- bia_tsa provides high-frequency time series data as training input
- bia_dwh provides dimensional data as structured training datasets
- moni tracks model accuracy drift and triggers retraining workflows
- schd drives periodic retraining with wflo approval gates
