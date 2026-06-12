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
- churn_risk_scoring: Per-customer P(churn) x CLV revenue-at-risk with Decimal precision
- retraining_automation: PSI- and accuracy-triggered automated retraining policies
- model_lineage: Full provenance DAG from dataset through features to predictions
- prediction_lift_roi: Incremental revenue lift and ROI estimation with Decimal arithmetic
- ab_experimentation: Champion/challenger Thompson Sampling experiments
- bayesian_hpo: GP-UCB Bayesian hyperparameter search
- sla_monitoring: P50/P95/P99 latency histograms and SLO compliance reporting

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
| sla_target_ms | 500 | Serving latency SLO ceiling in milliseconds |
| churn_psi_threshold | 0.2 | PSI threshold triggering retraining |
| ab_min_observations | 30 | Min observations per arm before significance test |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/pda/models | GET/POST | List/train models | bia_pda:models |
| /api/bia/pda/models/<id>/deploy | POST | Deploy model | bia_pda:train |
| /api/bia/pda/models/<id>/lineage | GET | Provenance DAG | bia_pda:models |
| /api/bia/pda/models/<id>/hpo | POST | Bayesian HPO | bia_pda:train |
| /api/bia/pda/forecasts | GET/POST | List/generate forecasts | bia_pda:forecasts |
| /api/bia/pda/scenarios | GET/POST | List/simulate scenarios | bia_pda:scenarios |
| /api/bia/pda/features | GET/POST | Feature store | bia_pda:features |
| /api/bia/pda/predict | POST | Serve prediction | bia_pda:models |
| /api/bia/pda/churn/score | POST | Churn risk + revenue-at-risk | bia_pda:models |
| /api/bia/pda/lift | POST | Prediction lift / ROI | bia_pda:models |
| /api/bia/pda/experiments | GET/POST | A/B experiments | bia_pda:experiments |
| /api/bia/pda/sla/<id> | GET | Serving SLA report | bia_pda:models |
| /api/bia/pda/retraining/<id>/policy | PUT | Retraining policy | bia_pda:train |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| min_samples_enforced | <100 training samples | deny |
| forecast_requires_deployed_model | model_state=training | deny |
| deprecated_model_cannot_be_deployed | state=deprecated | deny |
| failed_model_cannot_serve | state=failed | deny |
| scenario_limit_enforced | >10 scenarios | deny |
| ab_traffic_split_capped | traffic_split > 0.5 | deny |
| churn_clv_decimal_required | non-string CLV input | error |

## Data Models
- MLModelResponse: id, tenant_id, name, model_type, state, version, accuracy_score, trained_at
- ForecastResponse: id, model_id, horizon, output_type, confidence_interval, forecast_data
- ScenarioResponse: id, model_id, name, scenario_type, parameters, results
- FeatureResponse: id, name, feature_type, source_column, datasource_id
- PredictionResponse: id, model_id, input_data, output, confidence, served_at
- ChurnRiskResponse: id, model_id, results[{customer_id, churn_probability, revenue_at_risk_decimal, tier}]
- ABExperimentResponse: id, champion_id, challenger_id, status, winner, significance_reached
- SLAReportResponse: id, model_id, p50_ms, p95_ms, p99_ms, slo_compliance_pct
- ModelLineageResponse: id, model_id, node_count, edge_count, nodes[], edges[]
- PredictionLiftResponse: id, model_id, net_lift_per_action_decimal, annualised_roi_pct_decimal

## Streaming Events
- model_training_started, model_trained, model_deployed, model_deprecated
- forecast_generated, trend_analysed, scenario_simulated, prediction_served
- feature_registered, anomaly_predicted
- churn_risk_scored, retraining_policy_configured, retraining_trigger_evaluated
- model_lineage_accessed, prediction_lift_estimated
- ab_experiment_created, experiment_outcome_recorded
- bayesian_hpo_completed, sla_report_generated

## Edge Cases Handled
- Deprecated models cannot be redeployed — require training a new version
- Failed models reject serving requests — explicit retrain or rollback required
- Scenario limit per model enforced — prevents unbounded scenario proliferation
- Auto-versioning is mandatory — cannot be disabled at the tenant level
- Forecasts validate that the backing model is in deployed state, not just trained
- A/B traffic_split capped at 0.5 — challenger never receives majority traffic
- Churn CLV inputs validated as Decimal strings — float inputs raise AssertionError
- Bayesian HPO warm-starts with 3 random trials before GP-UCB acquisition
- SLA percentile computation returns 0.0 gracefully when no observations recorded

## New Method Reference
| Method | Description |
|--------|-------------|
| score_churn_risk | P(churn) x CLV per customer, Decimal revenue-at-risk |
| configure_retraining_policy | PSI/accuracy/cron retraining policy |
| evaluate_retraining_triggers | Check if policy thresholds are breached |
| get_model_lineage | Provenance DAG (EU AI Act Art. 13) |
| estimate_prediction_lift | ROI estimator with Decimal arithmetic |
| record_serving_latency | Append latency observation for SLA |
| get_serving_sla_report | P50/P95/P99 + SLO compliance report |
| create_ab_experiment | Thompson Sampling champion/challenger |
| record_experiment_outcome | Update Beta posteriors, auto-conclude |
| bayesian_hyperparameter_search | GP-UCB HPO, writes best config to model |

## Composability Notes
- bia_psa consumes bia_pda model outputs as baseline for optimisation
- bia_tsa provides high-frequency time series data as training input
- bia_dwh provides dimensional data as structured training datasets
- moni tracks model accuracy drift and triggers retraining workflows
- schd drives periodic retraining with wflo approval gates
- crm consumes churn_risk scores directly for campaign targeting
- fin consumes revenue_at_risk Decimal outputs for P&L planning

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements for bia_pda — Predictive Analytics
- **I2.** Online / Incremental Model Learning
- **I3.** Probabilistic Forecasting with Conformal Prediction Intervals
- **I4.** Churn Probability Scorer with Revenue Impact
- **I5.** Multi-Armed Bandit Champion/Challenger Framework
- **I6.** Demand Forecasting with Hierarchical Reconciliation
- **I7.** Model Carbon Footprint Tracker
- **I8.** Federated Model Aggregation
- **I9.** Automated Retraining Trigger Engine
- **I10.** Demand Sensing via External Signal Fusion
- **I11.** Model Lineage Graph
- **I12.** Prediction Monetisation: Revenue Lift Estimator
- **I13.** Hyperparameter Optimisation with Bayesian Search
- **I14.** Real-Time Model Serving SLA Dashboard
- **I15.** Feature Store with Point-in-Time Correctness

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
