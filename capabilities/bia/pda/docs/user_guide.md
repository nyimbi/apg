# Predictive Analytics — User Guide

**Capability ID**: `bia_pda` | **Domain**: `bia` | **Version**: `2.0.0`
**Copyright**: 2025 Datacraft | Author: Nyimbi Odero

## Overview

`bia_pda` covers the full ML lifecycle: model authoring, training, evaluation, deployment,
drift detection, automated retraining, batch inference, churn scoring with Decimal-precision
revenue-at-risk, A/B champion/challenger experiments, Bayesian HPO, model lineage, and
P99 serving SLA reporting. All operations are tenant-scoped with tamper-evident audit.

## Installation

```bash
pip install apg-bia-pda
```

## Provides

- `ml_model_training`, `demand_forecasting`, `trend_analysis`
- `regression_modelling`, `scenario_simulation`, `anomaly_prediction`
- `model_versioning`, `prediction_serving`, `churn_risk_scoring`
- `retraining_automation`, `model_lineage`, `prediction_lift_roi`
- `ab_experimentation`, `bayesian_hpo`, `sla_monitoring`

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
| `/bia/pda/models/<id>/lineage` | `bia_pda:models` | Models |
| `/bia/pda/models/<id>/hpo` | `bia_pda:train` | Models |
| `/bia/pda/forecasts` | `bia_pda:forecasts` | Forecasting |
| `/bia/pda/forecasts/<id>` | `bia_pda:forecasts` | Forecasting |
| `/bia/pda/trends` | `bia_pda:trends` | Analysis |
| `/bia/pda/scenarios` | `bia_pda:scenarios` | Simulation |
| `/bia/pda/churn` | `bia_pda:models` | Churn |
| `/bia/pda/experiments` | `bia_pda:experiments` | A/B Testing |
| `/bia/pda/sla` | `bia_pda:models` | SLA |

## Key Service Methods

### Core ML Lifecycle
- `create_model()`, `train_model()`, `evaluate_model()`, `deploy_model()`, `deprecate_model()`
- `get_model()`, `list_models()`, `delete_model()`, `model_registry()`

### Prediction Serving
- `run_prediction()`, `batch_predict()`, `serve_prediction()`, `prediction_explanation()`

### Churn and Revenue Intelligence
- `score_churn_risk(tenant_id, model_id, customer_ids, clv_map)` — Decimal revenue-at-risk

### Automated Retraining
- `configure_retraining_policy(tenant_id, model_id, psi_threshold, accuracy_floor, cron)`
- `evaluate_retraining_triggers(tenant_id, model_id)` — returns should_retrain bool

### Model Lineage
- `get_model_lineage(tenant_id, model_id)` — provenance DAG for EU AI Act Art. 13

### ROI Estimation
- `estimate_prediction_lift(tenant_id, model_id, cost_decimal, revenue_decimal, baseline_rate)`

### SLA Monitoring
- `record_serving_latency(tenant_id, model_id, latency_ms)`
- `get_serving_sla_report(tenant_id, model_id, period, sla_target_ms)`

### A/B Experimentation
- `create_ab_experiment(tenant_id, champion_id, challenger_id, traffic_split)`
- `record_experiment_outcome(tenant_id, experiment_id, model_id, reward)`

### Bayesian HPO
- `bayesian_hyperparameter_search(tenant_id, model_id, param_space, n_trials, optimise_for)`

### AutoML and Feature Store
- `auto_ml()`, `feature_importance()`, `register_feature()`, `model_drift_detection()`

_(52 total async methods — see `service.py` for complete API.)_

## Interoperability

`bia_pda` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bia_pda;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BIA_PDA_`.

## Usage Examples

### Churn Scoring with Revenue-at-Risk

```python
report = await svc.score_churn_risk(
    tenant_id="acme", model_id=model["id"],
    customer_ids=["c1", "c2"],
    clv_map={"c1": "8500.00", "c2": "24000.00"},
)
# report["results"][0]["revenue_at_risk_decimal"] -> Decimal string
# report["total_revenue_at_risk_decimal"] -> portfolio total
```

### Automated Retraining

```python
await svc.configure_retraining_policy(
    tenant_id="acme", model_id=model["id"],
    psi_threshold=0.20, accuracy_floor=0.78,
)
trigger = await svc.evaluate_retraining_triggers("acme", model["id"])
if trigger["should_retrain"]:
    await svc.train_model("acme", model["id"])
```

### A/B Champion/Challenger

```python
exp = await svc.create_ab_experiment(
    "acme", champion["id"], challenger["id"], traffic_split=0.2,
)
result = await svc.record_experiment_outcome(
    "acme", exp["id"], challenger["id"], reward=1,
)
# result["winner"] set once significance_reached=True (chi2 > 3.841, n >= 30)
```

### Bayesian HPO

```python
hpo = await svc.bayesian_hyperparameter_search(
    "acme", model["id"],
    param_space={"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]},
    n_trials=20, optimise_for="f1",
)
# hpo["best_config"] written to model["hyperparameters"] automatically
```

### SLA Monitoring

```python
await svc.record_serving_latency("acme", model["id"], latency_ms=38.2)
sla = await svc.get_serving_sla_report("acme", model["id"], sla_target_ms=200.0)
# sla["p99_ms"], sla["slo_compliance_pct"]
```

## Further Reading

- `service.py` — 52 async methods, full business logic
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 planned enhancements
- `README.md` — Quick reference
