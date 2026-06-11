# Predictive Intelligence — User Guide

**Capability ID**: `intel_prediction` | **Domain**: `intel` | **Version**: `2.0.0`

---

## Overview

`intel_prediction` is an executable APG capability package for building governed
predictive-intelligence applications. It provides a complete runtime for lawful authority
management, analytical workspaces, threat scenarios, signal indicators, validated ML models,
probabilistic forecasts, projections, early warnings, recommendations, and regulatory
compliance scoring.

The service is fully async. All write operations enforce deterministic policy rules before
mutating state. Violations raise `PermissionError` with a human-readable reason string.

---

## Installation

```bash
pip install apg-intel-prediction
```

---

## Quick Start

```python
import asyncio
from capabilities.intel.prediction import PredictiveIntelligenceService

svc = PredictiveIntelligenceService(tenant_id="acme", actor_id="analyst-1")

async def quick_start():
    # 1. Establish governance chain
    authority = svc.record_authority(
        "auth-1", "acme", "mission_order", "scope-ref",
        "confidential", "approver-1", "2027-01-01", "ev-auth",
    )
    ws = svc.record_workspace(
        "ws-1", "acme", "threat_prediction", "APT Workspace",
        "confidential", authority["id"], "ev-ws",
    )
    sc = svc.record_scenario(
        "sc-1", "acme", ws["id"], "geopolitical",
        "APT-42 lateral movement", "short_term", "analyst-1", "ev-sc",
    )
    ind = svc.record_indicator(
        "ind-1", "acme", sc["id"], "behavioral",
        "APT-42 c2_beacon", 0.82, "ev-ind",
    )

    # 2. Model lifecycle
    mdl = await svc.create_prediction_model(
        model_type="gradient_boost",
        training_data={"features": ["c2_count", "ttps"], "sample_count": 800},
        target_variable="intrusion_probability",
    )
    await svc.train_model(mdl["id"], features=["c2_count", "ttps"])
    await svc.model_deployment(mdl["id"])

    # 3. Inference
    result = await svc.prediction_run(
        mdl["id"], input_data={"c2_count": 12, "ttps": 0.75}
    )
    print("Intrusion probability:", result["output_probability"])

asyncio.run(quick_start())
```

---

## Core Concepts

### Governance Chain

Every action requires an established authority chain:
`PredictionAuthority` → `PredictionWorkspace` → `PredictionScenario`.

| Object | Purpose |
|---|---|
| `PredictionAuthority` | Legal mandate and classification approval |
| `PredictionWorkspace` | Bounded analytical environment |
| `PredictionScenario` | Specific threat or event under analysis |
| `PredictionIndicator` | Observable signal with confidence score |
| `PredictionModel` | Registered ML model specification |
| `PredictionForecast` | Model output with analyst attribution |
| `PredictionProjection` | Probabilistic risk projection |
| `PredictionWarning` | Threshold-triggered early warning |
| `PredictionRecommendation` | Approved response action |
| `PredictionReview` | Governance review record |
| `PredictionAgent` | Bounded AI agent registration |

---

## Model Lifecycle

```
CREATED → TRAINING → TRAINED → DEPLOYED → RETIRED
```

| Transition | Method |
|---|---|
| CREATED | `record_model` or `create_prediction_model` |
| TRAINING → TRAINED | `train_model` |
| TRAINED → DEPLOYED | `model_deployment` |
| DEPLOYED → STALE | `check_concept_drift` (automatic on PSI > threshold) |
| any → RETIRED | `model_retirement` |

---

## Async Method Reference

### Model Operations

#### `create_prediction_model(model_type, training_data, target_variable)`

Bootstraps a model under the first available scenario for the tenant. Infers risk level from
`sample_count` (< 100 → high, < 1000 → medium, ≥ 1000 → low).

```python
mdl = await svc.create_prediction_model(
    model_type="lstm",
    training_data={"features": ["ttp_vector", "geo_cluster"], "sample_count": 2000},
    target_variable="campaign_probability",
)
```

#### `train_model(model_id, features)`

Runs a training step (log-saturation accuracy curve: `1 - exp(-0.3 * run_n)`). Supports
repeated calls for incremental training.

#### `model_deployment(model_id)`

Promotes a `TRAINED` model to `DEPLOYED`. Raises `RuntimeError` if not yet trained.

#### `model_update(model_id, new_data)`

Online learning step. Calls `train_model` with features extracted from `new_data["features"]`.

#### `model_retirement(model_id, reason)`

Retires a model with audit trail. All state is preserved for compliance review.

---

### Inference

#### `prediction_run(model_id, input_data)`

Single inference run. When `OLLAMA_BASE_URL` is set, routes to the local Ollama instance via
`MLCapability.score()`. Falls back to a sigmoid scorer on the numeric feature mean.

```python
result = await svc.prediction_run(
    "mdl-abc", input_data={"ttp_vector": 0.6, "geo_cluster": 3.0}
)
# {"run_id": "...", "output_probability": 0.4812, "model_accuracy": 0.9502, ...}
```

#### `ensemble_predict(model_ids, input_data, weights=None)`

Weighted soft-voting across multiple models. Weights default to each model's latest accuracy.
Includes a Brier-score calibration decomposition.

```python
ensemble = await svc.ensemble_predict(
    model_ids=["mdl-a", "mdl-b", "mdl-c"],
    input_data={"ttp_vector": 0.6, "geo_cluster": 3.0},
)
print(ensemble["ensemble_probability"], ensemble["brier_calibration"])
```

#### `multi_horizon_forecast(model_id, input_data)`

Runs `prediction_run` once, then applies temporal decay (`exp(-0.001 * days)`) across all
five horizons: `near_term` (30d) through `strategic` (730d). Produces an inverse-day-weighted
consensus probability.

```python
mhf = await svc.multi_horizon_forecast("mdl-abc", {"ttp_vector": 0.6})
# {"consensus_probability": 0.39, "horizons": [...per-horizon breakdown...]}
```

---

### Scenario & Threat Analysis

#### `scenario_analysis(model_id, scenarios)`

Batch-runs a model over a list of scenario dicts and sorts by probability descending.

```python
results = await svc.scenario_analysis("mdl-abc", [
    {"label": "worst_case", "input": {"ttp_vector": 0.9}},
    {"label": "base_case",  "input": {"ttp_vector": 0.5}},
])
```

#### `forecast_event_probability(event_type, timeframe, indicators)`

Computes event probability from matching indicator confidence scores with horizon-based decay.

#### `threat_trajectory(threat_actor_id, period)`

Projects escalation trend (escalating / de_escalating / stable) from forecast history.

#### `threat_actor_profiling(threat_actor_id, include_trajectories=True)`

Builds a structured threat actor profile: risk band (LOW / MEDIUM / HIGH / CRITICAL), threat
level score, and linked forecasts, warnings, and projections. Optionally includes trajectory.

```python
profile = await svc.threat_actor_profiling("APT-42")
print(profile["threat_band"])   # "HIGH"
```

---

### Quality Assurance & Robustness

#### `adversarial_stress_test(model_id, input_data, n_samples=200, perturbation_scale=0.1)`

Monte Carlo perturbation test. Mutates numeric features by `N(0, scale × |value|)` and
records the empirical output distribution. Returns `robustness_score` (proportion of runs
that do not flip the decision) and worst-case example.

```python
stress = await svc.adversarial_stress_test("mdl-abc", {"ttps": 0.7}, n_samples=500)
print(stress["robustness_score"])   # 0.94
```

#### `counterfactual_analysis(model_id, input_data, decision_threshold=0.5)`

Identifies which ±σ feature mutations would flip the model's decision. Returns features
ranked by probability delta magnitude.

#### `detect_temporal_anomaly(indicator_id, observations, cusum_h=5.0, ewma_lambda=0.2)`

CUSUM + EWMA dual-detector on a scalar time series. Returns alarm indices, control limits,
and a boolean `anomaly_detected` flag. When an anomaly is detected the caller should trigger
`record_warning` with an appropriate severity.

```python
anom = await svc.detect_temporal_anomaly(
    "ind-beacon", observations=[0.3, 0.31, 0.29, 0.85, 0.9, 0.88]
)
if anom["anomaly_detected"]:
    print("CUSUM alarms at:", anom["cusum_alarm_indices"])
```

#### `check_concept_drift(model_id, current_feature_dist, psi_threshold=0.2)`

PSI-based concept-drift detection per feature. When `max_psi > psi_threshold` the model is
automatically marked `STALE` and an audit event is recorded. Integrate this into your
production inference pipeline to close the MLOps feedback loop.

```python
drift = await svc.check_concept_drift(
    "mdl-abc",
    current_feature_dist={"ttps": [0.4, 0.5, 0.9, 0.85, 0.88]},
)
if drift["drift_detected"]:
    await svc.train_model("mdl-abc", features=["ttps"])
```

---

### Indicators & Signals

#### `early_warning_indicators(domain)`

Returns top-10 indicators matching the domain string, sorted by confidence.

#### `indicator_correlation_matrix()`

Computes pairwise Pearson correlations between indicator confidence scores within each
scenario. Flags highly correlated pairs (`|r| > 0.8`) as potentially redundant.

---

### Compliance & Governance

#### `regulatory_compliance_scorecard()`

Scores each model against EU AI Act, NIST AI RMF, and ISO/IEC 42001. Dimensions: validation
reference, evidence reference, audit trail, human oversight, and training runs. Returns
per-model scores, compliance gaps, and a tenant-aggregate grade (A–F).

```python
card = await svc.regulatory_compliance_scorecard()
print(card["compliance_grade"])   # "B"
```

#### `compliance_validation()`

Checks that all deployed models have required governance documentation. Returns a list of
specific issues (MISSING_VALIDATION_REFERENCE, MISSING_EVIDENCE_REFERENCE,
DEPLOYED_WITHOUT_TRAINING).

#### `warning_escalation(warning_id, escalation_level)`

Escalates a warning to TACTICAL / OPERATIONAL / STRATEGIC / NATIONAL authority.

---

### Intelligence Sharing & Export

#### `intelligence_sharing(forecast_ids, recipients, classification)`

Shares forecasts with partner organisations under a supported classification. Generates a
sharing record per (forecast, recipient) pair with full audit trail.

#### `osint_collection_trigger(subject, source_types)`

Triggers simulated OSINT collection and returns coverage metadata per source type.

#### `export_forecasts(fmt="json")`

Exports forecast records. `fmt` supports `"json"` and `"csv"`. Returns a content fingerprint
for integrity verification.

---

### Dashboards & Monitoring

#### `prediction_dashboard()`

Per-model dashboard: status, training runs, latest accuracy, inference run count.

#### `prediction_analytics()`

Tenant aggregate: model counts, trained/deployed split, average accuracy, forecast/projection/
warning/scenario/indicator counts.

#### `prediction_accuracy_report(model_id, period)`

Accuracy history trend for a specific model. Reports `improving` or `stable` trend.

#### `health_check()`

Service health status and operational metrics. Always returns synchronously-safe data
suitable for liveness/readiness probes.

---

## UI Routes

| Path | Permission | Nav Group |
|---|---|---|
| `/intel-prediction/dashboard` | `intel_prediction:view` | Overview |
| `/intel-prediction/authorities` | `intel_prediction:authorities` | Governance |
| `/intel-prediction/workspaces` | `intel_prediction:workspaces` | Planning |
| `/intel-prediction/scenarios` | `intel_prediction:scenarios` | Planning |
| `/intel-prediction/indicators` | `intel_prediction:indicators` | Signals |
| `/intel-prediction/models` | `intel_prediction:models` | Models |
| `/intel-prediction/forecasts` | `intel_prediction:forecasts` | Forecasts |
| `/intel-prediction/projections` | `intel_prediction:projections` | Forecasts |
| `/intel-prediction/warnings` | `intel_prediction:warnings` | Warnings |
| `/intel-prediction/compliance` | `intel_prediction:compliance` | Governance |

---

## Guardrails

The capability denies:

- Unsupported automated decisions
- Hallucinated forecast scopes
- Privacy-bypass scopes
- Unapproved model deployment
- Autonomous warnings (without human approval)
- Autonomous recommendations (without human approval)
- Privileged agent actions without human approval recorded

Supported AI agent runtimes: `codex`, `claude_code`, `opencode`, `pi`.

---

## Interoperability

```apg
use intel_prediction;
use intel_correlation;   // knowledge graph — link forecasts to correlation entities
use intel_alerts;        // alerting pipeline — route warnings downstream
use intel_threats;       // threat registry — enrich threat actor profiles
```

Set `OLLAMA_BASE_URL` to route `prediction_run` inference through a locally hosted model via
`MLCapability.score()`. Set `INTEL_PREDICTION_*` environment variables for tenant-scoped
configuration.

---

## Requires

- `auth` — authority validation
- `audl` — audit log sink
- `ntfy` — notification dispatch
- `nlpc` — NLP correlation
- `grph` — knowledge graph traversal

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Policy rules and supported taxonomies
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 engineering enhancements
- `README.md` — Quick reference
