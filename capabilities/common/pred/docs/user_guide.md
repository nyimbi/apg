# Predictive Analytics — User Guide

**Capability ID**: `pred` | **Domain**: `common` | **Version**: `1.1.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

---

## Overview

PRED is the APG capability for governed forecasting, scoring, scenario
simulation, and predictive model lifecycle management. All operations are
tenant-scoped, audit-logged, and subject to policy evaluation before state
changes are committed.

---

## Installation

```bash
pip install apg-common-pred
```

---

## Core Concepts

| Concept | Description |
|---|---|
| `PredictiveModel` | A registered, versioned model with algorithm, target, approval, and explainability state |
| `FeatureSet` | Named set of features with ETLP lineage refs and source system |
| `ForecastRun` | A horizon-bounded time-series forecast tied to a model |
| `ScoreRun` | A single entity inference result with impact classification |
| `ScenarioSimulation` | What-if projection from a baseline score with explicit assumptions |
| `DriftReport` | Measured deviation of model output from reference distribution |
| `PredictionAgentRecord` | AI agent registration for governed prediction workflows |
| `PredLifecycleBatchRecord` | Bytewax-validated mutation batch |
| `PredAuditEvent` | Immutable audit record for every state-changing operation |

---

## Provides

- `predictive_analytics`
- `forecasting`
- `prediction_agent_composition`
- `score_calibration`
- `monetary_impact_tracking`
- `champion_challenger_routing`
- `sla_monitoring`
- `lineage_graph`

## Requires

- `aicr`
- `mlcm`
- `etlp`
- `conf`

---

## UI Routes

| Path | Permission | Nav Group |
|---|---|---|
| `/pred/dashboard` | `pred:view` | Overview |
| `/pred/forecasts` | `pred:forecast` | Forecasts |
| `/pred/scores` | `pred:score` | Scoring |
| `/pred/features` | `pred:manage_models` | Scoring |
| `/pred/scenarios` | `pred:simulate` | Simulation |
| `/pred/models` | `pred:manage_models` | Models |
| `/pred/drift` | `pred:govern` | Models |
| `/pred/batch` | `pred:score` | Scoring |

---

## Service API Reference

### Model Operations

#### `register_model(model_id, tenant_id, name, owner, algorithm, target, ...)`
Registers a predictive model. Sets `status="pending_review"` when training
history is below threshold or feature metadata is missing. All parameters are
validated against the capability contract rules before persistence.

#### `approve_model(model_id, tenant_id, approver, explainability_ref=None)`
Approves a model for production scoring. Requires an approver identity.
Sets `status="pending_review"` when explainability evidence is absent.

#### `async train_model(tenant_id, model_id, training_data, hyperparams=None)`
Updates `training_history_points` and auto-approves when `>= 10` samples.
Emits a `model_trained` audit event.

#### `async model_evaluate(tenant_id, model_id, eval_data, metric="rmse")`
Evaluates the model against labelled eval data. Supports `rmse` and `mse`.

#### `async model_version(tenant_id, model_id, version_tag)`
Snapshots current model state as a named version. Stored as a derived model
record with `status="versioned"`.

#### `async model_compare(tenant_id, model_id_a, model_id_b, eval_data, metric="rmse")`
Compares two models on shared eval data. Returns winner and metric delta.

#### `async model_retrain(tenant_id, model_id, new_training_data)`
Triggers a retrain cycle. Delegates to `train_model` with updated samples.

#### `async auto_ml(tenant_id, candidate_algorithms, feature_set_id, training_data, owner)`
Registers, trains, and compares all candidate algorithms. Returns the best
model by RMSE.

#### `async calibrate_scores(tenant_id, model_id, calibration_pairs)`
Fits Platt scaling (logistic sigmoid) to convert raw scores to calibrated
probabilities. Uses 50-iteration gradient descent. Stores `(A, B)` parameters
per `(tenant_id, model_id)`.

**Input**:
```python
calibration_pairs = [
    {"predicted": 72.3, "actual": 1.0},
    {"predicted": 31.5, "actual": 0.0},
]
result = await service.calibrate_scores("tenant-a", "model-001", calibration_pairs)
# result["platt_A"], result["platt_B"] — fitted parameters
```

#### `async compute_confidence_decay(tenant_id, model_id, half_life_days=90.0, decay_threshold=0.5)`
Applies exponential decay to model confidence based on age since last training.
Half-life defaults to 90 days. Returns `decayed_below_threshold: bool` and
`recommended_action`.

```python
decay = await service.compute_confidence_decay("tenant-a", "model-001")
if decay["decayed_below_threshold"]:
    # Schedule retrain
```

---

### Feature Set Operations

#### `register_feature_set(feature_set_id, tenant_id, name, owner, feature_names, lineage_refs, source_system)`
Registers a named feature set. Sets `status="pending_review"` when lineage
refs are absent.

#### `async feature_importance(tenant_id, model_id, feature_set_id)`
Returns deterministic, normalised feature importance scores using hash-based
weighting.

---

### Scoring Operations

#### `score_entity(score_id, tenant_id, model_id, feature_set_id, entity_id, feature_values, environment, impact, explanation_ref)`
Scores a single entity. Blocks production scoring without an approved model,
feature scoring without lineage, and high-impact scoring without
explainability.

#### `async predict_real_time(tenant_id, model_id, feature_set_id, entity_id, feature_values)`
Single-entity real-time scoring wrapper. Returns score with `latency_mode:
"real_time"`.

#### `async predict_batch(tenant_id, model_id, feature_set_id, entities)`
Batch scoring over a list of entity dicts. Each must have an `id` key.

#### `async prediction_explain(tenant_id, score_id, method="shap_approx")`
Returns SHAP-style approximate attribution values per feature.

#### `async attach_monetary_outcome(tenant_id, score_id, amount, currency="KES")`
Attaches a `Decimal`-precision monetary outcome to a score. `amount` must be
passed as a string — floats are rejected for IFRS 13 compliance.

```python
# Correct — string preserves precision
await service.attach_monetary_outcome("tenant-a", "score-001", "125000.75", "KES")

# Wrong — float will raise ValueError
await service.attach_monetary_outcome("tenant-a", "score-001", 125000.75, "KES")  # ERROR
```

#### `async aggregate_monetary_impact(tenant_id, model_id, currency="KES")`
Returns the Decimal total of all monetary outcomes for a model. Uses
`ROUND_HALF_EVEN` throughout. Returns a string representation of the Decimal.

#### `async record_prediction_latency(tenant_id, score_id, latency_ms, sla_threshold_ms=100.0)`
Records inference latency for SLA tracking. Emits a `sla_breach_detected`
audit event when `latency_ms > sla_threshold_ms`.

#### `async compute_sla_report(tenant_id, model_id=None, sla_threshold_ms=100.0)`
Returns P50/P95/P99 latency percentiles and breach rate from stored latency
records. Filter to a specific model by providing `model_id`.

---

### Champion-Challenger Routing

#### `async register_champion_challenger(tenant_id, policy_id, model_id_champion, model_id_challenger, traffic_split_pct=10)`
Registers a routing policy. `traffic_split_pct` must be 1–49. Both models
must be registered under the same tenant.

#### `async route_score_request(tenant_id, policy_id, feature_set_id, entity_id, feature_values)`
Routes a scoring request under the active policy. Uses deterministic SHA-256
hashing of `entity_id` so the same entity always hits the same model arm.
Returns both routing decision and score.

```python
# Register policy: 10% of traffic to challenger
await service.register_champion_challenger(
    "tenant-a", "policy-001",
    model_id_champion="model-v1",
    model_id_challenger="model-v2",
    traffic_split_pct=10,
)

# Score entity — routing is deterministic per entity_id
result = await service.route_score_request(
    "tenant-a", "policy-001", "features-001", "entity-42",
    {"demand": 55, "season": 3},
)
print(result["routed_to"])  # "champion" or "challenger"
```

---

### Drift and Monitoring

#### `record_drift(report_id, tenant_id, model_id, metric_name, drift_score, threshold)`
Records a drift measurement. Status is `review_required` when
`drift_score > threshold`.

#### `async drift_detect(tenant_id, model_id, reference_scores, current_scores, threshold=0.1)`
Automatic mean-shift drift detection. Creates a `DriftReport` when drift
exceeds threshold.

#### `async stream_drift_window(tenant_id, model_id, reference_scores, current_scores, n_bins=10)`
Computes PSI and KL-divergence using equal-width binning.

| PSI Range | Band |
|---|---|
| < 0.1 | stable |
| 0.1 – 0.2 | warning |
| > 0.2 | critical |

```python
result = await service.stream_drift_window(
    "tenant-a", "model-001",
    reference_scores=[0.6, 0.7, 0.65, 0.72],
    current_scores=[0.4, 0.45, 0.38, 0.42],
)
print(result["psi"], result["stability_band"])
```

---

### Forecast Operations

#### `create_forecast(forecast_id, tenant_id, model_id, series_name, history_values, horizon_days)`
Creates a forecast run. Horizon > 365 days triggers `pending_review`.
History < 10 points triggers `pending_review`.

#### `async forecast_horizon(tenant_id, forecast_id)`
Returns the horizon, series name, and forecast values for an existing forecast.

#### `async confidence_interval(tenant_id, forecast_id, confidence=0.95)`
Returns symmetric confidence intervals (z=1.96 at 95%) around each forecast
step using a deterministic std approximation.

---

### Explainability and Attestation

#### `async register_explanation_attestation(tenant_id, score_id, model_version_id, method, feature_coverage_pct, attested_by)`
Creates a non-repudiable attestation for a score explanation. The attestation
hash covers `score_id + model_version_id + method + attested_by`. High-impact
scores require `feature_coverage_pct >= 80`.

#### `async verify_explanation_attestation(tenant_id, score_id)`
Recomputes the attestation hash and verifies it matches the stored value.
Returns `{valid: bool, tampered: bool}`.

```python
# Attest an explanation
attestation = await service.register_explanation_attestation(
    "tenant-a", "score-001", "model-v1", "shap_approx", 92.5, "data-scientist-1",
)

# Verify later (e.g. during audit)
verification = await service.verify_explanation_attestation("tenant-a", "score-001")
assert verification["valid"] is True
assert verification["tampered"] is False
```

---

### Lineage Graph

#### `async build_lineage_graph(tenant_id)`
Constructs a directed acyclic graph of data-flow dependencies for the tenant.
Edges: score → model, score → feature_set, forecast → model,
feature_set → lineage_refs. Returns adjacency list and node/edge counts.

#### `async trace_decision_lineage(tenant_id, score_id)`
BFS traversal from a score ID back to all root nodes (ETL sources with no
further upstream dependencies). Returns ordered lineage path and root nodes
for regulatory audit exhibits.

```python
graph = await service.build_lineage_graph("tenant-a")
trace = await service.trace_decision_lineage("tenant-a", "score-001")
print(trace["lineage_path"])   # [score-001, model-001, features-001, etlp://...]
print(trace["root_nodes"])     # ["etlp://pipelines/demand/features"]
```

---

### Governance and Agents

#### `register_prediction_agent(agent_id, tenant_id, name, runtime, role, scope, owner, purpose)`
Registers an AI prediction agent. Blocks unsupported runtimes/roles, missing
scope/owner/purpose, undisclosed machine contribution. Routes privileged roles
to pending human review.

#### `validate_pred_lifecycle_batch(tenant_id, event_stream, mutation_count, operation, batch_id)`
Validates a Bytewax lifecycle mutation batch. Rejects non-Bytewax streams.

---

### Dashboard and Analytics

#### `dashboard_summary(tenant_id)`
Returns model, feature, forecast, score, scenario, drift, agent, lifecycle,
and audit counts with pending-review breakdowns.

#### `async prediction_kpi_summary(tenant_id, period)`
Returns a KPI card with approval rate, avg score, and forecast counts.

#### `async prediction_analytics(tenant_id, days=30)`
Aggregated activity statistics for the tenant within a rolling window.

---

## Guardrails Summary

| Guardrail | Enforced On |
|---|---|
| Missing tenant context | All operations |
| Missing model owner / algorithm / target | `register_model` |
| Missing feature owner / names / source | `register_feature_set` |
| Production scoring without approved model | `score_entity` |
| High-impact scoring without explainability | `score_entity` |
| Scoring without feature lineage | `score_entity` |
| Forecast with horizon < 1 | `create_forecast` |
| Forecast horizon > 365 → pending_review | `create_forecast` |
| Missing drift metric or threshold | `record_drift` |
| Non-Bytewax stream in batch | `validate_pred_lifecycle_batch` |
| float amount in monetary outcome | `attach_monetary_outcome` |
| Champion split outside 1–49 | `register_champion_challenger` |
| High-impact attestation < 80% coverage | `register_explanation_attestation` |
| Cross-tenant data access | All `_require_*` lookups |

---

## Configuration

All keys are tenant-scoped. Set via `conf` capability or env vars prefixed
with `PRED_`.

| Key | Default | Description |
|---|---|---|
| `PRED_SLA_THRESHOLD_MS` | `100.0` | Default latency SLA threshold |
| `PRED_CONFIDENCE_HALF_LIFE_DAYS` | `90.0` | Model decay half-life |
| `PRED_CALIBRATION_LR` | `0.1` | Platt scaling learning rate |
| `PRED_DRIFT_PSI_CRITICAL` | `0.2` | PSI critical threshold |
| `PRED_ATTESTATION_MIN_COVERAGE_HIGH` | `80.0` | Min feature coverage for high-impact attestations |

---

## Interoperability

```apg
use pred;
```

PRED integrates with APG capabilities through the composition engine:

- `aicr` — AI agent credential and runtime management
- `mlcm` — Model lifecycle and compliance management
- `etlp` — ETL pipeline lineage references
- `conf` — Tenant-scoped configuration
- `auth` — Permission enforcement
- `audl` — External audit log forwarding
- `moni` — Metrics and alerting
- `cach` — Score and feature caching

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural improvement proposals
- `SPECIFICATION.md` — Complete functional scope
- `capability_contract.py` — Rules, UI routes, and adapter configuration
