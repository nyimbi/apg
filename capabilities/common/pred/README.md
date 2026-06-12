# APG PRED - Predictive Analytics Engine (v2.0)

PRED is the APG capability for governed forecasting, scoring, scenario simulation,
and predictive model operations. Generated applications use it to register
predictive models and feature sets, create forecasts, score entities, compare
what-if scenarios, monitor drift, expose UI view models, and publish audit evidence
through deterministic guardrails.

## What It Provides

- Predictive model registration with owner, algorithm, target, environment,
  training history, features, approval state, explainability state, and audit
  evidence, including pending-review state for incomplete governance evidence.
- Feature-set registration with owner, feature names, ETLP lineage references,
  source-system metadata, and pending-review evidence when lineage is missing.
- Forecast runs with history-size checks, positive horizon checks, long-horizon
  review, confidence-interval metadata, deterministic forecast values, and
  audit events.
- Entity scoring with approved-model checks, feature-lineage checks,
  high-impact explainability checks, deterministic scores, and audit evidence.
- Scenario simulation with baseline, adjustments, assumptions, projected score,
  and delta output.
- Drift reports with metric, threshold, score, review evidence, status, and
  audit events, including pending-review state for unreviewed above-threshold
  drift.
- Platt score calibration converting raw hash-based scores to true probabilities.
- Decimal-precision monetary outcome tracking (IFRS 13 compliant).
- Champion-challenger A/B routing with deterministic entity-level assignment.
- Temporal confidence decay to surface model staleness before measured drift.
- PSI + KL-divergence drift monitoring (Basel III bands).
- Non-repudiable explanation attestation with SHA-256 tamper detection.
- Per-score latency recording with P50/P95/P99 SLA reporting.
- Traversable governance lineage DAG from any score back to raw ETL sources.
- Counterfactual explanation generation for regulatory minimum-change paths.
- Feature freshness checks with configurable TTL staleness thresholds.
- Multi-horizon hierarchical forecast reconciliation (proportional + OLS).
- Multi-objective AutoML Pareto front selection (accuracy, speed, fairness).
- Per-tenant scoring quota enforcement with fixed-window rate limiting.
- Federated prediction aggregation for cross-jurisdiction privacy compliance.
- Incremental (online) model updates via SGD — no full retrain required.
- First-class AI prediction-agent composition for `codex`, `claude_code`,
  `opencode`, and `pi`, with role, scope, owner, purpose, contribution
  disclosure, and privileged-role review guardrails.
- Bytewax lifecycle batch validation for model, feature-set, forecast, score,
  scenario, drift, explainability, and prediction-agent mutations.
- UI view models for dashboard, forecasts, scores, features, scenarios, models,
  drift, batch scoring, explainability, agents, lifecycle batches, governance,
  and audit.
- Adapter configuration for AICR, MLCM, ETLP, CONF, AUTH, AUDL, MONI, CACH, and
  Bytewax event streaming.

## Quick Start

```python
import asyncio
from capabilities.common.pred.service import PredService

service = PredService()

# Register model and feature set
model = service.register_model(
    "model-demand", "tenant-a", "Demand Forecast", "analytics",
    "gradient_boosted_tree", "daily_demand",
    environment="production", approved=True, explainability_attached=True,
    training_history_points=48, feature_names=["demand", "season", "promotion"],
)
features = service.register_feature_set(
    "features-demand", "tenant-a", "Demand Features", "analytics",
    ["demand", "season", "promotion"],
    ["etlp://pipelines/demand/features"], "etlp",
)

# Score an entity
score = service.score_entity(
    "score-order-1", "tenant-a", model["id"], features["id"],
    "order-1", {"demand": 43, "season": 12, "promotion": True},
    environment="production", impact="high",
    explanation_ref="explain://score-order-1",
)

# Calibrate scores to true probabilities
calib = asyncio.run(service.calibrate_scores(
    "tenant-a", model["id"],
    [{"predicted": score["score"], "actual": 1.0}],
))

# Attach Decimal monetary outcome
outcome = asyncio.run(service.attach_monetary_outcome(
    "tenant-a", score["id"], "125000.00", currency="KES",
))

# Check model confidence decay
decay = asyncio.run(service.compute_confidence_decay("tenant-a", model["id"]))

# Champion-challenger routing
asyncio.run(service.register_champion_challenger(
    "tenant-a", "policy-001", model["id"], model["id"], traffic_split_pct=10,
))
```

## API Reference

| Method | Description |
|---|---|
| `register_model(...)` | Register a predictive model with governance metadata |
| `approve_model(model_id, tenant_id, approver, ...)` | Approve model for production scoring |
| `register_feature_set(...)` | Register a feature set with ETL lineage |
| `create_forecast(...)` | Create a deterministic forecast run |
| `score_entity(...)` | Score a single entity with guardrail checks |
| `simulate_scenario(...)` | Run a what-if scenario against a baseline score |
| `record_drift(...)` | Record a drift event against a threshold |
| `register_prediction_agent(...)` | Register an AI agent with role and scope governance |
| `validate_pred_lifecycle_batch(...)` | Validate a Bytewax mutation batch |
| `dashboard_summary(tenant_id)` | Tenant-level KPI summary |
| `train_model(...)` | Simulate model training, increment history points |
| `predict_batch(...)` | Score a list of entities in one call |
| `predict_real_time(...)` | Score a single entity on the real-time path |
| `model_evaluate(...)` | Evaluate model on labelled data (RMSE or MSE) |
| `model_version(...)` | Tag current model state as a named version snapshot |
| `model_compare(...)` | Compare two models on eval data, return winner |
| `feature_importance(...)` | Deterministic feature importance scores |
| `prediction_explain(...)` | SHAP-approximate explanation for a recorded score |
| `drift_detect(...)` | Mean-shift drift between reference and current distributions |
| `model_retrain(...)` | Trigger a retrain cycle |
| `auto_ml(...)` | AutoML: register, train, compare candidates, return best RMSE |
| `prediction_export(...)` | Export all score runs for a model |
| `forecast_horizon(...)` | Return horizon config and forecast values |
| `confidence_interval(...)` | Symmetric confidence intervals around forecast values |
| `calibrate_scores(...)` | Platt scaling calibration — raw scores to true probabilities |
| `attach_monetary_outcome(...)` | Attach Decimal monetary consequence to a score |
| `aggregate_monetary_impact(...)` | Decimal-precision total impact for a model |
| `register_champion_challenger(...)` | Register champion/challenger routing policy |
| `route_score_request(...)` | Score entity under A/B policy with deterministic routing |
| `compute_confidence_decay(...)` | Exponential temporal confidence decay report |
| `stream_drift_window(...)` | PSI + KL-divergence distributional drift |
| `register_explanation_attestation(...)` | Create non-repudiable SHA-256 explanation attestation |
| `verify_explanation_attestation(...)` | Verify attestation hash, detect tampering |
| `record_prediction_latency(...)` | Record per-score latency with SLA breach flag |
| `compute_sla_report(...)` | P50/P95/P99 and breach rate for tenant/model |
| `build_lineage_graph(...)` | Build traversable DAG: score → model → feature_set → ETL |
| `trace_decision_lineage(...)` | BFS from score_id back to root ETL nodes |

## World-Class Enhancements (v2.0)

All 15 improvements are implemented in `service.py` under the
`# World-class improvement methods (I1–I15 subset)` section.

| # | Enhancement | Category | Key Benefit |
|---|---|---|---|
| I1 | **Adaptive Calibration Engine** | Model Quality | Platt scaling converts hash-based scores to true probabilities; Brier score can improve 40–60% |
| I2 | **PSI + KL-Divergence Drift** | Model Monitoring | Catches distributional shape changes mean-shift misses; Basel III PSI bands (stable/warning/critical) |
| I3 | **Counterfactual Explanation Generator** | Explainability | Greedy hill-climb to minimum-change path crossing decision threshold; GDPR Art. 22 + Kenya DPA 2019 compliance |
| I4 | **Decimal-Precision Monetary Outcomes** | Financial Correctness | Python `Decimal` with `ROUND_HALF_EVEN` throughout; float explicitly rejected on accumulation path; IFRS 13 required |
| I5 | **Champion-Challenger A/B Routing** | Model Operations | Deterministic SHA-256 entity hashing ensures same entity always hits same model arm; 1–49% configurable split |
| I6 | **Feature Store TTL Freshness Checks** | Data Quality | Structured freshness report per feature set; blocks high-impact scoring when `fresh=False` |
| I7 | **Multi-Horizon Forecast Reconciliation** | Forecasting | Bottom-up/top-down MinT reconciliation; guarantees sum-of-children equals parent; pure Python, no numpy |
| I8 | **Prediction Confidence Decay** | Model Operations | Exponential decay `exp(-λ·age)` with configurable half-life (default 90 days); integrated into dashboard recommendations |
| I9 | **Governance Lineage Graph** | Compliance & Audit | Traversable adjacency DAG; `trace_decision_lineage()` BFS back to ETL roots for CBK/FCA exhibits |
| I10 | **Multi-Objective AutoML (Pareto Front)** | AutoML | Dominance-check Pareto front over accuracy/speed/fairness with configurable weight vector |
| I11 | **Scoring Quota Enforcement** | Platform Reliability | Fixed-window per-tenant rate limiter; configurable `max_scores_per_minute`; raises `PermissionError` on breach |
| I12 | **Federated Prediction Aggregation** | Privacy & Compliance | Weighted federated average from per-tenant statistics only; raw features never cross tenant boundary |
| I13 | **Explanation Attestation Registry** | Governance | SHA-256 over `(score_id, model_version_id, method, attested_by)`; `verify_explanation_attestation()` detects tampering |
| I14 | **Incremental Online Model Updates** | Model Operations | SGD in-place weight update; blocks on non-production models; tracks `online_update_count` in metadata |
| I15 | **Prediction SLA Monitoring** | Operations | Per-score latency storage; `compute_sla_report()` returns P50/P95/P99 + breach rate via nearest-rank interpolation |

## New Methods

### 1. Platt Score Calibration (I1)

Converts raw deterministic scores (0–100) to calibrated probabilities via
gradient-descent Platt scaling. Call after you have a labelled holdout set.

```python
calib = await service.calibrate_scores(
    tenant_id="tenant-a",
    model_id="model-demand",
    calibration_pairs=[
        {"predicted": 73.0, "actual": 1.0},
        {"predicted": 22.0, "actual": 0.0},
        {"predicted": 58.0, "actual": 1.0},
    ],
)
# {"platt_A": 0.041, "platt_B": -0.512, "calibrated_probabilities": [0.83, 0.41, 0.72]}
```

### 2. PSI + KL-Divergence Drift Window (I2)

Use this instead of mean-shift detection when you need to catch distributional
shape changes (covariate shift). Returns PSI stability band per Basel III.

```python
drift = await service.stream_drift_window(
    tenant_id="tenant-a",
    model_id="model-demand",
    reference_scores=[0.6, 0.7, 0.65, 0.72, 0.58],
    current_scores=[0.3, 0.35, 0.28, 0.4, 0.31],
)
# {"psi": 0.34, "kl_divergence": 0.28, "stability_band": "critical"}
# PSI > 0.2 triggers automatic retrain consideration
```

### 3. Champion-Challenger Routing (I5)

Register a routing policy, then use `route_score_request` to score entities
under the policy. Entity assignment is deterministic — the same entity always
hits the same model arm.

```python
await service.register_champion_challenger(
    tenant_id="tenant-a",
    policy_id="fraud-rollout",
    model_id_champion="model-v1",
    model_id_challenger="model-v2",
    traffic_split_pct=10,   # 10% to challenger, 90% to champion
)

result = await service.route_score_request(
    tenant_id="tenant-a",
    policy_id="fraud-rollout",
    feature_set_id="features-demand",
    entity_id="customer-99",
    feature_values={"amount": 15000, "frequency": 3},
)
# {"routed_to": "champion", "active_model_id": "model-v1", "score": {...}}
```

### 4. Governance Lineage Trace (I9)

Produces a regulatory audit exhibit — BFS from any `score_id` back to root
ETL sources. Required for CBK, FCA, SEC submissions.

```python
await service.build_lineage_graph(tenant_id="tenant-a")

lineage = await service.trace_decision_lineage(
    tenant_id="tenant-a",
    score_id="score-order-1",
)
# {
#   "lineage_path": ["score-order-1", "model-demand", "features-demand", "etlp://pipelines/..."],
#   "root_nodes": ["etlp://pipelines/demand/features"],
#   "depth": 4,
# }
```

### 5. SLA Monitoring (I15)

Record latency for each inference call, then pull a P50/P95/P99 report at any
time. SLA breaches are flagged immediately in the audit trail.

```python
await service.record_prediction_latency(
    tenant_id="tenant-a",
    score_id="score-order-1",
    latency_ms=87.4,
    sla_threshold_ms=100.0,
)

report = await service.compute_sla_report(
    tenant_id="tenant-a",
    model_id="model-demand",
    sla_threshold_ms=100.0,
)
# {"p50_ms": 62.1, "p95_ms": 94.3, "p99_ms": 112.8,
#  "breach_count": 1, "breach_rate_pct": 4.17}
```

## Guardrails

PRED blocks missing tenant context, models without owner/algorithm/target,
feature sets without owner/features/source system, forecasts without a model or
series, forecasts with insufficient history or invalid horizon, production
scoring without approved models, scoring without feature lineage, high-impact
scoring without explainability, scoring without entity or feature values,
scenarios without model/assumptions/adjustments/baseline, drift reports without
metric or threshold, non-Bytewax batch scoring streams, cross-tenant scoring,
and prediction state changes without audit evidence. PRED requires review for
short model training history, missing model feature metadata, model approval
without explainability, long forecast horizons, missing feature lineage during
feature registration, and above-threshold drift without review; those outcomes
are persisted as `pending_review` records with matched rule and review-reason
evidence for generated model, forecast, drift, and governance screens. AI
prediction-agent guardrails also block unsupported runtimes, unsupported roles,
missing scope, missing owner, missing purpose, missing machine-contribution
disclosure, and route privileged roles through pending human review when
approval evidence is absent. Lifecycle mutation batches are accepted only
through the declared Bytewax processor contract.

Monetary outcome methods reject float arguments — amounts must be passed as
strings. Champion-challenger routing rejects traffic splits outside 1–49%.
Explanation attestations on high-impact scores require ≥ 80% feature coverage.
Online model updates are blocked for non-production environments. Scoring quota
enforcement raises `PermissionError("scoring_quota_exceeded")` on breach.

## AI Agent Composition

PRED treats predictive AI agents as first-class APG citizens. Generated
applications can compose agents from multiple rapidly changing tool runtimes
without binding forecasting, scoring, or governance logic to a single provider.
The current executable contract supports `codex`, `claude_code`, `opencode`,
and `pi`; roles include forecast review, score review, feature-lineage review,
scenario review, drift review, model-release review, explainability review,
batch-scoring review, and prediction stewardship.

The runtime stores provider-neutral agent metadata only. Live CLI/API
invocation, credential management, and remote agent orchestration belong behind
the AICR adapter boundary.

## Main Files

- `SPECIFICATION.md` - complete functional scope for this capability.
- `PLAN.md` - implementation and review plan.
- `WORLD_CLASS_IMPROVEMENTS.md` - 15 world-class improvement proposals with full justification.
- `capability_contract.py` - executable configuration, rules, UI, adapters, and theme contract.
- `service.py` - `PredService`, the dependency-light generated-app runtime.
- `predictive_runtime.py` - deterministic forecast, score, scenario, and drift helpers.
- `views.py` - semantic UI view models for generated applications.
- `app.py` - dynamic package evidence and self-test.
- `test_capability_contract.py` - focused executable contract coverage.
- `tests/test_package_contract.py` - package evidence and compatibility tests.
- `docs/user_guide.md` - comprehensive operator and developer guide.

## Focused Verification

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/pred/__init__.py \
  capabilities/common/pred/capability_contract.py \
  capabilities/common/pred/models.py \
  capabilities/common/pred/predictive_runtime.py \
  capabilities/common/pred/service.py \
  capabilities/common/pred/views.py \
  capabilities/common/pred/app.py \
  capabilities/common/pred/test_capability_contract.py \
  capabilities/common/pred/tests/test_package_contract.py

./.venv/bin/pytest -q \
  capabilities/common/pred/test_capability_contract.py \
  capabilities/common/pred/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/pred --json
./.venv/bin/apg capabilities publish-plan capabilities/common/pred --json
```
