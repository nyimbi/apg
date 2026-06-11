# Predictive Intelligence — World-Class Improvement Catalogue

**Capability**: `intel_prediction` | **Version target**: 2.0  
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Bayesian Belief Network Integration

Replace the sigmoid point-estimate scorer with a Bayesian belief network that propagates
uncertainty through a directed acyclic graph of indicator nodes. Each node holds a prior and
likelihood table; new evidence performs conjugate posterior updates. This produces calibrated
probability intervals (lower/upper credible bounds) rather than bare scalars, and makes model
uncertainty explicit to downstream consumers.

**Files**: `service.py` → `prediction_run`, new `bayesian_net.py`

---

## 2. Temporal Anomaly Detection via CUSUM / EWMA

Add streaming anomaly detection on indicator time-series using Cumulative Sum (CUSUM) and
Exponentially Weighted Moving Average (EWMA) detectors. When a signal drifts beyond a
control threshold a `PredictionWarning` is automatically emitted through `record_warning`,
giving analysts near-real-time early warning without manual review.

**Files**: `service.py` → new `detect_temporal_anomaly`, `anomaly_detector.py`

---

## 3. Ensemble Model Voting

Support registering multiple prediction models under a shared scenario and aggregating their
outputs via weighted majority voting (soft or hard). The ensemble weight is tuned by each
model's historical accuracy stored in `_model_state["accuracy_history"]`. Result includes
per-model votes, ensemble probability, and a Brier-score decomposition for calibration audit.

**Files**: `service.py` → new `ensemble_predict`

---

## 4. Counterfactual Scenario Engine

Implement a counterfactual generator that systematically inverts individual input features and
re-runs `prediction_run` to identify which features, if altered, would flip the outcome below
a threshold. Returns a ranked list of `(feature, delta, counterfactual_probability)` tuples.
Invaluable for adversarial red-teaming and decision justification under GDPR Article 22.

**Files**: `service.py` → new `counterfactual_analysis`

---

## 5. Causal Graph Inference (do-calculus)

Add a causal discovery layer that infers do-calculus interventional distributions from
observational indicator data using the PC algorithm skeleton + V-structure orientation.
Exposes `do(X=x)` query capability so analysts can distinguish correlation from causation in
threat projections.

**Files**: `service.py` → new `causal_inference`, `causal_graph.py`

---

## 6. Federated Learning Aggregation

Enable privacy-preserving model updates from multiple tenants by implementing FedAvg
aggregation: each tenant trains a local delta, the service aggregates gradient vectors with
differential-privacy noise (Gaussian mechanism, ε-δ budget), and distributes the updated
global model without exposing raw training data. Critical for multi-agency intelligence
sharing agreements.

**Files**: `service.py` → new `federated_aggregate`, `federated.py`

---

## 7. Explainability Report (SHAP-style Attribution)

For each `prediction_run` output, compute feature importance scores via a model-agnostic
permutation approach (surrogate of SHAP TreeExplainer for tree-based; kernel SHAP for black
boxes). Expose per-feature signed attributions that sum to the log-odds output. Satisfies
XAI governance requirements and surfaces unexpected feature drivers for adversarial detection.

**Files**: `service.py` → new `explain_prediction`

---

## 8. Red Team Adversarial Stress Testing

Add a structured red-team harness that mutates input features by `±σ` increments (Monte Carlo
perturbation, N=1000 by default) and records the empirical distribution of output
probabilities. Produces a robustness score (proportion of perturbations that do not flip the
decision), worst-case adversarial example, and a stability band around the point estimate.

**Files**: `service.py` → new `adversarial_stress_test`

---

## 9. Knowledge Graph Event Linkage

Integrate with APG's `intel_correlation` capability to link `PredictionForecast` nodes to
correlation-graph entities via typed edges (`PREDICTS`, `TRIGGERED_BY`, `CONFIRMS`). Enables
graph traversal queries such as "which threat actors are implicated by high-confidence
forecasts?" and feeds downstream STIX-2.1 export.

**Files**: `service.py` → new `link_to_knowledge_graph`, composability bridge

---

## 10. Adaptive Re-training Triggers

Implement concept-drift detection using the Population Stability Index (PSI) between training
and inference feature distributions. When PSI > 0.2 the service marks the model
`STALE`, emits a `PredictionWarning` at severity `high`, and optionally triggers an
incremental `model_update` call. This closes the MLOps feedback loop without manual
intervention.

**Files**: `service.py` → new `check_concept_drift`, enhanced `model_update`

---

## 11. Structured Prediction Confidence Intervals

Replace single-scalar `confidence_score` with a `ConfidenceInterval` value object holding
`(point_estimate, lower_ci, upper_ci, ci_level)`. Propagate intervals through
`record_forecast` and `record_projection` arithmetic using error propagation rules. Consumers
receive honest uncertainty bounds rather than false precision.

**Files**: `models.py`, `views.py`, `service.py`

---

## 12. Streaming Event Bus Integration (Bytewax)

Replace the stub `validate_batch` method with a live Bytewax dataflow that ingests CloudEvents
from the `apg.intel.prediction.lifecycle` Kafka topic, applies windowed indicator aggregation
(tumbling 5-minute windows), and emits `PredictionWarning` records downstream. Makes the
entire prediction pipeline event-driven and horizontally scalable.

**Files**: `service.py`, new `dataflow.py`, `prediction_runtime.py`

---

## 13. Graph-of-Thought Reasoning Chain

Add a `reasoning_chain` field to `PredictionForecast` that records the step-by-step inference
path from raw indicators to final probability estimate as a directed reasoning graph. Each
node stores: evidence item, logical operator, and intermediate confidence. Supports audit trail
requirements and enables LLM-assisted chain-of-thought verification.

**Files**: `models.py`, `service.py` → enhanced `record_forecast`

---

## 14. Multi-Horizon Ensemble Forecast

Aggregate predictions across all registered `SUPPORTED_HORIZONS` for a given scenario, weight
each horizon's forecast by temporal proximity (inverse-horizon-days weighting), and return a
unified consensus forecast with per-horizon breakdown. Enables strategic planners to see the
same threat through near-, medium-, and long-term lenses simultaneously.

**Files**: `service.py` → new `multi_horizon_forecast`

---

## 15. Regulatory Compliance Scoring Matrix

Produce a structured compliance scorecard mapping each prediction model against regulatory
frameworks (EU AI Act risk tiers, NIST AI RMF, ISO/IEC 42001). Each dimension is scored 0–1
based on the presence of validation references, evidence, audit trails, human oversight records,
and explainability artefacts. Surfaces a `compliance_gap` list with specific remediation
actions, enabling automated governance reporting to oversight authorities.

**Files**: `service.py` → enhanced `compliance_validation`, new `compliance_matrix.py`
