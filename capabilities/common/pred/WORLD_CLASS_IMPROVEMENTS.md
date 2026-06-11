# PRED - World Class Improvements

**Capability**: Prediction Engine (`pred`) | **Domain**: `common`
**Author**: Nyimbi Odero | **Copyright**: © 2025 Datacraft

---

### I1. Adaptive Calibration Engine
**Category**: Model Quality | **Justification**: Raw scores from deterministic hash-based models are systematically biased; calibration converts raw scores to true probabilities (Brier score can improve 40-60%). Without it, high-impact financial decisions built on `score_entity` will be miscalibrated — same failure mode as Facebook's ad CTR prediction pre-2014. | **Implementation**: Add `async calibrate_scores()` that fits a Platt scaling sigmoid (logistic regression over predicted vs. actual on a holdout set), stores `(A, B)` parameters per model, and wraps every `deterministic_score` output with `sigmoid(A * raw + B)`. Cache calibration curves in `_calibrations` dict keyed by `(tenant_id, model_id)`. | **Competitor**: Google Vizier / Vertex AI has per-model calibration metadata; SageMaker exposes calibration as a first-class post-training step.

---

### I2. Streaming Window Drift (PSI + KL-Divergence)
**Category**: Model Monitoring | **Justification**: Current drift detection is a mean-shift scalar — a one-dimensional signal blind to distributional shape changes. Population Stability Index and KL-divergence catch covariate shift that mean-shift misses entirely, which is the primary failure mode of credit-scoring models in production. PSI > 0.2 is the industry standard for automatic retrain triggers. | **Implementation**: Add `async stream_drift_window()` that buckets reference and current score arrays into 10 equal-width bins, computes PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct)) and KL = sum(p * ln(p/q)), returns both metrics with band classification (stable/warning/critical) as per Basel III model risk standards. | **Competitor**: WhyLabs and Evidently AI both expose PSI + KL as the default drift metrics; Arize adds EMD (Earth Mover's Distance) on top.

---

### I3. Counterfactual Explanation Generator
**Category**: Explainability | **Justification**: SHAP attribution tells you what contributed most — it does not tell you what needs to change to flip the prediction. For loan denial / insurance exclusion use-cases this is legally required under GDPR Article 22 and Kenya's Data Protection Act 2019 Section 32. Counterfactuals are the only explanation method that directly answers "what would have to change?" | **Implementation**: Add `async generate_counterfactual()` that runs a greedy hill-climb over the feature space: for each feature, compute the score delta from incrementing / decrementing by one standard deviation (estimated from the feature set lineage), return the minimum-change path to cross the decision threshold stored on the model. | **Competitor**: IBM AI Fairness 360 + DiCE (Diverse Counterfactual Explanations by Microsoft Research) are the reference implementations. Seldon Core exposes this via its explainability microservice API.

---

### I4. Monetary Outcome Tracking with Decimal Precision
**Category**: Financial Correctness | **Justification**: Prediction engines that drive pricing, credit limits, or fraud thresholds accumulate rounding error when using `float`. At scale (10M decisions/day), a 0.001 rounding error compounds to material financial misstatement. IEEE 754 float is explicitly prohibited in financial audit contexts under IFRS 13. | **Implementation**: Add `async attach_monetary_outcome()` that stores a `Decimal` value (via Python's `decimal` stdlib, `prec=10`) as the downstream monetary consequence of a score decision (e.g. approved credit line, fraud loss prevented). Store in `_monetary_outcomes: dict[str, Decimal]` keyed by `score_id`. Expose `async aggregate_monetary_impact()` returning `Decimal` sum per tenant+model+period using `decimal.ROUND_HALF_EVEN`. | **Competitor**: Stripe ML uses `Decimal` throughout their fraud model outcome tracking. Bloomberg's model risk platform enforces Decimal for all financial consequence fields.

---

### I5. Champion-Challenger A/B Routing
**Category**: Model Operations | **Justification**: Shadow deployment (deploy model B, route N% of live traffic to it without acting on its output) is the only safe way to validate production models before full cutover. Without it, every model swap is a big-bang deployment — the primary cause of model-induced production incidents at Netflix and Uber. | **Implementation**: Add `async register_champion_challenger()` that stores a routing policy `{model_id_champion, model_id_challenger, traffic_split_pct}` in `_routing_policies`. Add `async route_score_request()` that uses a deterministic hash of `entity_id` to assign the request to champion or challenger (ensuring the same entity always hits the same model for consistency), scores both, returns both results with routing decision. | **Competitor**: SageMaker Endpoints have native champion-challenger routing. Databricks MLflow has `MlflowClient.transition_model_version_stage()` with shadow/canary stages.

---

### I6. Feature Store Integration with TTL-Based Staleness Checks
**Category**: Data Quality | **Justification**: Stale features are the #1 cause of training-serving skew. If a feature was computed 72h ago but the model was trained on hourly refreshes, the model operates in a distributional regime it was never trained on. Google's rules of ML (Rule #5) explicitly flags feature freshness as a reliability requirement. | **Implementation**: Add `async check_feature_freshness()` that accepts a `feature_set_id`, a `computed_at` timestamp, and a `max_staleness_seconds` threshold; computes age against `utc_now()`; returns a structured freshness report with `{fresh: bool, age_seconds: int, staleness_pct_of_ttl: float}`. Block `score_entity` calls when `fresh=False` on high-impact models (impact == "high"). | **Competitor**: Feast (open-source feature store) exposes `max_age` on every FeatureView. Tecton enforces SLA-based feature freshness with automatic alerting.

---

### I7. Multi-Horizon Forecast Reconciliation (Bottom-Up / Top-Down)
**Category**: Forecasting | **Justification**: Hierarchical time series (e.g. SKU -> Category -> National) are inconsistent when forecast independently — a fundamental error in demand planning. MinT (Minimum Trace) reconciliation guarantees that sum-of-children equals parent forecast, which is required by S&OP processes in manufacturing/retail. Inconsistent hierarchical forecasts lead to inventory misallocation. | **Implementation**: Add `async reconcile_forecast_hierarchy()` that accepts a dict of `{level_name: [forecast_id, ...]}` and applies proportional (top-down) or OLS (bottom-up) reconciliation. Uses matrix operations expressed as pure Python lists-of-lists (no numpy dependency). Returns reconciled forecast values per node with consistency proof (sum check). | **Competitor**: Nixtla's `hierarchicalforecast` library and AWS Forecast's hierarchical reconciliation both use MinT. darts (Unit8) has `BottomUp` and `TopDown` reconciliators as first-class ensemble strategies.

---

### I8. Prediction Confidence Decay Model
**Category**: Model Operations | **Justification**: A model trained in Q1 is less reliable in Q4 due to concept drift, even if no measured drift has been detected yet. Temporal confidence decay is the statistically correct representation of this uncertainty — it converts model age into a confidence penalty and forces the system to surface models that need review before they silently degrade. | **Implementation**: Add `async compute_confidence_decay()` that accepts `model_id`, fetches `updated_at` (last training timestamp), and applies exponential decay: `confidence = exp(-lambda * days_since_training)` where `lambda` defaults to `ln(0.5) / 90` (half-life of 90 days). Store and return `{model_id, confidence, days_since_training, half_life_days, decayed_below_threshold: bool}`. Integrate into `dashboard_summary` as `model_confidence_health`. | **Competitor**: Neptune.ai tracks model age and surfaces staleness warnings. Weights & Biases (wandb) has model registry with `last_trained` metadata and alert thresholds.

---

### I9. Governance Lineage Graph
**Category**: Compliance & Audit | **Justification**: Regulators (CBK, FCA, SEC) require complete lineage from raw data → feature → model → score → decision. A flat audit log is insufficient — you need a traversable directed acyclic graph from any decision back to source data. PRED currently has audit events but no traversable lineage graph. | **Implementation**: Add `async build_lineage_graph()` that traverses `_audit_events`, `_scores`, `_models`, and `_feature_sets` to construct a `dict[str, list[str]]` adjacency list keyed by entity ID. Add `async trace_decision_lineage()` that does a BFS from a `score_id` back to root `feature_set` lineage refs, returning an ordered path list. Persist in `_lineage_graph: dict[str, set[str]]`. | **Competitor**: DataHub (LinkedIn open-source) and Apache Atlas expose lineage graphs via REST APIs. Alation's data catalog has automated lineage stitching with UI traversal.

---

### I10. Multi-Objective Model Selection (Pareto Front)
**Category**: AutoML | **Justification**: Current `auto_ml()` selects the model with the lowest RMSE — a single-objective optimization that ignores inference latency, model size, fairness metrics, and regulatory compliance. Production ML requires Pareto-optimal trade-offs: a model 5% less accurate but 10x faster and provably fair is the better choice for real-time scoring at scale. | **Implementation**: Add `async auto_ml_pareto()` that evaluates each candidate on `[accuracy, inference_time_proxy, fairness_score]`, computes the Pareto front using a pure-Python dominance check (`O(n^2)` is fine for < 50 candidates), and returns the full Pareto front ranked by a configurable weight vector `(w_accuracy, w_speed, w_fairness)`. Mark non-dominated models with `pareto_optimal: true`. | **Competitor**: Google Vertex AI Vizier does multi-objective hyperparameter optimization with Pareto front visualization. H2O AutoML exposes accuracy/latency trade-off curves as a first-class output.

---

### I11. Prediction Request Rate Limiting and Quota Enforcement
**Category**: Platform Reliability | **Justification**: Without per-tenant rate limiting, a single runaway batch job can exhaust scoring capacity and create denial-of-service for other tenants. This is a Tier-1 SRE concern — every major ML platform (SageMaker, Vertex AI, Azure ML) enforces per-tenant QPS quotas. PRED's `predict_batch()` currently has no throttle. | **Implementation**: Add `async enforce_scoring_quota()` that maintains a `_quota_registry: dict[str, dict]` tracking `{tenant_id: {window_start, count, limit}}`. Apply a fixed-window rate limiter (configurable `max_scores_per_minute`, default 10_000). Raise `PermissionError("scoring_quota_exceeded")` when limit is breached. Expose `async get_quota_status()` returning current usage, limit, and reset time. Integrate as a pre-check in `score_entity` and `predict_batch`. | **Competitor**: Azure ML endpoints have per-deployment RPM quotas. AWS SageMaker enforces invocation throttling at the endpoint level with `TooManyRequestsException`.

---

### I12. Federated Prediction Aggregation
**Category**: Privacy & Compliance | **Justification**: Cross-border data residency regulations (GDPR, Kenya DPA 2019) prohibit raw feature data from leaving jurisdiction. Federated averaging allows training/inference aggregation across tenants without raw data exchange — a critical architectural requirement for multi-region deployments. | **Implementation**: Add `async aggregate_federated_predictions()` that accepts a list of `{tenant_id, model_id, local_score_mean, local_sample_count}` records, computes the weighted federated average `sum(mean_i * n_i) / sum(n_i)`, and stores the result as a synthetic `ForecastRun` with `series_name="federated_aggregate"`. Validate that no raw feature values are passed — only statistics. | **Competitor**: Google's TensorFlow Federated and PySyft (OpenMined) implement FL averaging. Apple's on-device ML uses federated aggregation to train models without raw data leaving device.

---

### I13. Explainability Attestation Registry
**Category**: Governance | **Justification**: Attaching an `explanation_ref` string to a score is necessary but not sufficient — regulators need to verify that the explanation is current (produced by the same model version), complete (covers all features), and non-repudiable (signed by an authorized reviewer). Without attestation, explainability claims are trivially forgeable. | **Implementation**: Add `async register_explanation_attestation()` that stores an `ExplanationAttestation` record containing `{score_id, model_version_id, method, feature_coverage_pct, attested_by, attested_at, attestation_hash}`. The hash is `sha256(score_id + model_version_id + method + attested_by)`. Add `async verify_explanation_attestation()` that recomputes the hash and returns `{valid: bool, tampered: bool}`. | **Competitor**: IBM OpenScale (now Watson OpenScale) has model explanation attestation with audit trails. Arthur AI stores explanation metadata with model card linkage.

---

### I14. Incremental Learning Support (Online Model Updates)
**Category**: Model Operations | **Justification**: Batch retraining has a latency of hours-to-days — unacceptable for real-time fraud detection or dynamic pricing where concept drift is measured in minutes. Incremental/online learning updates model parameters in-place from a stream of observations without full retrain. | **Implementation**: Add `async update_model_online()` that accepts `(tenant_id, model_id, observation: dict, label: float)` and applies a stochastic gradient step to an in-memory weight vector stored in `model.metadata["online_weights"]`. Use a simple SGD update: `w = w - lr * (predicted - label) * feature_vector` with configurable `learning_rate` (default `0.01`). Track `online_update_count` in model metadata. Block online updates on models with `environment != "production"` to enforce governance. | **Competitor**: River (formerly Creme) is the Python reference library for online/incremental ML. Vowpal Wabbit is the original production online learning system used by Microsoft for Bing ads.

---

### I15. Prediction SLA Monitoring and Breach Reporting
**Category**: Operations | **Justification**: Production prediction services have contractual latency SLAs (e.g., P99 < 100ms). Without SLA tracking, breaches are invisible until customers complain. The cost of an undetected SLA breach in a real-time credit scoring API is both financial (SLA credits) and reputational. PRED has no timing infrastructure. | **Implementation**: Add `async record_prediction_latency()` that stores `{score_id, latency_ms: float, sla_threshold_ms: float, breached: bool}` in `_latency_records: dict[str, dict]`. Add `async compute_sla_report()` returning `{p50, p95, p99, breach_count, breach_rate_pct, sla_threshold_ms}` computed from a sorted list of latency values using index-based percentile interpolation. Integrate as an optional decorator pattern in `score_entity`. | **Competitor**: Grafana + Prometheus is the standard SLA monitoring stack for ML serving. Datadog ML Monitoring exposes prediction latency percentiles with configurable alert thresholds.
