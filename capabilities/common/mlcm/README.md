# APG MLCM - AI Model Lifecycle Management

MLCM is the APG capability for governed AI model operations. It gives generated
applications a tenant-scoped model registry, version lineage, evaluation gates,
promotion approvals, deployment controls, drift response, rollback, retirement,
audit evidence, UI metadata, and a Bytewax event-stream adapter contract.

## What It Provides

- Model registry with owner, problem type, risk level, status, tags, and tenant
  boundaries.
- Version management with artifact URI, stage, model card, training data,
  baseline lineage, evaluation score, promotion state, matched guardrail rules,
  and pending-review state for incomplete release evidence.
- Evaluation runs with baseline references, metrics, evidence references, and
  deterministic pass/fail or pending-review status against configurable
  thresholds, fairness review, and explainability review.
- Promotion gates for dev, staging, and production, including production
  approval enforcement and low-score blocking.
- Deployment targets and deployment records for APG-generated applications,
  including target state, replicas, canary percentage, and approval metadata.
- Drift signals, drift review, dashboard summaries, and deployment blocking
  when unresolved drift exists.
- Rollback records and model retirement after impact review and deployment drain.
- First-class model lifecycle agents for Codex, Claude Code, OpenCode, and Pi
  reviewers, including role, scope, owner, purpose, contribution disclosure, and
  privileged approval status.
- Bytewax lifecycle batch validation for model, version, evaluation, promotion,
  deployment, drift, rollback, retirement, and model-lifecycle-agent mutations.
- Rule-engine metadata, UI route metadata, theme tokens, and generated-app
  semantic package evidence.
- Durable review evidence for review-required versions, evaluations,
  privileged model lifecycle agents, and denied lifecycle batches.
- Adapter configuration for AICR, AUTH, AUDL, MONI, file artifacts, and Bytewax
  event streaming.
- A/B testing between deployment pairs with traffic-split configuration.
- Hyperparameter tuning records with random, grid, and Bayesian search support.
- Model export to ONNX, TorchScript, SavedModel, MLflow, and HuggingFace formats.
- Performance degradation alerting with configurable delta thresholds.
- Comprehensive analytics aggregation across models, deployments, drift, and training.
- Training job submission and status tracking per model.
- Model-level concept/performance drift detection distinct from data drift signals.
- Bias auditing across protected attributes with disparity scoring.
- Model explanation (SHAP, LIME, integrated gradients, attention) with feature importance.
- Structural model validation with per-field pass/fail reporting.

## Main Files

- `SPECIFICATION.md` - complete functional scope for this packet.
- `PLAN.md` - implementation and review plan.
- `capability_contract.py` - executable configuration, rules, UI, adapters, and
  theme contract.
- `models.py` - dependency-light domain records.
- `service.py` - `MlcmService`, the generated-app runtime facade.
- `api.py` - API-shaped helper calls over the service.
- `views.py` - semantic UI view models.
- `app.py` - dynamic package evidence and self-test.

## Generated-App Usage

```python
from capabilities.common.mlcm.service import MlcmService

service = MlcmService()
model = service.register_model(
	"fraud-risk",
	"tenant-a",
	"Fraud Risk Model",
	"risk-ai",
	"classification",
	risk_level="high",
)
version = service.create_version(
	"fraud-risk-v1",
	"tenant-a",
	model["id"],
	"1.0.0",
	"s3://models/fraud-risk/1.0.0/model.pkl",
	model_card={
		"purpose": "score payment fraud risk",
		"owner": "risk-ai",
		"training_data": "payments-2026-q1",
		"limitations": "not calibrated for card-present transactions",
	},
	training_data_ref="dataset:payments-2026-q1",
	baseline_ref="baseline:fraud-risk-2026-q1",
)
service.record_evaluation(
	"eval-fraud-v1",
	"tenant-a",
	version["id"],
	0.91,
	"baseline:fraud-risk-2026-q1",
	metrics={"auc": 0.94, "precision": 0.87},
	evidence_refs=["report:eval-fraud-v1"],
	fairness_review_recorded=True,
	explainability_recorded=True,
)
service.request_promotion(
	"promote-fraud-v1-prod",
	"tenant-a",
	version["id"],
	"production",
	"risk-ai",
	approval_recorded=True,
	approval_ref="approval:ml-gate-42",
)
target = service.create_target(
	"risk-prod",
	"tenant-a",
	"Risk Production Endpoint",
	"production",
	"aicr-python",
	"risk-platform",
)
service.deploy_model("deploy-fraud-v1", "tenant-a", version["id"], target["id"])
service.register_model_lifecycle_agent(
	"model-steward",
	"tenant-a",
	"Model Steward",
	"codex",
	"model_steward",
	"fraud model registry",
	"risk-ai",
	"Keep release evidence complete.",
)
service.validate_mlcm_lifecycle_batch(
	"tenant-a",
	"bytewax",
	4,
	"model_lifecycle_agent_batch",
)
```

## World-Class Enhancements (v2.0)

These 15 improvements address the gap between the v1 synchronous, in-memory
implementation and production-grade governed AI operations.

1. **Native Async Service Layer** — async-native write methods
   (`async_register_model`, `async_record_evaluation`, etc.) backed by an async
   lock. Sync methods remain as backward-compatible wrappers. Eliminates thread
   executor overhead in FastAPI/LangGraph pipelines.

2. **Pluggable Persistent Store Adapter** — `MlcmStoreAdapter` ABC with
   `InMemoryStore` (tests) and `PostgresStore` (asyncpg) implementations.
   `MlcmService(store=PostgresStore(...))` wires PostgreSQL persistence; no
   code changes elsewhere. Satisfies the mandatory PostgreSQL constraint.

3. **Regulatory Framework Compliance Profiles** — `compliance_profile` field on
   `ModelArtifact` selects rule overlays for `eu_ai_act`, `nist_ai_rmf`,
   `iso_42001`, or `internal`. `record_evaluation` enforces profile-appropriate
   evidence gates (e.g., EU AI Act requires conformity assessment docs; NIST
   requires TEVV evidence).

4. **Model Lineage Graph** — `async_build_lineage_graph(tenant_id, version_id)`
   returns a DAG with typed nodes (model, version, dataset, feature-pipeline,
   base-model) and edges (derived_from, trained_on, evaluated_against).
   Serialisable to JSON-LD for ML metadata store interoperability.

5. **Continuous Fairness Monitoring** — `async_record_fairness_metric` stores
   time-series fairness observations per protected attribute.
   `async_fairness_regression_check` computes moving-window disparity trends and
   raises `FairnessAlert` audit events on threshold breach or consecutive-window
   worsening.

6. **Explainability Evidence Registry** — `ExplainabilityRecord` links
   `version_id`, `evaluation_id`, `method`, and `global_importances`.
   `async_record_global_explanation` stamps `explainability_recorded=True` on
   the linked evaluation. `async_get_explainability_evidence(version_id)` returns
   the full evidence chain for audit queries.

7. **Policy-as-Code Hot Reload** — `async_reload_policy(policy_source)` accepts
   JSON/YAML, validates against the policy schema, atomically replaces the
   in-memory rule set, and emits a `policy_reloaded` diff event. Gated to the
   `privileged_admin` agent role. Enables sub-second emergency rule tightening.

8. **Canary Promotion Orchestration** — `async_advance_canary(deployment_id,
   new_pct, health_check_results)` validates error rate, latency P99, and drift
   score against configurable acceptance gates, updates `canary_percent`, and
   auto-calls `request_promotion` with evidence refs when `new_pct == 100`.

9. **Multi-Tenant Governance Report** — `async_governance_report(operator_token)`
   returns operator-scoped cross-tenant aggregates: models by risk level,
   unresolved drift, pending reviews, failed bias audits, and policy violations.
   Per-tenant counts only — no cross-tenant record exposure.

10. **Shadow-Mode Deployment** — `async_create_shadow_deployment` creates a
    `DeploymentRecord` with `status="shadow"`. `async_record_shadow_observation`
    captures output divergence per input hash. `async_shadow_promotion_check`
    gates canary promotion on measured divergence rate.

11. **Model Card Completeness Linter** — `async_lint_model_card(version_id)` runs
    structured completeness checks against a configurable required-sections list
    (intended use, limitations, training data, evaluation results, ethical
    considerations), returns per-section pass/fail with remediation hints, and
    records a `model_card_linted` audit event. Replaces the naive truthy check in
    the deployment gate.

12. **Automated Retraining Trigger** — `async_trigger_retraining(version_id,
    trigger_reason, approved_by)` evaluates unresolved drift + evaluation score
    delta against a retrain threshold, creates a `TrainingJobRecord` with
    `trigger=automatic`, links causal drift signals, and emits
    `retraining_triggered`. Human approval is gated by model risk level.

13. **Composable Audit Query Engine** — `async_query_audit(tenant_id, filters)`
    accepts a typed `AuditQuery` (event_types, subject_ids, from_ts, to_ts,
    min_severity, policy_decisions, page, page_size) and returns paginated
    results with a `correlation_chain` linking causally related events
    (drift_recorded → retraining_triggered → model_evaluated → promoted).

14. **SBOM-Style Model Bill of Materials** — `async_generate_mbom(version_id)`
    produces a CycloneDX-analogous JSON document covering base model, datasets,
    framework versions, dependency hashes, training infrastructure, and active
    deployment targets. Stored as a version attachment and linked from the model
    card.

15. **Federated Model Registry Bridge** — `FederatedRegistryAdapter` protocol
    with `async_pull_remote_model`, `async_push_version`, and
    `async_sync_evaluation`. Reference `MlflowRegistryAdapter` implementation
    included. Surfaces through standard `register_model`/`create_version`/
    `record_evaluation` calls with `metadata.source` stamped for traceability.

## New Methods

### Bias Audit

```python
result = service.bias_audit(
	tenant_id="tenant-a",
	version_id="fraud-risk-v1",
	protected_attributes=["gender", "age_band", "postcode"],
	dataset_ref="dataset:payments-2026-q1",
	auditor="fairness-team",
)
# result["passed"]       -> bool, True when max_disparity < 0.1
# result["disparities"]  -> {"gender": 0.05, "age_band": 0.07, ...}
# result["max_disparity"] -> float
```

### Model Explanation

```python
explanation = service.model_explain(
	tenant_id="tenant-a",
	version_id="fraud-risk-v1",
	sample_input={"amount": 1500.0, "merchant_category": "travel", "hour": 23},
	method="shap",
)
# explanation["feature_importances"] -> {"amount": 0.41, "hour": 0.35, ...}
# explanation["top_feature"]         -> "amount"
```

### A/B Testing

```python
ab = service.model_ab_test(
	tenant_id="tenant-a",
	deployment_a_id="deploy-fraud-v1",
	deployment_b_id="deploy-fraud-v2",
	traffic_split_pct=20,
	metric="auc",
)
# ab["ab_test_id"]        -> stable deterministic ID
# ab["traffic_split_pct"] -> 20
# ab["status"]            -> "active"
```

### Performance Degradation Alert

```python
alert = service.performance_degrade_alert(
	tenant_id="tenant-a",
	version_id="fraud-risk-v1",
	current_score=0.83,
	baseline_score=0.91,
	threshold_delta=0.05,
)
# alert["degraded"]   -> True (delta=0.08 > threshold=0.05)
# alert["severity"]   -> "warning" | "critical"
# alert["delta"]      -> 0.08
```

### Analytics Aggregation

```python
stats = service.mlcm_analytics(tenant_id="tenant-a", period_label="2026-q2")
# stats["average_eval_score"]        -> float
# stats["unresolved_drift_count"]    -> int
# stats["approved_promotion_count"]  -> int
# stats["hyperparameter_tuning_count"] -> int
```

### Hyperparameter Tuning

```python
tuning = service.hyperparameter_tune(
	tenant_id="tenant-a",
	model_id="fraud-risk",
	param_grid={"lr": [1e-3, 1e-4], "depth": [4, 6, 8]},
	tuning_strategy="bayesian",
	max_trials=30,
	metric="auc",
)
# tuning["best_params"] -> {"lr": 0.001, "depth": 4}
# tuning["best_score"]  -> float
# tuning["status"]      -> "completed"
```

## API Reference

| Method | Description |
|---|---|
| `register_model` | Register a new model artifact with tenant scope |
| `create_version` | Create a versioned artifact with lineage and model card |
| `record_evaluation` | Record evaluation run with fairness and explainability gates |
| `request_promotion` | Request stage promotion with approval enforcement |
| `create_target` | Register a deployment target endpoint |
| `deploy_model` | Deploy a version to a target with drift blocking |
| `record_drift` | Record a data drift signal for a version |
| `record_drift_review` | Mark a drift signal as reviewed |
| `rollback_deployment` | Rollback a deployment to a prior version |
| `retire_model` | Retire a model after impact review |
| `register_model_lifecycle_agent` | Register a first-class ML agent with guardrail evidence |
| `validate_mlcm_lifecycle_batch` | Validate Bytewax lifecycle mutation batch |
| `list_pending_reviews` | List all pending-review versions, evaluations, and agents |
| `dashboard_summary` | Per-tenant operational summary |
| `model_upload` | Upload-annotated model registration |
| `model_validate` | Structural field validation with per-check pass/fail |
| `model_deploy` | Convenience alias for `deploy_model` |
| `model_rollback` | Convenience alias for `rollback_deployment` |
| `model_retire` | Convenience alias for `retire_model` |
| `model_ab_test` | Configure A/B test between two deployment pairs |
| `model_explain` | Generate SHAP/LIME/attention feature importance |
| `model_export` | Export version to ONNX, TorchScript, SavedModel, MLflow, HuggingFace |
| `bias_audit` | Disparity audit across protected attributes |
| `data_drift_detect` | Convenience alias for `record_drift` |
| `model_drift_detect` | Concept/performance drift detection distinct from data drift |
| `performance_degrade_alert` | Alert on score delta exceeding threshold |
| `training_job_submit` | Submit a training job record for a model |
| `hyperparameter_tune` | Submit hyperparameter tuning run (random/grid/Bayesian) |
| `mlcm_analytics` | Cross-domain analytics aggregation for a tenant |
| `list_models` | List all model artifacts for a tenant |
| `list_versions` | List all versions for a tenant |
| `list_evaluations` | List all evaluation runs for a tenant |
| `list_promotion_requests` | List all promotion requests for a tenant |
| `list_deployments` | List all deployment records for a tenant |
| `list_drift_signals` | List all drift signals for a tenant |
| `list_rollbacks` | List all rollback records for a tenant |
| `list_retirements` | List all retirement records for a tenant |
| `list_audit_events` | List all audit events for a tenant |

## Guardrails

MLCM blocks missing tenant context, missing model owners, missing model names,
missing problem type or risk metadata, missing registered models, missing
artifact URIs, non-development versions without model cards, evaluations without
baselines, promotions without evaluation, production promotions without
approval, deployments without model cards, low-score promotion, inactive
deployment targets, unresolved drift, rollback target mismatch, retirement
without impact review, retirement while deployments are serving, cross-tenant
model access, state changes without audit evidence, non-Bytewax monitoring
streams, model lifecycle agents without supported runtime, supported role,
scope, owner, purpose, or contribution disclosure, non-Bytewax lifecycle
batches, incomplete release lineage, and critical-risk operations without human
review.

Incomplete version lineage and high-risk evaluation evidence are not silently
accepted. `create_version()` records pending-review versions for missing
training or baseline lineage, and `record_evaluation()` records pending-review
evaluations for missing evidence, fairness review, or explainability review.
These records include `decision`, `matched_rules`, `review_reasons`, and
`audit_evidence` so applications can compose review queues directly from
service state. `list_pending_reviews()` and the dashboard, version,
evaluation, agent, lifecycle, and governance view models expose the durable
queues without replaying the policy engine.

Privileged model lifecycle agents without human approval are also persisted as
`pending_review` with the same evidence shape. Non-Bytewax lifecycle batches
remain hard denials, but their denied records preserve matched rules and audit
evidence for operator inspection.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/mlcm/__init__.py capabilities/common/mlcm/capability_contract.py capabilities/common/mlcm/models.py capabilities/common/mlcm/lifecycle_runtime.py capabilities/common/mlcm/service.py capabilities/common/mlcm/api.py capabilities/common/mlcm/views.py capabilities/common/mlcm/app.py capabilities/common/mlcm/test_capability_contract.py capabilities/common/mlcm/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/mlcm/test_capability_contract.py capabilities/common/mlcm/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/mlcm --json
./.venv/bin/apg capabilities publish-plan capabilities/common/mlcm --json
```
