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
- Rollback records and model retirement after impact review and deployment
  drain.
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
