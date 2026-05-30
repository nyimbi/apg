# RECS - Recommender Systems

RECS is APG's governed recommendation and personalization capability. It
provides tenant-scoped recommendation datasets, interaction events, catalog
items, user/profile features, ranking policies, model training, model approval,
model deployment, recommendation generation, feedback capture, experiments,
AI recommender-agent registration, audit, UI model surfaces, and Bytewax
lifecycle stream policy.

The package is deterministic and dependency-light. It can generate ranked
recommendations locally for tests and demos while production systems attach
through explicit APG adapters for prediction, AI orchestration, NLP features,
master data, ETL, monitoring, audit, and Bytewax stream processing.

## What RECS Provides

- Dataset registration with owner, source reference, schema fields, source
  policy, and event counts.
- Interaction event capture for impressions, clicks, dismissals, conversions,
  and ratings.
- Catalog item registration with features, tags, categories, and sensitive
  attribute filtering.
- Profile feature recording with consent and segments.
- Ranking policy management with objective, owner, confidence threshold,
  diversity constraints, and sensitive attribute filtering.
- Deterministic model training for collaborative filtering, content-based,
  hybrid, and contextual-bandit algorithms.
- Model approval and deployment with target runtime, approval evidence, and
  rollback-plan evidence.
- Recommendation generation with profile consent, policy, candidate, confidence,
  diversity, sensitive-filtering, and explainability guardrails.
- Feedback loop capture linked to recommendation sets.
- Experiment creation with approval, holdout, business metric, and large-test
  review guardrails.
- First-class AI recommender agents for runtimes such as Codex, Claude Code,
  OpenCode, and Pi.
- Bytewax lifecycle stream contract for batch and runtime recommendation
  mutations.

## Minimal Usage

```python
from capabilities.common.recs.service import RecsService

service = RecsService()
tenant_id = "tenant-recs"

dataset = service.register_dataset(
	"events",
	tenant_id,
	"Interaction Events",
	"personalization-team",
	"etlp:events",
	["profile_id", "item_id", "event_type", "occurred_at"],
	"dataset-policy:events",
	event_count=2500,
)
item = service.register_catalog_item("course-ai", tenant_id, "AI Course", "course", "learning", {"ai": 0.9}, ["ai"])
profile = service.record_profile("profile-001", tenant_id, {"ai": 0.95}, ["ai"], consent_recorded=True)
policy = service.attach_ranking_policy("policy-safe", tenant_id, "Safe Ranking", "relevance", owner="risk-team", minimum_confidence=0.25)
model = service.train_model("model-hybrid", tenant_id, "Hybrid Model", "hybrid", "personalization-team", 2500, ["ai"])
service.approve_model(model["id"], tenant_id, "approval:model-hybrid")
service.deploy_model("deploy-hybrid", tenant_id, model["id"], "python", "apg://models/recs/model-hybrid", True, "rollback:model-hybrid")
recommendations = service.generate_recommendations("recset-001", tenant_id, model["id"], profile["id"], policy["id"], [item["id"]])
service.record_feedback("feedback-001", tenant_id, recommendations["id"], profile["id"], item["id"], "conversion")
```

## Guardrail Summary

RECS denies operations that lack tenant context, dataset owner/source/schema/
policy, interaction actor/item/timestamp, profile consent, ranking policy,
candidate items, ranking policy owner, sufficient training events, model owner,
drift monitoring, model approval, deployment target, deployment approval,
rollback plan, high-impact explanations, feedback actor/event, recommender-agent
registration/runtime/scope/disclosure, state-change reason/audit, tenant
isolation, or Bytewax stream evidence for batch mutations.

Some rules return `require_review`, including large experiments and empty
recommendation outputs.

## Focused Verification

Battery-conscious checks for this capability:

```bash
./.venv/bin/python -m py_compile capabilities/common/recs/__init__.py capabilities/common/recs/models.py capabilities/common/recs/recommendation_runtime.py capabilities/common/recs/service.py capabilities/common/recs/api.py capabilities/common/recs/views.py capabilities/common/recs/capability_contract.py capabilities/common/recs/app.py capabilities/common/recs/test_capability_contract.py capabilities/common/recs/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/recs/test_capability_contract.py capabilities/common/recs/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.recs import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/recs --json
./.venv/bin/apg capabilities publish-plan capabilities/common/recs --json
```

