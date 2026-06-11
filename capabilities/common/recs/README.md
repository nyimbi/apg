# RECS - Recommender Systems

RECS is APG's governed recommendation and personalization capability. It
provides tenant-scoped recommendation datasets, interaction events, catalog
items, user/profile features, ranking policies, model training, model approval,
model deployment, recommendation generation, feedback capture, A/B experiments,
AI recommender-agent registration, audit, UI surfaces, and Bytewax lifecycle
stream policy.

The package is deterministic and dependency-light. It generates ranked
recommendations locally for tests and demos; production systems attach through
explicit APG adapters for prediction, AI orchestration, NLP features, master
data, ETL, monitoring, audit, and Bytewax stream processing.

## What RECS Provides

**Core Pipeline**
- Dataset registration with owner, source reference, schema fields, and policy.
- Interaction event capture: `impression`, `click`, `dismiss`, `conversion`, `rating`.
- Catalog item registration with features, tags, categories, and sensitive attribute filtering.
- Profile feature recording with consent and segments.
- Ranking policy management: objective, owner, confidence threshold, diversity constraints.
- Model training for `collaborative_filtering`, `content_based`, `hybrid`, `contextual_bandit`.
- Model approval, deployment with target runtime, rollback evidence.
- Recommendation generation with consent, policy, candidate, confidence, diversity, and explainability guardrails.
- Feedback loop capture linked to recommendation sets.
- Experiment creation with approval, holdout, business metric, and large-test review guardrails.
- Recommender agent registration for runtimes such as Codex, Claude Code, OpenCode, Pi.
- Bytewax lifecycle stream contract for batch and runtime recommendation mutations.

**New Async Methods (v1.1)**

| Method | Description |
|--------|-------------|
| `publish_interaction_stream()` | Publish interaction events to NATS JetStream (`recs.interactions.{tenant_id}`) for real-time Bytewax pipeline consumption. Graceful in-process fallback when broker unavailable. |
| `update_profile_features()` | Incremental EMA profile feature update. Emits `recs.profiles.updated.{tenant_id}` to NATS. |
| `contextual_bandit_rank()` | Epsilon-greedy bandit ranking. Separates exploit (highest estimated reward) from explore (random novel) slots. |
| `compute_catalog_popularity()` | Time-decayed popularity scoring with configurable half-life. Writes `popularity_score` back onto catalog item features. |
| `evaluate_experiment_stopping()` | Bayesian early stopping via Beta posterior sampling. Terminates A/B experiments when P(A > B) ≥ threshold. |
| `fairness_rerank()` | MMRS fairness-aware re-ranking enforcing minimum category exposure constraints. |
| `stack_ensemble_rank()` | Weighted ensemble of multiple deployed models via `asyncio.gather`. |
| `anonymize_profile_features()` | k-Anonymity projection — suppresses feature dimensions where fewer than k profiles share the same decile bucket (GDPR Art. 25). |
| `get_or_generate_recommendations()` | In-process TTL cache wrapping `generate_recommendations`. Publishes `recs.cache.invalidated.{tenant_id}` on cache miss. |

## Minimal Usage

```python
from capabilities.common.recs.service import RecsService

service = RecsService()
tenant_id = "tenant-recs"

dataset = service.register_dataset(
    "events", tenant_id, "Interaction Events", "personalization-team",
    "etlp:events", ["profile_id", "item_id", "event_type", "occurred_at"],
    "dataset-policy:events", event_count=2500,
)
item = service.register_catalog_item(
    "course-ai", tenant_id, "AI Course", "course", "learning", {"ai": 0.9}, ["ai"]
)
profile = service.record_profile(
    "profile-001", tenant_id, {"ai": 0.95}, ["ai"], consent_recorded=True
)
policy = service.attach_ranking_policy(
    "policy-safe", tenant_id, "Safe Ranking", "relevance",
    owner="risk-team", minimum_confidence=0.25
)
model = service.train_model(
    "model-hybrid", tenant_id, "Hybrid Model", "hybrid",
    "personalization-team", 2500, ["ai"]
)
service.approve_model(model["id"], tenant_id, "approval:model-hybrid")
service.deploy_model(
    "deploy-hybrid", tenant_id, model["id"], "python",
    "apg://models/recs/model-hybrid", True, "rollback:model-hybrid"
)
recs = service.generate_recommendations(
    "recset-001", tenant_id, model["id"], profile["id"], policy["id"], [item["id"]]
)
service.record_feedback(
    "feedback-001", tenant_id, recs["id"], profile["id"], item["id"], "conversion"
)
```

## Async Feature Usage

```python
import asyncio
from capabilities.common.recs.service import RecsService

svc = RecsService()
# ... register items, profiles, models as above ...

async def run():
    # Stream interaction to NATS for real-time Bytewax pipeline
    await svc.publish_interaction_stream(
        tenant_id, "ev-001", "profile-001", "course-ai", "click",
        "2026-06-11T10:00:00+00:00"
    )

    # Incrementally update profile preferences (EMA blend)
    await svc.update_profile_features(
        "profile-001", tenant_id, {"ai": 1.0, "data_science": 0.8}, ema_alpha=0.3
    )

    # Epsilon-greedy bandit ranking
    bandit_result = await svc.contextual_bandit_rank(
        tenant_id, "profile-001", ["course-ai", "course-ml", "course-db"],
        "model-hybrid", "policy-safe", limit=3, exploration_factor=0.15
    )

    # Fairness-constrained re-ranking (20% local content minimum)
    fair = await svc.fairness_rerank(
        "recset-001", tenant_id, {"category:local": 0.2}
    )

    # Ensemble two models
    ensemble = await svc.stack_ensemble_rank(
        tenant_id, "profile-001", ["course-ai", "course-ml"],
        [("model-cf", 0.6), ("model-cb", 0.4)], "policy-safe", limit=2
    )

    # Cached recommendation generation (TTL 60s)
    r = await svc.get_or_generate_recommendations(
        "recset-002", tenant_id, "model-hybrid", "profile-001", "policy-safe",
        ["course-ai", "course-ml"], ttl_seconds=60
    )
    print(r["cache_hit"])  # False on first call, True on repeat within TTL

asyncio.run(run())
```

## NATS Streaming Integration

RECS publishes to the following NATS subjects when `NATS_URL` is set:

| Subject | Published by |
|---------|--------------|
| `recs.interactions.{tenant_id}` | `publish_interaction_stream()` |
| `recs.profiles.updated.{tenant_id}` | `update_profile_features()` |
| `recs.cache.invalidated.{tenant_id}` | `get_or_generate_recommendations()` on cache miss |

Bytewax pipelines subscribe to `recs.interactions.*` for windowed popularity
aggregation and model drift detection. Set `NATS_URL=nats://localhost:4222` in
the process environment.

## Guardrail Summary

RECS denies operations that lack: tenant context, dataset owner/source/schema/
policy, interaction actor/item/timestamp, profile consent, ranking policy,
candidate items, ranking policy owner, sufficient training events (≥ 1000),
model owner, drift monitoring, model approval, deployment target, deployment
approval, rollback plan, high-impact explanations, feedback actor/event,
recommender-agent registration/runtime/scope/disclosure, state-change
reason/audit, or Bytewax stream evidence for batch mutations.

Rules returning `require_review`: large experiments (> 20% traffic), empty
recommendation outputs.

Valid interaction/feedback event types: `impression`, `click`, `dismiss`,
`conversion`, `rating`.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/recs/service.py
./.venv/bin/pytest -q capabilities/common/recs/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.recs import app; r=app.self_test(); print(r); assert r['passed']"
```
