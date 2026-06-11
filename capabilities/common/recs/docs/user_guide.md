# RECS User Guide

**Capability ID**: `recs` | **Domain**: `common` | **Version**: `1.1.0`

RECS is APG's governed recommendation and personalization capability. This guide
covers the full service API, configuration, NATS integration, and best practices.

---

## Installation

```bash
pip install apg-common-recs
```

## Architecture Overview

```
Interaction Events
       |
       v
NATS JetStream (recs.interactions.{tenant_id})
       |
       v
Bytewax Pipeline (windowed popularity aggregation, drift detection)
       |
       v
RecsService (in-process, tenant-scoped)
  |- Catalog Items
  |- Profiles (with EMA feature updates, k-anonymity)
  |- Ranking Policies (fairness, diversity, confidence)
  |- Models (CF, content-based, hybrid, bandit, ensemble)
  |- Recommendation Sets (cached, explainable)
  |- Feedback Loop
  |- A/B Experiments (Bayesian auto-stopping)
  |- Audit Trail
```

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `NATS_URL` | NATS broker for streaming events | _(disabled)_ |
| `OLLAMA_BASE_URL` | Ollama inference for ML-enhanced ranking | _(disabled)_ |
| `RECS_*` | Tenant-scoped config keys via `conf` capability | — |

---

## Core Workflow

### 1. Register a Dataset

```python
from capabilities.common.recs.service import RecsService

svc = RecsService()
svc.register_dataset(
    dataset_id="events-2026",
    tenant_id="acme",
    name="User Interactions 2026",
    owner="data-team",
    source_ref="etlp:s3://events/2026",
    schema_fields=["profile_id", "item_id", "event_type", "occurred_at"],
    policy_ref="dataset-policy:pii-v2",
    event_count=50_000,
)
```

### 2. Build a Catalog

```python
svc.register_catalog_item(
    item_id="prod-001", tenant_id="acme",
    name="Wireless Headphones", item_type="product", category="electronics",
    features={"audio_quality": 0.9, "price_tier": 0.6},
    tags=["audio", "wireless"],
)
```

### 3. Record Profile + Consent

```python
svc.record_profile(
    profile_id="user-42", tenant_id="acme",
    features={"audio_quality": 0.85, "price_tier": 0.4},
    segments=["premium", "music_fan"],
    consent_recorded=True,   # REQUIRED for recommendation generation
)
```

### 4. Attach a Ranking Policy

```python
svc.attach_ranking_policy(
    policy_id="pol-safe", tenant_id="acme",
    name="Safe Relevance Policy", objective="relevance",
    owner="risk-team",
    minimum_confidence=0.5,
    diversity_constraints_enabled=True,
    sensitive_attribute_filtering=True,
    max_per_category=3,
)
```

### 5. Train, Approve, Deploy a Model

```python
result = svc.train_model(
    model_id="model-hybrid-v1", tenant_id="acme",
    name="Hybrid Rec v1", algorithm="hybrid",
    owner="ml-team", training_event_count=12_000,
    feature_names=["audio_quality", "price_tier"],
    drift_monitoring_enabled=True,
    metric_name="ndcg_at_10", metric_value=0.81,
)
svc.approve_model("model-hybrid-v1", "acme", approval_ref="jira:ML-442")
svc.deploy_model(
    "dep-hybrid-v1", "acme", "model-hybrid-v1",
    target_runtime="python", target_ref="apg://models/recs/hybrid-v1",
    approval_recorded=True, rollback_plan_ref="runbook:recs-rollback-v1",
)
```

### 6. Generate Recommendations

```python
recs = svc.generate_recommendations(
    recommendation_id="recset-001", tenant_id="acme",
    model_id="model-hybrid-v1", profile_id="user-42",
    policy_id="pol-safe",
    candidate_item_ids=["prod-001", "prod-002", "prod-003"],
    limit=5, impact_level="low", explanation_attached=True,
)
```

### 7. Close the Feedback Loop

Valid event types: `impression`, `click`, `dismiss`, `conversion`, `rating`.

```python
svc.record_feedback(
    "fb-001", "acme", recs["id"], "user-42", "prod-001", "conversion", value=1.0
)
```

---

## Async Features

All async methods are `await`-able and safe to call from any asyncio event loop.

### Real-Time Interaction Streaming (NATS)

```python
# Set NATS_URL=nats://localhost:4222 before calling
await svc.publish_interaction_stream(
    tenant_id="acme", event_id="ev-stream-1",
    profile_id="user-42", item_id="prod-001",
    event_type="click", occurred_at="2026-06-11T10:00:00+00:00",
    weight=1.0,
)
# Publishes to: recs.interactions.acme
# Bytewax pipeline picks this up for windowed aggregation
```

### Incremental Profile Updates (EMA)

```python
# alpha=0.3 means 30% new signal, 70% historical
await svc.update_profile_features(
    "user-42", "acme",
    delta_features={"audio_quality": 1.0, "price_tier": 0.2},
    ema_alpha=0.3,
)
# Publishes to: recs.profiles.updated.acme
```

### Contextual Bandit Ranking

```python
result = await svc.contextual_bandit_rank(
    tenant_id="acme", profile_id="user-42",
    candidate_item_ids=["prod-001", "prod-002", "prod-003", "prod-004"],
    model_id="model-hybrid-v1", policy_id="pol-safe",
    limit=5,
    exploration_factor=0.15,  # 15% random exploration, 85% exploitation
)
# result["ranked_items"][i]["strategy"] is "exploit" or "explore"
```

### Catalog Popularity Decay

```python
# Recompute with 14-day half-life; updates features["popularity_score"] on each item
result = await svc.compute_catalog_popularity("acme", half_life_days=14.0)
# Schedule this via Bytewax cron or APG scheduler daily
```

### Bayesian A/B Experiment Auto-Stopping

```python
# After sufficient feedback has been recorded:
result = await svc.evaluate_experiment_stopping(
    "exp-001", "acme", superiority_threshold=0.95
)
if result["stop_early"]:
    print(f"Winner: variant {result['winner']} — P(A>B)={result['p_a_wins']}")
```

### Fairness-Aware Re-Ranking

```python
# Ensure >= 20% of slots are filled with "local" category items
result = await svc.fairness_rerank(
    "recset-001", "acme",
    fairness_constraints={"category:local": 0.2, "category:indie": 0.1},
)
```

### Ensemble Ranking (Multiple Models)

```python
result = await svc.stack_ensemble_rank(
    tenant_id="acme", profile_id="user-42",
    candidate_item_ids=["prod-001", "prod-002", "prod-003"],
    model_weights=[("model-cf-v1", 0.5), ("model-cb-v1", 0.3), ("model-hybrid-v1", 0.2)],
    policy_id="pol-safe", limit=5,
)
```

### k-Anonymity Profile Privacy

```python
# Suppress feature dimensions where fewer than k=5 profiles share the same decile
result = await svc.anonymize_profile_features("user-42", "acme", k=5)
print(result["suppressed_features"])   # features that were removed
print(result["suppression_ratio"])     # 0.0 means no suppression needed
```

### Cached Recommendation Generation

```python
# First call: cache miss, generates fresh recommendations
r = await svc.get_or_generate_recommendations(
    "recset-002", "acme", "model-hybrid-v1", "user-42", "pol-safe",
    ["prod-001", "prod-002", "prod-003"], ttl_seconds=60,
)
assert not r["cache_hit"]

# Second call within TTL: served from cache
r2 = await svc.get_or_generate_recommendations(
    "recset-003", "acme", "model-hybrid-v1", "user-42", "pol-safe",
    ["prod-001", "prod-002", "prod-003"], ttl_seconds=60,
)
assert r2["cache_hit"]
```

---

## A/B Testing

```python
# Create experiment
svc.create_experiment(
    "exp-001", "acme", "Algorithm Comparison",
    model_id="model-hybrid-v1", policy_id="pol-safe",
    experiment_percent=10, holdout_percent=10,
    business_metric="revenue_per_session",
    approved=True, review_recorded=True,
)

# Assign a profile to a variant
result = svc.rec_ab_test(
    "exp-001", "acme", "user-42",
    variant_a_model_id="model-cf-v1",
    variant_b_model_id="model-hybrid-v1",
)
# result["variant"] is "A" or "B", deterministic on (profile_id, experiment_id)
```

---

## Analytics

```python
analytics = svc.rec_analytics("acme")
# Returns: ctr, cvr, item_coverage_pct, recommendation_set_count, feedback_count, ...
```

---

## Cold Start

```python
# New profile with no interaction history
result = svc.cold_start_handle(
    profile_id="new-user-99", tenant_id="acme",
    strategy="popular_items", limit=5,
)
```

---

## Guardrails Reference

| Operation | Guardrail |
|-----------|-----------|
| All | Tenant context required |
| `register_dataset` | owner, source, schema, policy required |
| `record_interaction` | actor, item, timestamp required |
| `generate_recommendations` | profile consent, ranking policy, ≥ 1 candidate |
| `train_model` | ≥ 1000 training events, owner, drift monitoring |
| `deploy_model` | model approved, deployment target, approval, rollback plan |
| `create_experiment` | approval required, holdout ≥ 1%, business metric |
| High-impact recommendations | `explanation_attached=True` required |
| Large experiments (> 20%) | `review_recorded=True` required |

---

## Service Methods Reference

### Sync (Core Pipeline)

- `describe(tenant_id)` — capability contract
- `evaluate(context)` — rule evaluation
- `register_catalog_item(...)` — catalog registration
- `register_dataset(...)` — dataset registration
- `record_interaction(...)` — interaction event
- `record_profile(...)` — profile features + consent
- `attach_ranking_policy(...)` — ranking policy
- `train_model(...)` — model training
- `approve_model(...)` — model approval
- `deploy_model(...)` — model deployment
- `generate_recommendations(...)` — ranked recommendation set
- `record_feedback(...)` — feedback capture
- `create_experiment(...)` — A/B experiment
- `record_drift(...)` — drift status update
- `register_recommender_agent(...)` — AI agent registration
- `change_model_state(...)` — lifecycle state transition
- `cold_start_handle(...)` — new-user bootstrap
- `diversity_inject(...)` — category diversity re-rank
- `serendipity_boost(...)` — novel item injection
- `recency_weight(...)` — recency-decayed item scores
- `multi_objective_rank(...)` — weighted multi-objective re-rank
- `session_based_rec(...)` — transient session recommendations
- `knowledge_graph_rec(...)` — item relationship traversal
- `explainable_rec(...)` — human-readable explanations
- `rec_ab_test(...)` — variant assignment
- `rec_analytics(...)` — system analytics
- `dashboard_summary(...)` — counts dashboard

### Async (v1.1)

- `publish_interaction_stream(...)` — NATS interaction publish
- `update_profile_features(...)` — EMA incremental feature update
- `contextual_bandit_rank(...)` — epsilon-greedy bandit ranking
- `compute_catalog_popularity(...)` — time-decayed popularity scoring
- `evaluate_experiment_stopping(...)` — Bayesian early stopping
- `fairness_rerank(...)` — MMRS fairness-constrained re-ranking
- `stack_ensemble_rank(...)` — weighted multi-model ensemble
- `anonymize_profile_features(...)` — k-anonymity projection
- `get_or_generate_recommendations(...)` — TTL-cached recommendation generation
- `ml_generate_recommendations(...)` — Ollama-backed relevance classification

---

## Further Reading

- `service.py` — Complete business logic
- `models.py` — Dataclass domain models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Rule definitions and guardrails
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural improvement proposals
- `README.md` — Quick reference
