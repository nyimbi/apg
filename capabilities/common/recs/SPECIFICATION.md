# RECS Capability Specification

## Identity

- Capability ID: `recs`
- Display name: Recommender Systems
- Category: common
- Runtime target: Python capability package
- Primary purpose: compose governed recommendation systems from interaction
  datasets, catalogs, profiles, ranking policies, model training, model
  deployment, feedback loops, experiments, AI recommender agents, and audit.

## Goals

RECS must provide a practical path from recommendation data to executable
recommendation results while keeping personalization governance explicit. The
local runtime remains deterministic so APG examples, generated applications,
and tests can run without production prediction infrastructure.

The capability must support:

- Dataset and interaction-event lifecycle.
- Catalog and profile feature management.
- Ranking policy design and enforcement.
- Recommendation model training, approval, deployment, drift tracking, and
  lifecycle state changes.
- Recommendation generation with consent, policy, confidence, diversity,
  sensitive-filtering, and explainability guardrails.
- Feedback loops for impressions, clicks, dismissals, conversions, and ratings.
- Experiment creation with holdout, approval, business metric, and review gates.
- First-class AI recommender-agent composition.
- Bytewax lifecycle stream policy.
- UI models for operations, governance, and analytics.

## Lifecycle

1. **Register dataset**: capture owner, source reference, schema fields, source
   policy, and known event count.
2. **Record interactions**: store timestamped profile-item events.
3. **Register catalog**: define candidate items with features, tags, category,
   and sensitive attributes.
4. **Record profiles**: capture profile features, segments, and consent.
5. **Attach ranking policy**: define objective, owner, confidence threshold,
   diversity, sensitive filtering, and category limits.
6. **Train model**: train a deterministic model with algorithm, owner,
   features, training-event count, metric, and drift monitoring.
7. **Approve model**: attach approval evidence before deployment.
8. **Deploy model**: attach target runtime, target reference, deployment
   approval, and rollback-plan evidence.
9. **Generate recommendations**: rank candidates using policy, profile, model,
   confidence threshold, diversity, sensitive filtering, and explainability
   rules.
10. **Capture feedback**: record feedback against recommendation sets.
11. **Run experiments**: create governed experiments with approval, holdout,
   business metric, and review controls.
12. **Register agents**: configure AI recommender agents by runtime, role,
   scope, policy, registration, and disclosure.
13. **Monitor and govern**: record drift, audit events, analytics, and state
   changes.

## Domain Model

- `RecommendationDataset`: governed interaction/source dataset.
- `InteractionEvent`: timestamped profile-item event.
- `RecommendationCatalogItem`: recommendable item/content/entity.
- `RecommendationProfile`: profile features, segments, and consent.
- `RankingPolicy`: ranking objective and constraints.
- `RecommendationModel`: trained model with owner, algorithm, approval, drift,
  and status.
- `TrainingRun`: training evidence and metric.
- `ModelDeployment`: model deployment with target, approval, and rollback.
- `RecommendationSet`: ranked recommendation output.
- `RecommendationFeedback`: feedback linked to a recommendation set.
- `RecommendationExperiment`: governed experiment.
- `RecommenderAgent`: first-class AI agent collaborator.
- `RecsAuditEvent`: local audit evidence for recommendation operations.

## Rule Engine

The deterministic rule engine evaluates operation context and returns `allow`,
`require_review`, or `deny`. Rules cover tenant context, dataset governance,
interaction validity, consent, ranking policy, candidates, empty outputs, model
training, model approval, model deployment, explainability, experiments,
feedback, AI recommender agents, state changes, cross-tenant access, and Bytewax
batch mutation requirements.

## UI Contract

RECS exposes APG Python UI model routes:

- `/recs/dashboard`
- `/recs/recommendations`
- `/recs/datasets`
- `/recs/models`
- `/recs/deployments`
- `/recs/catalogs`
- `/recs/profiles`
- `/recs/feedback`
- `/recs/experiments`
- `/recs/policies`
- `/recs/agents`
- `/recs/audit`
- `/recs/analytics`
- `/recs/settings`

## Adapter Boundaries

RECS does not call external systems directly. Production integrations should be
attached through:

- `pred` for production model scoring.
- `aicr` for AI orchestration and agent services.
- `nlpc` for text/content features.
- `mdm` for master data catalogs and entity references.
- `etlp` for data ingestion and feature pipelines.
- `audl` for durable audit.
- `moni` for metrics and observability.
- `bytewax` for lifecycle event streams and batch recommendation mutation.

## Non-Goals For This Packet

- Live feature-store integration.
- Live vector retrieval.
- Live production prediction serving.
- External AI-agent CLI invocation.
- Browser-rendered UI verification.
- Full performance tuning.

Those are adapter and hardening passes after the executable lifecycle spine is
stable.

