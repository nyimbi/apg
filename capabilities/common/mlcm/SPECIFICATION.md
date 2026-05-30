# MLCM Capability Specification

## Purpose

AI Model Lifecycle Management provides the APG model-operations substrate for
generated applications. It turns AI models into governed, tenant-scoped
application components with clear lifecycle state, auditable evidence, UI
surfaces, deterministic policy checks, and composition adapters.

## Lifecycle

The coherent lifecycle packet is:

1. Register a model with tenant, owner, problem type, risk level, and metadata.
2. Create model versions with artifact, model card, training data, and baseline
   lineage.
3. Record evaluation evidence and score the version against the configured
   release threshold.
4. Request promotion through dev, staging, and production gates.
5. Create deployment targets and deploy approved versions.
6. Monitor drift and quality signals through the Bytewax stream adapter.
7. Record drift reviews and allow continued serving only when review is present.
8. Roll back deployments to a same-model version when release risk emerges.
9. Retire models only after impact review and serving deployment drain.
10. Emit audit evidence for lifecycle state changes.

## Functional Requirements

- The capability must expose an executable contract through
  `get_capability_contract()`.
- Configuration must cover registry, versions, evaluation, promotion,
  deployment, monitoring, governance, observability, adapters, UI, and theme.
- The rule engine must be deterministic and dependency-light.
- Generated applications must be able to call `MlcmService` without external
  infrastructure.
- UI routes must cover dashboard, registry, versions, model cards, evaluation,
  baselines, promotion, deployments, drift, rollback, governance, audit, and
  settings.
- The theme must provide model-operations console tokens and component hints.
- The package evidence must be generated from the current contract, not copied
  static JSON.
- Bytewax must be the event-stream adapter; Kafka is intentionally not used.

## Runtime Requirements

- `MlcmService` owns in-process records for models, versions, evaluations,
  promotions, targets, deployments, drift signals, rollbacks, retirements, and
  audit events.
- Service methods must raise `PermissionError` for policy denials and
  `LookupError` for missing tenant-scoped records.
- Summary and list calls must remain tenant-scoped.
- Compatibility calls `create_record()` and `list_records()` must continue to
  map to model registry behavior for older package callers.

## Guardrails

The contract must include guardrails for tenant context, model ownership,
registration metadata, version lineage, evaluation baseline/evidence, high-risk
review, promotion approval, model cards, score thresholds, active deployment
targets, deployment approval, rollout review, health checks, drift review,
unresolved drift, rollback reasons, rollback target model matching, retirement
impact review, deployment drain, cross-tenant access, audit evidence, Bytewax
streaming, release lineage, and critical-risk human review.

## Composition Interfaces

- AICR: AI model/provider and runtime integration.
- AUTH: tenant and permission enforcement.
- AUDL: audit sink for model lifecycle evidence.
- MONI: metrics and drift monitoring sink.
- Bytewax: streaming event engine for model monitoring flows.
- File artifact store: model artifact and evidence references.

## Non-Goals For This Packet

- Live model-provider calls.
- Live Bytewax stream execution.
- Persistent database schema migration.
- Browser-rendered UI implementation.
- Load, latency, token-cost, drift, and throughput benchmarking.
