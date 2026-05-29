# AI Model Lifecycle Management Capability Specification

- **Capability Name**: AI Model Lifecycle Management
- **Capability ID**: `mlcm`
- **Category**: common
- **Version**: 1.0.0

## Purpose

MLCM is APG's package-backed model operations capability. It gives composed
applications a deterministic model registry, version manager, evaluation
evidence store, promotion gate, deployment board, drift monitor, rollback
ledger, governance rule surface, UI route model, and theme contract.

The package is intentionally executable without external infrastructure. Live
model registries, artifact stores, inference platforms, monitoring systems,
and audit stores should sit behind explicit adapters while this package keeps
the APG contract and local lifecycle behavior testable.

## Provided Services

- `model_registry`
- `model_versioning`
- `model_evaluation`
- `promotion_gates`
- `deployment_board`
- `drift_monitoring`
- `model_governance`
- `mlcm_operations`

## Required Services

- `tenant_context`
- `aicr` for AI-core runtime and inference integration
- `moni` for monitoring and operational metrics
- `audl` for durable audit publication
- Optional `cach`, `conf`, and `auth` integration from registration metadata

## Runtime Surfaces

| File | Responsibility |
| --- | --- |
| `capability_contract.py` | Configuration schema, deterministic rule engine, UI routes, and theme. |
| `models.py` | Domain dataclasses for models, versions, evaluations, promotions, deployment targets, deployments, drift signals, rollbacks, and audit events. |
| `lifecycle_runtime.py` | Deterministic IDs, stage and score normalization, model-card completeness, promotion posture, deployment posture, and drift posture helpers. |
| `service.py` | In-process lifecycle service enforcing tenant, owner, approval, model-card, evaluation, and drift-review guardrails. |
| `api.py` | Thin payload helpers for registry, version, evaluation, promotion, deployment, drift, rollback, status, and compatibility calls. |
| `views.py` | Dashboard, registry, version, evaluation, deployment, drift, and governance view models. |
| `app.py` | Package entrypoint, manifest, semantic model, and self-test surface. |

## Lifecycle Behavior

1. Register a tenant-scoped model with an owner, problem type, risk level, and
   metadata.
2. Create a model version with an artifact URI, optional model card, training
   data reference, and baseline reference.
3. Record evaluation evidence against a version. Scores are normalized to
   `[0, 1]` and compared with the configured minimum evaluation score.
4. Request promotion through `dev`, `staging`, and `production` stages.
   Production promotion requires approval and all promotion requests enforce
   the evaluation-score gate.
5. Create deployment targets and deploy eligible versions. Deployments require
   complete model-card evidence and unresolved drift blocks continued serving.
6. Record drift signals and resolve detected drift with review records.
7. Roll back deployments to a compatible version and emit audit evidence.

## Rules

- `tenant_context_required`
- `model_registration_requires_owner`
- `production_promotion_requires_approval`
- `deployment_requires_model_card`
- `low_eval_score_blocks_promotion`
- `drifted_model_requires_review`

## UI

The package exposes 8 APG Python UI routes through `views.py` and the package
semantic model:

- `/mlcm/dashboard`
- `/mlcm/models`
- `/mlcm/versions`
- `/mlcm/evaluation`
- `/mlcm/deployments`
- `/mlcm/drift`
- `/mlcm/governance`
- `/mlcm/settings`

## Theme

The package uses the `mlcm_model_ops_console` APG theme contract. The theme is
optimized for compact model-ops work: version rows, promotion gate panels,
drift-monitor charts, and model-card evidence panels.

## Adapter Boundaries

The executable package does not call external systems directly. Production
integrations should be introduced through adapters for:

- Model registry backends such as MLflow, SageMaker Model Registry, Vertex AI,
  Azure ML, or an APG-native registry service.
- Artifact and dataset stores such as object storage, feature stores, vector
  stores, and lineage catalogs.
- Evaluation runners, safety harnesses, bias tests, and red-team evidence
  capture systems.
- Deployment substrates such as Kubernetes, serverless inference endpoints,
  batch scoring jobs, edge deployments, or APG AI Core runtimes.
- Monitoring, drift detection, observability, alerting, audit, and incident
  response systems.

## Focused Verification

Use focused checks while battery-constrained:

```bash
./.venv/bin/python -m py_compile capabilities/common/mlcm/__init__.py capabilities/common/mlcm/models.py capabilities/common/mlcm/lifecycle_runtime.py capabilities/common/mlcm/service.py capabilities/common/mlcm/api.py capabilities/common/mlcm/views.py capabilities/common/mlcm/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/mlcm/test_capability_contract.py capabilities/common/mlcm/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/mlcm --json
./.venv/bin/apg capabilities publish-plan capabilities/common/mlcm --json
```
