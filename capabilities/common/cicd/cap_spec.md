# Continuous Integration and Delivery Capability Specification

- **Capability Name**: Continuous Integration and Delivery
- **Capability ID**: `cicd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package implements the executable APG contract for `cicd` as a
dependency-light pipeline and release-governance runtime. It provides tenant
pipeline definitions, worker/cache/secret policy state, build runs, trace IDs,
artifact digests and signatures, quality gate results, promotions, audit
events, UI route metadata, semantic-model publication, and publish-plan evidence
without requiring an external build runner or deployment platform.

## Provided Services

- `pipeline_management`
- `build_orchestration`
- `quality_gates`
- `artifact_promotion`
- `release_automation`
- `capability_rules`

## Required Services

- `tenant_context`
- `deployment_environment`
- `log_trace_storage`
- `artifact_store`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations. Pipelines require owners, source policy, workers, secret scope,
cache policy, stages, and quality gate policy. Builds require active pipelines,
secret scope, and trace capture. Promotions require signed artifacts, passing
quality gates, and approval evidence.

## Rules

- `tenant_context_required`
- `pipeline_requires_owner`
- `build_requires_secret_scope`
- `artifact_requires_signature`
- `promotion_requires_quality_gate`
- `high_parallelism_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model. The dashboard view model surfaces
pipelines, builds, artifacts, quality gates, promotions, audit events, and
pipeline summary metrics from `CicdService`.

## Theme

The package uses the `cicd_pipeline_ops` APG theme contract.

## Runtime Behavior

`CicdService` maintains deterministic in-memory registries for pipeline
definitions, build runs, artifacts, quality gates, promotions, and audit events.
`cicd_engine.py` generates canonical build trace IDs, artifact digests, and
quality gate findings. Pipeline creation enforces owner, source, worker, secret,
cache, stage, quality gate, and capacity review rules. Promotion enforces signed
artifacts, passing quality gates, and explicit approval.

## Known Integration Boundary

This package intentionally avoids live Git, build-runner, container registry,
scanner, or deployment calls. Actual build execution, log shipping, artifact
storage, secret retrieval, environment provisioning, and deployment rollout
should be composed through APG capabilities such as `depl`, `envm`, `logt`,
`scpt`, `comp`, `edge`, `auth`, `audl`, and `ntfy`.
