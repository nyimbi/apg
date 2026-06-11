# DEPL — World-Class Improvement Opportunities

© 2025 Datacraft | Author: Nyimbi Odero

---

## 1. Async-First Service Layer

All public methods are synchronous. Real deployments involve I/O-bound work
(Kubernetes API, container registries, S3, Vault). Converting to `async def`
throughout and replacing the in-memory dicts with an `asyncpg`-backed store
eliminates the async/sync impedance mismatch at composition boundaries.

## 2. Persistent Storage via asyncpg + Alembic

The in-memory dicts survive only within a single process. Migrating to
PostgreSQL with `asyncpg` + the existing Alembic skeleton (already scaffolded in
`alembic/`) gives ACID durability, concurrent tenant isolation, and the ability
to survive restarts. Queries on `tenant_id`, `status`, and `strategy` columns
should be indexed.

## 3. Progressive Canary Automation with SLO Guard-rails

`canary_promote` requires the caller to drive each step manually. A
`canary_autopilot` method that polls a metric adapter (error rate, p99 latency,
saturation) and auto-advances or auto-rolls-back based on SLO thresholds brings
the capability closer to Argo Rollouts / Flagger semantics.

## 4. Real-Time Deployment Event Bus

Audit events accumulate in a list but are never pushed anywhere. Integrating
with the `ntfy` capability via an `asyncio.Queue` + background emitter gives
subscribers (Slack, PagerDuty, audit stream) live visibility. Events should be
CloudEvents-shaped for interoperability.

## 5. Infrastructure-as-Code Plan Diffing

`deployment_diff` compares two `DeploymentRun` objects but ignores IaC
artifacts. Extending it to diff Terraform/Pulumi plan files attached to the
release manifest gives operators structural change impact before execution.

## 6. Multi-Approval Quorum Gate

`approve_deployment_plan` accepts any single approver. Production-grade
workflows need N-of-M approval quorums (e.g. 2 of 3 SRE leads). Tracking
individual approver votes in a `DeploymentApproval` model, requiring quorum
before status transitions to `approved`, blocks solo-actor bypass.

## 7. Release Window Enforcement

`change_freeze` marks a window but no method blocks deployment execution during
an active freeze. Adding an active-freeze check inside `execute_deployment` (and
exposing a `freeze_window_active` predicate) enforces blackout periods rather
than merely recording them.

## 8. Deployment Velocity and DORA Metrics

`deployment_analytics` computes rollback rate but misses DORA's four key metrics:
deployment frequency, lead time for changes, mean time to restore, and change
failure rate. Computing these from the existing timestamp fields would turn the
analytics endpoint into a true engineering health dashboard.

## 9. Secret / Config Drift Detection

Deployments often fail silently because a config key was removed or renamed
between versions. Comparing `config_overrides` between the current and previous
run for the same service + environment, and surfacing any missing or added keys
as a `config_drift` warning in `deployment_health`, prevents silent
misconfiguration.

## 10. Deployment Dependency Graph

Microservice deployments have inter-service dependencies (service B must deploy
after service A). Adding a `depends_on: list[str]` field to `DeploymentPlan` and
a topological-sort scheduler in `deployment_plan` ensures dependency ordering is
respected and surfaced in the plan preview.

## 11. Immutable Artifact Promotion Across Environments

Artifacts are registered once but there is no first-class concept of promoting
an artifact from `staging` to `production`. An `artifact_promote` method that
re-attaches provenance metadata, re-checks the digest signature, and creates an
audit trail bridges the inner and outer deployment loops.

## 12. Automatic Rollback on Post-Deploy Test Failure

`post_deploy_test` sets `run.status = "post_deploy_failed"` but does not
initiate a rollback. Adding an `auto_rollback` flag to `DeploymentPlan` and
triggering `execute_rollback` automatically on test failure closes the feedback
loop without operator intervention.

## 13. Multi-Region / Multi-Cluster Deployment Orchestration

A single `DeploymentRun` maps to one environment. Orchestrating a release across
multiple regions or clusters (with a configurable rollout order, inter-region
health checks, and the ability to halt mid-rollout) requires a `MultiRegionPlan`
aggregate that coordinates multiple child `DeploymentRun` records.

## 14. Policy-as-Code Rule Hot-Reload

`evaluate_capability_rules` is statically compiled into `capability_contract.py`.
Supporting OPA/Rego bundles or Polar rule files hot-reloaded at runtime gives
security and platform teams the ability to update deployment guardrails without
redeploying the service, with versioned rule bundles and A/B rule testing.

## 15. Deployment Attestation and SBOM Integration

Supply-chain security requires a Software Bill of Materials (SBOM) in CycloneDX
or SPDX format to be attached to every release. Adding an `sbom_reference` field
to `ReleaseManifest`, verifying the SBOM digest at `create_deployment_plan`, and
publishing a signed attestation record at `execute_deployment` closes the
SLSA Level 3 provenance gap.
