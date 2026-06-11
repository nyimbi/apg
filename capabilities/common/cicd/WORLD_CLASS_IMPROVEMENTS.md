# CICD - World Class Improvements

15 high-leverage improvements that raise this capability from solid to industry-leading.

---

### I1. Real DORA Metrics Engine with Timestamp-Based Computation

**Category**: Observability / Engineering Excellence
**Justification**: Current MTTR and lead-time methods return synthetic constants (15 min, 45 min). Real DORA metrics require actual start/end timestamps on build, deploy, and rollback records. Without them the analytics dashboard is decorative. Fix this and you get actionable SLO inputs rather than placeholder KPIs.
**Implementation**: Store `started_at`/`completed_at` as ISO strings on builds and deployments. In `mean_time_to_restore`, join rollback records to their parent deployments and compute `(rolled_back_at - deployed_at)`. In `lead_time_for_changes`, join commits (via `commit_ref`) to their first production deployment and compute elapsed minutes. Expose a `dora_report()` method returning all four DORA keys.
**Competitor**: LinearB, Faros.ai, Haystack — all parse actual git/deploy timestamps for DORA computation.

---

### I2. Deployment Cost Tracking with Decimal Precision

**Category**: FinOps / Cost Governance
**Justification**: Every pipeline run consumes compute. Ignoring cost makes CI/CD a black-box spend. Attaching `Decimal`-typed cost fields to builds and deployments enables chargeback per tenant, per pipeline, per environment — directly composable with the `walt` capability.
**Implementation**: Add `compute_cost_usd: Decimal` to build records (worker minutes × cost/min). Add `deployment_cost_usd: Decimal` to deployment records. Expose `cost_report(period: str)` returning total, per-pipeline, and per-environment cost breakdowns using `Decimal` arithmetic. Use `from decimal import Decimal, ROUND_HALF_UP`.
**Competitor**: AWS CodePipeline cost explorer, Harness Cloud Cost Management, Spacelift cost attribution.

---

### I3. Pipeline-as-Code Schema Validation

**Category**: Developer Experience / Correctness
**Justification**: Pipelines are currently defined imperatively via `pipeline_create()`. Large teams need declarative pipeline definitions in YAML/JSON that can be validated before execution. Schema validation at intake eliminates an entire class of runtime failures.
**Implementation**: Add `pipeline_import(definition: dict[str, Any]) -> _R` that validates required keys (`name`, `stages`, `triggers`, `environment`), stage names against an allowed list, and trigger types. Return structured validation errors rather than raising. Integrate with `cap_spec.md` schema anchors.
**Competitor**: GitHub Actions workflow validation, GitLab CI YAML linter, Tekton pipeline validation webhooks.

---

### I4. Approval Workflow with Multi-Party Sign-Off

**Category**: Governance / Compliance
**Justification**: `deployment_promote` accepts a single `approved_by` string, which is trivially spoofed. Enterprise compliance (SOC2, FedRAMP, ISO 27001) requires multi-party approval with timestamps and cryptographic non-repudiation. A single-string field is a compliance gap.
**Implementation**: Add `approval_request_create(artifact_id, required_approvers: list[str], min_approvals: int)` and `approval_submit(request_id, approver, decision, comment)`. Block promotion unless `len(approvals where decision=="approved") >= min_approvals`. Emit `approval_completed` audit events with all approver IDs.
**Competitor**: Spinnaker manual judgements, Harness approval steps, ArgoCD sync waves with RBAC.

---

### I5. Flaky Test Detection and Quarantine

**Category**: Build Intelligence
**Justification**: Flaky tests are the #1 source of false build failures and developer productivity loss. Without tracking per-test pass/fail history there is no way to distinguish real regressions from noise, leading to alarm fatigue and unsafe "retry until green" culture.
**Implementation**: Add `record_test_result(build_id, test_name, outcome, duration_ms)` and `flaky_test_report()`. Track per-test outcome history across builds. Flag a test as flaky if it alternates pass/fail across the last N builds. Expose quarantine API: `quarantine_test(test_name, reason)` that excludes it from gate evaluation.
**Competitor**: BuildKite test analytics, Gradle Enterprise flakiness detection, Trunk Flaky Tests.

---

### I6. Environment Drift Detection

**Category**: Deployment Safety
**Justification**: The gap between what was promoted and what is actually running in production is a primary cause of "it worked in staging" failures. Current service tracks promotions but has no concept of environment state divergence.
**Implementation**: Add `environment_snapshot(environment, artifact_versions: dict[str, str])` to record the current deployed state. Add `drift_report(environment)` comparing the latest snapshot against the last successful promotion records. Return `{drifted: True/False, diverged_services: [...]}`.
**Competitor**: Argo CD application sync status, Flux drift detection, Pulumi resource diff.

---

### I7. Parallel Stage Execution with DAG Scheduling

**Category**: Performance / Pipeline Speed
**Justification**: Linear stage execution serializes work that could run concurrently. Test, lint, and security scan stages are independent — running them in parallel cuts pipeline duration by 40-70%. The current model stores stages as a flat list with no dependency graph.
**Implementation**: Accept stages as `list[dict]` with optional `depends_on: list[str]` field. In `trigger_build`, build a DAG and track `stage_status: dict[str, str]`. Expose `advance_stage(build_id, stage_name, outcome)` which checks all predecessors are complete before marking the stage runnable. Add `build_dag_view(build_id)` returning the DAG as adjacency list.
**Competitor**: GitHub Actions `needs:` syntax, CircleCI workflows, Buildkite parallel steps.

---

### I8. Secret Rotation Integration with Vault

**Category**: Security / Zero-Trust
**Justification**: Pipeline secrets (tokens, credentials) are referenced by `secret_scope` string but never validated for rotation age. Stale secrets are a top attack vector. Pipelines should refuse to run if any secret exceeds the rotation policy.
**Implementation**: Add `secret_policy_register(scope, max_age_days, rotation_required)` and `secret_rotation_check(build_id)`. Before triggering a build, validate that all secrets in `secret_scope` were rotated within `max_age_days`. Return `{compliant: bool, stale_secrets: [...]}`. Fail the build gate if `not compliant`.
**Competitor**: HashiCorp Vault dynamic credentials, AWS Secrets Manager rotation, Doppler secret versioning.

---

### I9. Artifact Provenance with SLSA Level 3 Evidence

**Category**: Supply Chain Security
**Justification**: Signed artifacts alone do not meet SLSA Level 3. You also need: build isolation proof, hermetic build evidence, two-party review, and provenance attestation linking source commit to artifact digest via a non-falsifiable chain. Current `store_artifact` only records `signed: bool`.
**Implementation**: Add `artifact_attest(artifact_id, provenance: dict)` storing builder identity, source commit, build parameters, and material digests. Implement `slsa_level_check(artifact_id) -> int` returning 0-4 based on evidence present. Gate promotions to production at SLSA level >= 2.
**Competitor**: Sigstore/cosign, SLSA GitHub generator, Tekton Chains.

---

### I10. Canary Analysis with Automated Pass/Fail Decision

**Category**: Progressive Delivery
**Justification**: `canary_deploy` sets `canary_pct` but has no mechanism to evaluate canary health. Real canary automation requires comparing error rate, latency, and business metrics between canary and baseline, then automatically promoting or rolling back.
**Implementation**: Add `canary_analysis_submit(deployment_id, metrics: dict[str, float])` storing canary health metrics. Add `canary_evaluate(deployment_id, baseline_metrics: dict[str, float], thresholds: dict[str, float])` computing `{passed: bool, reasons: [...]}`. Auto-trigger `rollback_release` on failure or `promote_canary_to_full` on pass.
**Competitor**: Argo Rollouts canary analysis, Spinnaker kayenta, Flagger.

---

### I11. Build Cache Efficiency Reporting

**Category**: Developer Experience / Cost
**Justification**: Cache hit rate directly controls pipeline speed and cost. Without measuring it, cache policies are guesswork. A 10% improvement in cache hit rate translates to measurable throughput gains across all pipelines.
**Implementation**: Add `cache_event_record(build_id, layer, hit: bool, bytes_saved: int)` and `cache_efficiency_report(pipeline_id)` returning `{hit_rate, bytes_saved_total, estimated_minutes_saved}`. Track per-layer hit/miss ratios. Flag pipelines with `hit_rate < 0.4` as candidates for cache policy review.
**Competitor**: Buildkite caching analytics, Nx Cloud cache hit reporting, Gradle build scan cache statistics.

---

### I12. Policy-as-Code Gate Evaluation Engine

**Category**: Governance / Extensibility
**Justification**: Hard-coded quality thresholds (e.g., `coverage_pct < 70.0`) cannot be customised per pipeline, per environment, or per tenant without code changes. A policy engine allows teams to express gates declaratively and vary them without touching service logic.
**Implementation**: Add `policy_register(name, rules: list[dict])` where each rule is `{field, operator, value, action}`. Add `policy_evaluate(artifact_id, policy_name) -> {passed: bool, violations: [...]}`. Operators: `lt`, `gt`, `eq`, `contains`. Replace hardcoded thresholds in `quality_gate_add` with policy evaluation.
**Competitor**: OPA/Rego, Conftest, Kyverno, GitHub branch protection rules.

---

### I13. Multi-Tenant Pipeline Isolation Audit

**Category**: Security / Compliance
**Justification**: The current `_key(record_id)` prefixing is the only cross-tenant barrier. There is no runtime proof that tenant A cannot read tenant B's data if they share a service instance. An isolation audit method surfaces evidence of proper isolation for compliance reporting.
**Implementation**: Add `isolation_audit() -> _R` that scans all stores and verifies every record's `tenant_id` matches `self.tenant_id`. Return `{compliant: bool, violations: list[str]}` where violations list any records with mismatched tenant IDs. Run this in `health_check` and expose result in the compliance report.
**Competitor**: AWS IAM resource policies, GCP project-level isolation, Harness RBAC at account scope.

---

### I14. Pipeline Performance Regression Detection

**Category**: Build Intelligence / SRE
**Justification**: A pipeline that silently doubles in duration destroys developer throughput without triggering any alert. Comparing current build duration against a rolling baseline catches regressions before they become normalized.
**Implementation**: Store `duration_seconds` on completed builds. Add `build_duration_baseline(pipeline_id, window: int = 10)` returning the P50/P90 of the last N builds. Add `performance_regression_check(build_id) -> {regressed: bool, current_s, baseline_p50, pct_change}`. Alert (via audit event) when `pct_change > 20%`.
**Competitor**: CircleCI insights, Buildkite analytics, DataDog CI pipeline monitoring.

---

### I15. GitOps Reconciliation Loop with Desired-State Diff

**Category**: Deployment Automation / Reliability
**Justification**: Push-based deployments drift. GitOps reconciliation continuously compares declared state (from the pipeline) against observed state (from the environment). Without this loop, the CI/CD system is fire-and-forget rather than continuous delivery.
**Implementation**: Add `desired_state_register(environment, services: dict[str, str])` storing the intended deployed versions. Add `reconcile(environment) -> _R` comparing desired state against `environment_snapshot` records and returning `{in_sync: bool, out_of_sync: list[dict], actions: list[str]}`. Schedule-trigger `trigger_build` for out-of-sync services.
**Competitor**: Argo CD sync, Flux reconciliation, GitLab agent for Kubernetes.
