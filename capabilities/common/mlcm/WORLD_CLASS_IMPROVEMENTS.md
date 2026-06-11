# MLCM World-Class Improvements

**Capability**: ML Compliance (mlcm)
**Domain**: common
**Author**: Nyimbi Odero
**© 2025 Datacraft**

---

## 1. Native Async Service Layer

**Problem**: All service methods are synchronous. Under async workloads (FastAPI, Starlette, LangGraph pipelines) callers must run the service in a thread executor, adding overhead and masking latency.

**Improvement**: Introduce async-native versions of every write method (`async_register_model`, `async_record_evaluation`, etc.) backed by an async lock. The sync methods become thin wrappers that call `asyncio.get_event_loop().run_until_complete()` when there is no running loop, preserving backward compatibility while enabling zero-copy async paths.

---

## 2. Pluggable Persistent Store Adapter

**Problem**: All state lives in plain Python dicts. A process restart loses every model, version, evaluation, and audit event. There is no PostgreSQL persistence despite CLAUDE.local.md mandating it.

**Improvement**: Define a `MlcmStoreAdapter` abstract base with `put`, `get`, `list_by_tenant`, and `delete` async methods. Ship two concrete implementations: `InMemoryStore` (current behaviour, for tests) and `PostgresStore` (asyncpg). `MlcmService` accepts `store: MlcmStoreAdapter = InMemoryStore()` in its constructor. The Alembic migration already in the repo wires directly to `PostgresStore`.

---

## 3. Regulatory Framework Compliance Profiles

**Problem**: The capability applies a single rule engine to all models regardless of regulatory jurisdiction. EU AI Act, NIST AI RMF, and ISO 42001 impose different evidence requirements.

**Improvement**: Introduce a `compliance_profile` field (`eu_ai_act | nist_ai_rmf | iso_42001 | internal`) on `ModelArtifact`. The rule engine loads profile-specific rule overlays and `record_evaluation` enforces profile-appropriate evidence gates (e.g., EU AI Act requires conformity assessment documentation; NIST requires TEVV evidence).

---

## 4. Model Lineage Graph

**Problem**: Versions have `training_data_ref` and `baseline_ref` scalar strings but there is no way to traverse the full upstream lineage (parent model, dataset versions, feature pipelines, foundation model base).

**Improvement**: Add `async_build_lineage_graph(tenant_id, version_id)` that returns a DAG structure with `nodes` (model, version, dataset, feature-pipeline, base-model) and `edges` (derived_from, trained_on, evaluated_against). The graph is computed from existing records and can be serialised to JSON-LD for interoperability with ML metadata stores.

---

## 5. Continuous Fairness Monitoring

**Problem**: `bias_audit` is a one-shot method that returns synthetic disparities. There is no temporal tracking, no alerting on fairness regression, and no integration with protected attribute definitions stored in a registry.

**Improvement**: Add `async_record_fairness_metric` that stores time-series fairness observations per protected attribute. A companion `async_fairness_regression_check` computes moving-window disparity trend and raises a `FairnessAlert` audit event when disparity exceeds threshold or worsens by more than a configured delta across consecutive windows.

---

## 6. Explainability Evidence Registry

**Problem**: `model_explain` generates one-shot per-sample explanations with no linkage to the evaluation record, no storage of global explanations (population-level feature importance), and no structured evidence that can satisfy audit queries.

**Improvement**: Introduce an `ExplainabilityRecord` dataclass that links `version_id`, `evaluation_id`, `method`, and `global_importances` (dict). `async_record_global_explanation` persists this record and stamps `explainability_recorded=True` on the linked `EvaluationRun`. `async_get_explainability_evidence(version_id)` returns the full explanation evidence chain required for audit.

---

## 7. Policy-as-Code Hot Reload

**Problem**: `capability_contract.py` bakes rule engine configuration into a Python dict that requires a deploy to change. High-risk incidents may require emergency rule tightening within seconds, not hours.

**Improvement**: Add `async_reload_policy(policy_source: str)` that accepts a JSON or YAML policy blob, validates it against the policy schema, atomically replaces the in-memory rule set, and emits a `policy_reloaded` audit event with the diff. The method is gated behind the `privileged_admin` agent role.

---

## 8. Canary Promotion Orchestration

**Problem**: `deploy_model` accepts a `canary_percent` integer but there is no mechanism to progressively increase traffic, monitor canary health, and automatically promote or roll back based on observed metrics.

**Improvement**: Add `async_advance_canary(deployment_id, new_pct, health_check_results)` that validates health_check_results against a configurable acceptance gate (error rate, latency P99, drift score), updates `canary_percent`, and — when `new_pct == 100` — calls `request_promotion` automatically with evidence references from the health checks.

---

## 9. Multi-Tenant Governance Report

**Problem**: `dashboard_summary` is scoped to a single tenant. Operators managing a SaaS platform have no cross-tenant view of compliance posture, unresolved drift counts, or pending reviews.

**Improvement**: `async_governance_report(operator_token)` returns an operator-scoped summary across all tenants: counts of models by risk level, unresolved drift, pending reviews, failed bias audits, and policy violations. Data is returned per-tenant without cross-tenant record exposure; only aggregated counters are shared.

---

## 10. Shadow-Mode Deployment

**Problem**: There is no mechanism to run a new model version in shadow mode (receive production traffic, compute outputs, but not serve them to users) before committing to a canary deployment.

**Improvement**: Add `async_create_shadow_deployment(version_id, target_id, mirror_deployment_id)` that creates a `DeploymentRecord` with `status="shadow"`. A companion `async_record_shadow_observation(shadow_deployment_id, input_hash, shadow_output, live_output, latency_ms)` captures output divergence. `async_shadow_promotion_check` computes divergence rate and gates promotion.

---

## 11. Model Card Completeness Linter

**Problem**: `model_card_complete()` in `lifecycle_runtime.py` is a truthy check on the dict. Operators have no feedback about which required sections are missing (intended use, limitations, training data, evaluation results, ethical considerations).

**Improvement**: `async_lint_model_card(version_id)` runs a structured completeness check against a configurable required-sections list, returns per-section pass/fail with remediation hints, and records a `model_card_linted` audit event. The deployment gate is updated to consult lint results rather than the raw truthy check.

---

## 12. Automated Retraining Trigger

**Problem**: Drift is detected and reviewed, but the workflow stops there. Engineers must manually initiate retraining. Under operational pressure this gap grows, and model degradation continues.

**Improvement**: `async_trigger_retraining(version_id, trigger_reason, approved_by)` evaluates whether unresolved drift signals and evaluation score delta cross a retrain threshold. If so, it creates a `TrainingJobRecord` with `trigger=automatic`, links it to the causal drift signals, and emits a `retraining_triggered` audit event. Human approval is gated by model risk level.

---

## 13. Composable Audit Query Engine

**Problem**: `list_audit_events` returns raw flat dicts with no filtering, no time-range queries, no grouping by event type, and no correlation with subject records. Forensic investigation is impractical at scale.

**Improvement**: `async_query_audit(tenant_id, filters)` accepts a typed `AuditQuery` object (event_types, subject_ids, from_ts, to_ts, min_severity, policy_decisions, page, page_size) and returns paginated, structured results with a `correlation_chain` that links causally related events (e.g., drift_recorded → retraining_triggered → model_evaluated → promoted).

---

## 14. SBOM-Style Model Bill of Materials

**Problem**: There is no structured artifact that captures the full provenance of a deployed model: base model, fine-tuning datasets, framework versions, dependency hashes, training infrastructure, and evaluation data.

**Improvement**: `async_generate_mbom(version_id)` produces a Model Bill of Materials JSON document following a structure analogous to CycloneDX SBOM: components (base model, dataset, framework, hardware), hashes, licenses, known issues, and the deployment targets the version is currently serving. The MBOM is stored as a version attachment and linked from the model card.

---

## 15. Federated Model Registry Bridge

**Problem**: Organisations running multiple APG deployments or mixing APG with MLflow, SageMaker Model Registry, or Vertex AI have no way to import external model metadata, synchronise versions, or federate promotion approvals.

**Improvement**: Define a `FederatedRegistryAdapter` protocol with `async_pull_remote_model(remote_ref)`, `async_push_version(version_id, remote_target)`, and `async_sync_evaluation(remote_eval_ref)`. Ship a reference `MlflowRegistryAdapter` implementation. The adapter is registered with `MlcmService` and surfaces its operations through standard `register_model` / `create_version` / `record_evaluation` calls with `metadata.source="mlflow"` stamped for traceability.
