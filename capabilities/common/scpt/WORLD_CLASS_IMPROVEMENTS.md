# World-Class Improvements — Custom Scripting Engine (scpt)

**Capability**: `scpt` | **Domain**: `common` | **Date**: 2026-06-11

---

## 1. Async-Native Service Layer

All service methods are currently synchronous. Production workloads involving execution dispatch, batch validation, and agent registration need non-blocking I/O. Refactoring to `async def` throughout allows integration with asyncio event loops, FastAPI, and async Bytewax pipeline hooks without thread-pool bridging.

**Impact**: Removes the #1 throughput bottleneck in CI/workflow runner contexts.

---

## 2. Script Versioning with Immutable History

`ScriptDefinition` tracks a single `version: int` field but provides no mechanism to create a new version of an existing script while preserving the prior snapshot. Add `script_version_create(tenant_id, script_id, new_source, actor)` that clones the current definition, increments version, resets state to `draft`, and links to a `previous_version_id`. Queries for `script_history(tenant_id, script_id)` return the full lineage.

**Impact**: Enables safe rollback, diff-based review, and change-set audit trails.

---

## 3. Script Cloning and Template Instantiation

Published scripts should be first-class templates. A `script_clone(tenant_id, script_id, new_name, owner, actor)` method creates a new draft from the source of any published script, preserving policy bindings and tags. Combined with `template_library`, this closes the loop on template discovery → instantiation → customization.

**Impact**: Dramatically reduces time-to-first-working-script for new tenants.

---

## 4. Execution Replay

Allow re-running a prior execution with the exact same `input_payload` and sandbox, with a `replay_execution(tenant_id, execution_id, actor)` method. The replay records a `replayed_from` reference on the new execution for traceability. Useful for debugging intermittent failures, demonstrating determinism, and regression testing.

**Impact**: Turns the execution log into a reproducibility substrate.

---

## 5. Composite Health Score

`dashboard_summary` returns raw counts. Add a `health_score(tenant_id)` method that computes a 0–100 composite metric weighting: published script ratio, execution success rate, sandbox availability, pending review queue depth, and blocked/failed lifecycle batches. Returns per-dimension breakdown alongside the scalar.

**Impact**: Single glanceable KPI for SRE dashboards and alerting thresholds.

---

## 6. Permission Diff on Script Update

When source changes, re-derive `dangerous_permissions` and emit an audit event only if the permission surface changes. A `permissions_diff(tenant_id, script_id, new_source)` helper returns `added`, `removed`, and `unchanged` permission sets. Integrates into the review workflow so reviewers see only what changed.

**Impact**: Eliminates re-review noise for cosmetic/logic-only changes.

---

## 7. Cron Schedule Registry

`script_schedule` currently embeds schedule metadata as a tag string (`cron:<expr>`). Promote schedules to a first-class `ScriptSchedule` model with fields for `cron_expr`, `timezone`, `enabled`, `last_fired_at`, `next_fire_at`, `failure_count`, and `max_failures`. Add `schedule_enable`, `schedule_disable`, and `schedule_list` methods.

**Impact**: Enables production scheduler adapters to query and mutate cron state through a stable contract.

---

## 8. Runtime Variable Schema Validation

`variable_inject` accepts an arbitrary `dict`. Add an optional `variable_schema: dict` on `ScriptDefinition` (JSON Schema subset) and a `validate_variables(tenant_id, script_id, variables)` method that checks types and required keys before injection. Schema violations produce structured errors, not runtime crashes.

**Impact**: Shifts variable errors left from execution to injection time.

---

## 9. Execution Quota Enforcement

Track per-tenant execution counts with configurable daily/monthly caps stored in `ScriptPackagePolicy` or a new `ScriptTenantQuota` model. A `check_execution_quota(tenant_id, actor)` method returns `{allowed, quota_remaining, resets_at}`. The `execute_script` path calls this before dispatching. When exhausted, returns a policy-denied result with `quota_exceeded` reason.

**Impact**: Prevents runaway automation from starving shared sandbox infrastructure.

---

## 10. Structural AST Complexity Scoring

Extend `script_lint` with a `complexity_score` field. Parse Python ASTs and count branching nodes (if/for/while/try/with/lambda/comprehension). Score > configurable threshold triggers a warning-level lint result and logs an audit event. Integrates into CI gates and review workflows.

**Impact**: Identifies scripts that are operationally risky before they reach production.

---

## 11. Cross-Tenant Script Import (Governed)

Allow a tenant to import a published script from another tenant through a `script_import(tenant_id, source_tenant_id, script_id, actor, policy_override_ref)` method. The import creates a read-only copy in the importing tenant's registry, linked to the source via `imported_from_id`. Access is gated by the source script's `shared_with` list. Imports cannot be published without review.

**Impact**: Enables platform-wide script marketplace without breaking tenant isolation.

---

## 12. Execution Chain Orchestration

Scripts frequently call each other in pipelines. Add `execution_chain_create(tenant_id, steps, requested_by)` where `steps` is an ordered list of `{script_id, input_mapping}`. The service creates a `ScriptExecutionChain` record, fires step 1, and stores the chain state. Each step's output becomes the next step's input via the mapping. Chain-level cancellation, failure, and timeout policies apply.

**Impact**: Eliminates hand-rolled glue scripts for sequential automation patterns.

---

## 13. Policy Simulation (Dry-Run Evaluate)

Add `simulate_policy(tenant_id, operation, context_overrides)` that runs `evaluate_capability_rules` with a merged context without side effects or audit events. Returns the full decision, matched rules, and required actions. Useful for UI preflight checks (e.g., "would this script be denied if I add a network permission?") and integration test scaffolding.

**Impact**: Removes trial-and-error from policy configuration by operators.

---

## 14. Scripting Agent Activity Feed

`audit_events` returns all events. Add `agent_activity_feed(tenant_id, agent_id, limit)` that filters to events produced by or attributed to a specific scripting agent, ordered by `created_at desc`, with enriched context from the script/execution it touched. Drives agent-specific observability panels in the workbench UI.

**Impact**: Makes agent contribution legible without grep-ing the full audit log.

---

## 15. Bulk Script Import / Export

Add `scripts_export(tenant_id, script_ids, include_policies, include_sandboxes)` that serializes the requested scripts (and optionally their policies/sandboxes) to a portable JSON bundle, and `scripts_import(tenant_id, bundle, owner, actor)` that reconstructs them as drafts requiring review. Checksums in the bundle prevent tampering. Drives backup, migration, and environment promotion workflows.

**Impact**: Closes the last gap in the DevOps lifecycle — scripts remain trapped in the database today.
