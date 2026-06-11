# Configuration Management — World-Class Improvements

© 2025 Datacraft — www.datacraft.co.ke

15 targeted improvements across correctness, operability, and composability.

---

## 1. Async-Native `ConfService` with `asyncio.Lock` Isolation

**Problem**: `ConfService` mutates shared in-memory dicts synchronously, causing
data races in concurrent async handlers. Any `await` between read and write is
a race window.

**Improvement**: Replace bare dict mutations with `asyncio.Lock`-guarded async
methods. Each logical store gets its own lock so `_records`, `_changes`, and
`_deployments` operations don't serialize each other unnecessarily.

---

## 2. Pluggable Persistence Back-end via `StorageBackend` Protocol

**Problem**: All state lives only in RAM. A process restart wipes every record,
change, and audit event.

**Improvement**: Define a `StorageBackend` protocol with `async get`, `async put`,
`async delete`, and `async list` methods. Ship a `MemoryBackend` (current behavior)
and a `PostgresBackend` that uses asyncpg. `ConfService` accepts a backend at
construction, enabling persistence without changing service logic.

---

## 3. Streaming Audit Events via `asyncio.Queue`

**Problem**: Audit events are accumulated in a dict but never streamed. Consumers
polling `list_audit_events()` face staleness and O(n) scans.

**Improvement**: Add an internal `asyncio.Queue` fed on every `_record_audit` call.
Expose `async subscribe_audit_stream()` that yields events to any consumer (Bytewax
adapter, WebSocket handler, or test harness) without polling.

---

## 4. Tenant-Scoped Feature Flags with Hot-Reload

**Problem**: Feature defaults are hard-coded or require a service restart to change.
Multi-tenant flag overrides are not possible.

**Improvement**: Add `async set_feature_flag(tenant_id, flag, value)` and
`async get_feature_flag(tenant_id, flag, default)` backed by the `StorageBackend`.
Flag reads short-circuit through a `BoundedCache` with TTL, reload on miss, and
expose a `reload_feature_flags(tenant_id)` escape hatch for forced refresh.

---

## 5. Configuration Schema Registry with Forward-Compatibility Validation

**Problem**: Values are stored as raw `Any`. There is no schema contract, so a
`str` can silently overwrite an `int` field across versions.

**Improvement**: Add a `SchemaRegistry` that stores JSON Schema (draft-07) per
configuration key. `create_record` and `request_change` run schema validation
before accepting a value. The registry exposes `register_schema`, `get_schema`,
and `validate_against_schema`, enabling generated applications to assert
structural invariants at the governance boundary.

---

## 6. Configuration Inheritance and Override Chain

**Problem**: `ConfigurationRecord` has a flat environment field. There is no
mechanism for dev → staging → production promotion or fallback to a parent
namespace.

**Improvement**: Add `async resolve_config(tenant_id, key, environment)` that
walks an explicit inheritance chain: session → user → tenant → system → default.
Missing values cascade to the next level. The chain is tenant-configurable so
operators can add custom levels (e.g. `region` between `tenant` and `system`).

---

## 7. Cryptographic Integrity for Audit Evidence

**Problem**: Audit events are plain dicts. Nothing stops a bug or attacker from
retroactively modifying `policy_decision` or `matched_rules` in memory.

**Improvement**: Each `ConfigurationAuditEvent` gets a deterministic BLAKE2b digest
computed over `(id, tenant_id, event_type, actor, decision, matched_rules,
metadata)`. `list_audit_events()` optionally re-verifies digests and flags tampered
entries. For durable backends, the digest is stored alongside the event row.

---

## 8. Rollback Snapshots with Point-in-Time Recovery

**Problem**: `deploy_change` increments version but never persists the previous
value in a recoverable form. Rollback is only a plan string, not executable state.

**Improvement**: Before mutating a `ConfigurationRecord` in `deploy_change`,
snapshot the current value + version + environment into a `ConfigurationSnapshot`
store. Add `async rollback_to_version(tenant_id, record_id, version)` that
restores the snapshot, creates a synthetic change request, and records audit
evidence — making rollback as governed as a forward deploy.

---

## 9. Policy-as-Code with CEL Expression Evaluation

**Problem**: Policies are evaluated by the `evaluate_capability_rules` function
using a fixed rule table. Adding new rules requires code changes and redeploy.

**Improvement**: Allow operators to register custom policies as Common Expression
Language (CEL) strings stored in the `StorageBackend`. Add
`async register_cel_policy(tenant_id, name, expression, action)` and integrate
CEL evaluation into `evaluate`. This enables zero-downtime policy changes and
tenant-specific rule customization without code deployment.

---

## 10. Bulk Configuration Import/Export with Diff Report

**Problem**: Migrating configurations between environments requires record-by-record
API calls. There is no diff view before applying a migration.

**Improvement**: Add `async export_configs(tenant_id, environment, format)` (JSON
or YAML) and `async import_configs(tenant_id, data, dry_run)`. Dry-run mode returns
a structured diff (`added`, `modified`, `removed`) without persisting. Non-dry-run
creates change requests for each modified key, feeding them through the normal
governance workflow.

---

## 11. Configuration Dependency Graph

**Problem**: Capabilities declare dependencies in `cap_spec.md` but
`ConfigurationRecord` values often reference other keys (e.g. a service URL
assembled from host + port). Circular or broken references surface only at runtime.

**Improvement**: Add `async register_dependency(tenant_id, key, depends_on_keys)`
and `async validate_dependency_graph(tenant_id)`. Store the DAG in the backend.
On `deploy_change`, check that all declared dependencies exist and are non-drifted.
Expose `async get_dependency_subgraph(tenant_id, key)` for UI visualization.

---

## 12. Encrypted Secret Rotation with Zero-Downtime Swap

**Problem**: Secrets are stored with `secrets_encrypted: bool` flag but no
rotation mechanism. Rotation currently requires a full deploy cycle.

**Improvement**: Add `async rotate_secret(tenant_id, record_id, new_value,
rotation_strategy)` with strategies `immediate` (swap and audit) and `dual_write`
(write new + keep old valid for a TTL, then remove old). Emit a
`secret_rotation_initiated` audit event with the rotation strategy and expiry
timestamp. Integrates with the existing `contains_secrets` / `secrets_encrypted`
governance checks.

---

## 13. Environment Promotion Pipeline with Gating Rules

**Problem**: Promoting a configuration from dev to staging to production is
manual and ungoverned. There is no enforcement of "tested in staging before
production" invariants.

**Improvement**: Add `async create_promotion_pipeline(tenant_id, stages,
gating_rules)` where `stages` are ordered environments and `gating_rules` are
predicates (e.g. "must have at least 1 approved reviewer per stage"). Add
`async promote_config(tenant_id, record_id, pipeline_id)` that advances the
record to the next stage if the gate passes, recording evidence at each gate.

---

## 14. Real-Time Configuration Health Dashboard Metrics

**Problem**: `governance_summary()` returns raw counts. There is no health signal,
trend, or SLO metric for operators to monitor.

**Improvement**: Add `async compute_health_metrics(tenant_id)` returning:
- `stale_record_count` — records not updated in N days
- `drift_rate_7d` — drift detections per 7 days  
- `mean_change_approval_latency_hours` — from request to approved
- `secret_expiry_count` — secrets within 30 days of rotation deadline
- `policy_violation_rate` — denied operations / total operations

Expose via a `/config/health` REST endpoint and emit on the audit stream.

---

## 15. Composability Contracts via APG Capability Bus

**Problem**: Other capabilities consume `conf` by direct import. There is no
runtime composability contract, so a capability can read another tenant's secrets
if tenant context is not carefully threaded.

**Improvement**: Implement a `ConfBusAdapter` that exposes `conf` operations as
typed messages on the APG capability bus (`conf:read`, `conf:write`,
`conf:approve`, `conf:drift.report`). Bus messages carry tenant context as an
unforgeable token, enforced at the bus layer. Capabilities compose through message
passing rather than direct service calls, making cross-tenant access structurally
impossible.
