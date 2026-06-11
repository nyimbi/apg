# Multi-Tenancy (mten) — World-Class Improvements

**Capability**: `capabilities/common/mten`
**Author**: Nyimbi Odero
**Company**: Datacraft © 2025

---

## 1. Hierarchical Tenant Trees (Sub-Tenants)

Allow tenants to own child tenants, enabling reseller/partner/business-unit topologies. Each child inherits parent quota ceilings and governance rules. Cascading archival prevents orphaned sub-tenants. Resource quota consumption propagates up the tree so the root tenant always has an accurate aggregate view.

**Key additions**: `parent_tenant_id` on `Tenant`, `create_child_tenant()`, `get_tenant_subtree()`, `propagate_quota_event()`.

---

## 2. Quota Ledger with Real-Time Enforcement

Replace the static `ResourceAllocation` snapshot with a double-entry ledger. Every allocation or release is a signed transaction, making overcommit detection O(1) and enabling chargeback reconciliation down to the second. The ledger is append-only and hash-chained for tamper evidence without requiring a full blockchain.

**Key additions**: `QuotaLedger`, `QuotaTransaction`, `reserve_quota()`, `release_quota()`, `get_quota_balance()`.

---

## 3. Policy-as-Code Engine (OPA/Rego Bridge)

Define isolation, feature-flag, and billing rules in Rego policy bundles instead of hardcoded Python. Rules are versioned, testable offline, and hot-reloadable. Tenant tier upgrades and compliance requirement changes require only a policy bundle push, not a service deployment. Supports dry-run evaluation for what-if analysis.

**Key additions**: `PolicyEngine`, `evaluate_policy()`, `reload_policy_bundle()`, `explain_policy_decision()`.

---

## 4. Tenant Namespace Namespacing at the Database Layer

Implement row-level security (RLS) via PostgreSQL `SET LOCAL app.tenant_id` so every query in a transaction is automatically tenant-scoped at the engine level. Eliminates the class of bugs where application code forgets to add a `WHERE tenant_id = $1` clause. Works transparently with SQLAlchemy async sessions.

**Key additions**: `TenantAwareSession`, `set_tenant_rls_context()`, `TenantScopedBase` (SQLAlchemy declarative base with RLS hooks).

---

## 5. Graceful Tier Downgrade with Usage Conflict Resolution

When downgrading a tenant tier, enumerate all resources that exceed the new limits, queue a conflict resolution workflow, and hold the downgrade in a `DOWNGRADE_PENDING` state until the tenant or admin resolves each conflict. Prevents silent data loss or broken UX from immediate enforcement.

**Key additions**: `DowngradeConflict`, `stage_tier_downgrade()`, `resolve_downgrade_conflict()`, `finalize_tier_downgrade()`.

---

## 6. Cross-Tenant Data Masking and Tokenisation

Before any data leaves a tenant's isolation boundary (exports, AI training pipelines, inter-tenant sharing), run it through a reversible tokenisation layer. PII fields are replaced with stable tenant-scoped tokens. Original values are recoverable only with the originating tenant's key. Satisfies GDPR Art. 25 (data protection by design).

**Key additions**: `DataMaskingEngine`, `tokenise_tenant_record()`, `detokenise_tenant_record()`, `register_masking_policy()`.

---

## 7. Event-Driven Tenant Lifecycle Webhooks

Emit CloudEvents-compatible events for every lifecycle transition (created, provisioned, suspended, upgraded, migrated, archived). Consumers subscribe via per-tenant webhook registrations with HMAC-SHA256 signatures, exponential retry with dead-letter queuing, and event schema versioning. Enables zero-code integration with downstream billing, CRM, and notification systems.

**Key additions**: `TenantEventBus`, `emit_lifecycle_event()`, `register_webhook()`, `replay_failed_events()`.

---

## 8. Tenant-Scoped Secret Management

Store API keys, database credentials, and integration tokens in an envelope-encrypted vault keyed per tenant (KMS data-key per tenant). Rotation schedules are tenant-configurable. Secrets are never serialised to logs or audit trails; only opaque references appear in audit records. Satisfies PCI-DSS Req. 3.5.

**Key additions**: `TenantSecretVault`, `store_secret()`, `retrieve_secret()`, `rotate_secret()`, `schedule_rotation()`.

---

## 9. Tenant Activity Fingerprinting and Behaviour Baselining

Build a rolling statistical baseline of each tenant's request patterns (request rate, error rate, data volume, geographic distribution) and compare live traffic against it using z-score anomaly detection. Deviations beyond configurable thresholds trigger alerts before they escalate to incidents. The baseline is persisted across service restarts.

**Key additions**: `ActivityFingerprint`, `update_baseline()`, `score_activity_deviation()`, `get_deviation_report()`.

---

## 10. Immutable Tenant Configuration Snapshots

Before any configuration mutation (tier change, quota adjustment, policy update), persist a JSON snapshot of the pre-mutation state to an append-only audit store. Snapshots are linked by `previous_snapshot_id` forming a full change history. Any configuration can be restored from a snapshot for rollback or forensic auditing.

**Key additions**: `ConfigSnapshot`, `take_config_snapshot()`, `list_config_snapshots()`, `restore_from_snapshot()`.

---

## 11. SLA-Aware Provisioning with Circuit Breaker

Wrap the `_provision_tenant_async` phases in a circuit breaker per phase. If the compute-allocation phase exceeds its SLA budget, the breaker opens and the tenant is put in `PROVISIONING_DEGRADED` state with partial resources, rather than silently failing or blocking indefinitely. SLA budgets are configurable per phase.

**Key additions**: `ProvisioningCircuitBreaker`, `get_provisioning_sla_status()`, `reset_circuit_breaker()`, `set_phase_sla_budget()`.

---

## 12. Cost Attribution and Showback / Chargeback Reports

Attach a unit-cost model to each resource type (CPU-hour, GB-hour, API-call). Accumulate metered usage in a time-series store keyed by `(tenant_id, resource_type, interval)`. Generate showback reports (informational) or chargeback invoices (billable) at configurable cadences. Integrates with the quota ledger (improvement 2) for reconciliation.

**Key additions**: `CostAttributionEngine`, `meter_resource_usage()`, `generate_showback_report()`, `generate_chargeback_invoice()`.

---

## 13. Zero-Downtime Live Tenant Migration with State Sync

Extend the existing `migrate_tenant_cross_cloud` to support live traffic during migration via a dual-write phase. The source continues serving reads/writes; the target receives replicated writes. A final cutover drains in-flight requests with a configurable quiesce window. Rollback restores source as primary within seconds.

**Key additions**: `LiveMigrationCoordinator`, `start_dual_write_phase()`, `quiesce_and_cutover()`, `rollback_migration()`.

---

## 14. Tenant-Scoped Rate Limiting with Token Bucket and Burst Control

Replace the single `api_rate_limit` integer with a token-bucket implementation supporting burst capacity (short spikes allowed up to N×limit), replenish rates, and per-endpoint overrides. Limits are enforced in-process with Redis-backed distributed coordination. Soft limits trigger 429 with `Retry-After`; hard limits enforce circuit-break at the gateway.

**Key additions**: `TenantRateLimiter`, `consume_tokens()`, `get_rate_limit_status()`, `update_rate_limit_config()`.

---

## 15. Tenant Archival with GDPR Right-to-Erasure Execution

Replace the current soft-delete (status → ARCHIVED) with a scheduled erasure pipeline. On archive, a `TenantErasureJob` enumerates all data stores holding tenant data, runs anonymisation or deletion per the tenant's data-retention policy, and issues a cryptographic proof-of-erasure certificate. Retains billing records with PII stripped per regulatory minimums.

**Key additions**: `TenantErasureJob`, `schedule_erasure()`, `execute_erasure_phase()`, `issue_erasure_certificate()`, `get_erasure_status()`.
