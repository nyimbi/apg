# ENVM — World-Class Improvement Plan

**Capability**: Environment Management (`envm`)
**Domain**: `common`
**Author**: Nyimbi Odero — Datacraft © 2025

---

## Overview

The current service is functional and covers the happy path for environment lifecycle, drift detection, secret injection, promotion, and cost tracking. The 15 improvements below address production-grade gaps: correctness, observability, security, composability, and operational excellence.

---

## 1. Persistent Backend Abstraction (Repository Pattern)

**Problem**: `_Store` is a pure in-memory dict. Restart wipes all state. No real database integration.

**Improvement**: Extract a `StoreBackend` protocol with `get / put / list / delete / query`. Ship three implementations: `MemoryBackend` (current, tests), `PostgresBackend` (asyncpg), and `RedisBackend` (aioredis). Service receives a backend at construction time.

```python
class StoreBackend(Protocol):
    async def put(self, collection: str, record: dict) -> dict: ...
    async def get(self, collection: str, record_id: str) -> dict | None: ...
    async def list(self, collection: str, tenant_id: str | None = None,
                   filters: dict | None = None, limit: int = 500) -> list[dict]: ...
    async def delete(self, collection: str, record_id: str) -> bool: ...
```

**Impact**: Production deployability, horizontal scaling, zero data loss on restart.

---

## 2. Pydantic v2 Domain Models for All I/O

**Problem**: Service inputs/outputs are raw `dict[str, Any]`. No schema validation, no IDE completion, no automatic docs.

**Improvement**: Replace all `dict[str, Any]` I/O with typed Pydantic v2 models (`EnvironmentRecord`, `DriftReport`, `PromotionRecord`, etc.) using `model_config = ConfigDict(extra='forbid', validate_by_name=True)`. Keep `.model_dump()` for store serialisation.

**Impact**: Input sanitisation at boundary, serialisation correctness, OpenAPI schema generation, type safety end-to-end.

---

## 3. Multi-Stage Promotion Pipeline with Gate Checks

**Problem**: `env_promote` is a single-step write. No gate checks (tests pass, security scan clean, canary healthy, rollback plan attached).

**Improvement**: Add `promotion_gate_check(tenant_id, env_id, gates: list[str])` that evaluates configured gates before allowing a promotion run. Gates: `dependency_scan_pass`, `drift_compliant`, `health_pass`, `compliance_pass`, `approval_present`. Promotion fails atomically if any gate rejects.

**Impact**: Prevents broken artifacts from reaching production. Composable with `depl` and `moni` capabilities.

---

## 4. Automated Drift Remediation Engine

**Problem**: `config_drift_check` detects drift but never acts on it. Remediation is a field in the report only.

**Improvement**: Add `drift_remediate(tenant_id, env_id, drift_report_id, strategy)` where strategy is `auto_revert | pin_observed | notify_only`. Auto-revert updates the environment config to the declared snapshot. Each action is fully audited and raises a `DriftRemediatedEvent`.

**Impact**: Closes the detect-to-fix loop, reduces MTTR, satisfies SOC2 CC6.8 continuous monitoring.

---

## 5. Secret Vault Integration Layer

**Problem**: `secret_injection` records a vault path but never validates it. The value parameter is accepted but discarded — silent misconfiguration.

**Improvement**: Add a `VaultAdapter` protocol (`async def fetch(path: str) -> str`, `async def rotate(path: str, new_value: str) -> str`). Ship a `MockVaultAdapter` for tests. `secret_injection` calls `vault.fetch(vault_path)` to validate the path exists before persisting the record. `secret_rotation` calls `vault.rotate`.

**Impact**: Catches bad vault paths at injection time. Enables real HashiCorp Vault / AWS Secrets Manager integration without changing service logic.

---

## 6. Environment Template Registry

**Problem**: `provision_environment` accepts a `template_id` string but does nothing with it. Templates are not defined anywhere.

**Improvement**: Add `template_register(tenant_id, template_id, name, stage, cloud_provider, region, default_config, rbac_policy, secret_scope_policy)` and `template_instantiate(tenant_id, template_id, name, override_config)`. Templates are first-class records with version tracking.

**Impact**: Enforces standardisation across environments, reduces configuration sprawl, enables self-service environment provisioning.

---

## 7. Environment Tag and Label System

**Problem**: Environments have no tagging mechanism. Filtering is limited to tenant_id and stage. Grouping by team, cost centre, or service is impossible.

**Improvement**: Add `env_tag(tenant_id, env_id, tags: dict[str, str])` and `env_list_by_tags(tenant_id, tags: dict[str, str])`. Tags are stored as a flat `dict[str, str]` in the environment record. Index maintained in `_Store` for O(n) filtered queries.

**Impact**: Enables cost allocation by tag, targeted bulk operations, service topology visibility.

---

## 8. Promotion Rollback with Audit Trail

**Problem**: Rollback environment ID is recorded in `PromotionPath` but `rollback_promotion(run_id)` does not exist.

**Improvement**: Add `rollback_promotion(tenant_id, promotion_run_id, rolled_back_by, reason)` that reverses the promotion by re-promoting the rollback environment, marks the original run as `rolled_back`, and emits `env_rollback` audit events with `severity=high`.

**Impact**: Operational safety net. Satisfies change management requirements (ITIL, SOC2 CC8.1).

---

## 9. Cost Anomaly Detection

**Problem**: `env_cost_track` records costs but never compares them to baselines or alerts on spikes.

**Improvement**: Add `cost_anomaly_detect(tenant_id, env_id, period, threshold_pct: float = 20.0)` that fetches the last N cost records for the env, computes rolling mean + stddev, flags the current period if it exceeds `mean * (1 + threshold_pct/100)`, and emits a `cost_anomaly` audit event + notification.

**Impact**: Prevents billing surprises, enables FinOps automation, composable with `moni` alerting.

---

## 10. RBAC Policy Enforcement at Service Level

**Problem**: `rbac_policy` is stored as a string field but never enforced. Any caller can promote, deprovision, or rotate secrets.

**Improvement**: Add `RbacPolicy` dataclass and `_check_rbac(actor_id, action, env)` internal method. Policy rules are stored in `envm_rbac_policies`. Actions checked: `promote`, `deprovision`, `secret_rotate`, `env_create`. Raise `PermissionError` on deny.

**Impact**: Service becomes self-defending. Removes dependency on callers knowing to check permissions externally.

---

## 11. Event Streaming Outbox

**Problem**: Audit events are written to `_Store` but never emitted to an external event bus. Downstream capabilities (`moni`, `audl`) cannot react in real time.

**Improvement**: Add `_Outbox` that accumulates events during a request and flushes them to a pluggable `EventBusAdapter` (protocol: `async def publish(topic: str, event: dict) -> None`). Ship `NullEventBusAdapter` (default) and `KafkaEventBusAdapter`. Flush is best-effort, never blocks the primary operation.

**Impact**: Enables event-driven composition with `moni`, `audl`, `depl`. Required for Bytewax pipeline integration.

---

## 12. Environment Locking and Freeze Mechanism

**Problem**: Production environments can be mutated at any time. There is a `production_locked` flag but no enforcement logic.

**Improvement**: Add `env_lock(tenant_id, env_id, locked_by, reason)` / `env_unlock(tenant_id, env_id, unlocked_by, reason)` methods. When locked, all mutating operations (`env_lifecycle`, `secret_injection`, `config_drift_check` with remediation) raise `EnvironmentLockedError`. Lock state is a separate record in `envm_locks`.

**Impact**: Prevents concurrent changes to critical environments, enforces change-freeze windows, satisfies CAB (Change Advisory Board) requirements.

---

## 13. Drift History and Trend Analysis

**Problem**: Drift reports accumulate but there is no way to see whether drift is improving or worsening over time for an environment.

**Improvement**: Add `drift_trend(tenant_id, env_id, window: int = 10)` that returns the last N drift reports sorted by time, computes slope via linear regression on drift percentages, and classifies trend as `improving | stable | worsening`. Returns time-series data suitable for charting.

**Impact**: Proactive governance signal. Enables automated escalation when trend is `worsening` for N consecutive checks.

---

## 14. Multi-Region Replication Metadata

**Problem**: Environment records have a single `region` field. Multi-region deployments, active-active setups, and DR configurations cannot be represented.

**Improvement**: Add `env_add_replica(tenant_id, env_id, replica_region, replica_role: Literal["primary","secondary","dr"])` and `env_list_replicas(tenant_id, env_id)`. Replica metadata is stored in `envm_replicas`. Drift checks and health checks can be run per-replica.

**Impact**: Models real-world geo-distributed deployments, enables DR test automation, supports SLA compliance reporting.

---

## 15. Structured Capability Metrics Endpoint

**Problem**: `health_check` returns collection sizes but no SLI/SLO data. There is no way for an observability stack to scrape meaningful metrics.

**Improvement**: Add `capability_metrics(tenant_id: str | None)` that returns a structured `MetricsSnapshot` with: promotion success rate (last 30 days), mean drift percent per stage, secret rotation compliance rate (secrets rotated within `rotation_days`), environment health pass rate, cost variance coefficient per environment. Format is compatible with Prometheus text exposition.

**Impact**: Enables SLO alerting, capacity planning, and executive reporting without ad-hoc queries.
