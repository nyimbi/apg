# ENVM — Environment Management

`envm` is the APG common environment management capability. It provides
tenant-scoped environment inventory, staged promotion pipelines, configuration
drift detection and remediation, secret scope management, compliance checking,
cost tracking, AI-agent governance, and Bytewax stream guardrails for batch
mutation.

The service is dependency-light by design. Deployment providers, live
configuration stores, secret vaults, runtime access checks, monitoring
pipelines, and stream-worker deployments are adapter responsibilities.

## What It Provides

- Environment inventory with owner, stage, region, cloud provider, configuration
  source, RBAC policy, secret-scope policy, and SHA-256 fingerprint.
- Production-change approval guardrails and production locking metadata.
- Promotion paths with source/target environments, deployment links, rollback
  environments, approval state, and atomic promotion runs.
- Configuration drift reports with declared/observed diff, drift percentage,
  review state, and remediation action.
- Compliance checks against named frameworks (CIS, SOC2, etc.).
- Dependency scanning for pre-release packages in environment manifests.
- Secret scopes with policy references, secret references, and access roles.
- Secret injection (vault path only — secret value is never persisted) and
  rotation with full audit trail.
- Cost tracking with per-resource breakdown and analytics aggregation.
- Environment snapshots for point-in-time configuration capture.
- Bulk create / bulk delete with parallel execution via `asyncio.gather`.
- CSV and JSON export for any collection.
- AI ENVM-agent registration (codex, claude_code, opencode, pi).
- Bytewax stream guardrail for batch environment mutation.
- KPI dashboard and environment analytics aggregation.
- Per-environment access audit summarising actors and event types.

## Quick Start

```python
import asyncio
from capabilities.common.envm import EnvmService

service = EnvmService()

async def main():
    # Create environments
    dev = await service.env_create(
        tenant_id="tenant-acme",
        env_id="env-dev",
        name="Development",
        stage="development",
        region="ke-nairobi",
        cloud_provider="aws",
        owner="platform",
        config={"log_level": "debug"},
    )

    prod = await service.env_create(
        tenant_id="tenant-acme",
        env_id="env-prod",
        name="Production",
        stage="production",
        region="ke-nairobi",
        cloud_provider="aws",
        owner="operations",
        rbac_policy="rbac-prod",
        secret_scope_policy="secret-prod",
    )

    # Promote dev to staging
    run = await service.env_promote(
        tenant_id="tenant-acme",
        source_env_id="env-dev",
        target_stage="staging",
        approved_by="ops-lead",
        artifact_ref="sha256:abc123",
    )

asyncio.run(main())
```

## API Reference

| Method | Signature (key params) | Returns | Description |
|--------|------------------------|---------|-------------|
| `env_create` | `tenant_id, env_id, name, stage, region, cloud_provider, owner` | env record | Create and register a new environment |
| `env_clone` | `tenant_id, source_env_id, target_name, target_stage?, override_config?` | env record | Clone with optional config overlay |
| `env_compare` | `tenant_id, env1_id, env2_id` | diff report | Structured config diff between two environments |
| `env_promote` | `tenant_id, source_env_id, target_stage, approved_by, artifact_ref` | promotion record | Promote to next stage (approval required) |
| `env_snapshot` | `tenant_id, env_id, snapshot_label?` | snapshot record | Point-in-time config capture |
| `env_lifecycle` | `tenant_id, env_id, action, actor, reason?` | lifecycle record | `provision \| deprovision \| suspend \| resume` |
| `env_export` | `tenant_id, env_id, fmt?` | `str` | Export env as `json` or `csv` |
| `env_import` | `tenant_id, env_data, owner?` | env record | Import from exported record |
| `config_drift_check` | `tenant_id, env_id, declared, observed` | drift report | Detect key-level config drift |
| `configuration_drift_detection` | `env_id, declared_version, observed_version, changed_items, total_items` | drift report | Count-based drift (compat) |
| `secret_injection` | `env_id, secret_name, value, vault_path, tenant_id, rotation_days?` | secret record | Register vault path reference (value not stored) |
| `secret_rotation` | `tenant_id, env_id, secret_name, new_vault_path, rotated_by` | rotation record | Rotate a secret reference |
| `env_cost_track` | `tenant_id, env_id, period, resource_costs, currency?` | cost record | Record resource cost data |
| `cost_tracking` | `env_id, period, tenant_id, resource_costs?` | cost record | Compat alias |
| `compliance_check_env` | `tenant_id, env_id, framework?` | compliance record | Run framework compliance check |
| `dependency_scan` | `tenant_id, env_id, manifest` | scan record | Flag pre-release packages |
| `env_access_audit` | `tenant_id, env_id` | audit summary | Actors and event-type breakdown |
| `environment_health_check` | `env_id, tenant_id, checks?` | health record | Synthetic per-check health results |
| `create_promotion_path` | `path_id, tenant_id, source_environment_id, target_environment_id, deployment_link, rollback_environment_id, approval_recorded` | path record | Register a promotion path |
| `run_promotion` | `run_id, tenant_id, promotion_path_id, requested_by, artifact_ref, approval_recorded` | run record | Execute a promotion run |
| `register_secret_scope` | `scope_id, tenant_id, environment_id, name, policy_ref, secret_refs, access_roles` | scope record | Register a secret scope |
| `register_envm_agent` | `tenant_id, name, runtime, role, scope, contribution_disclosed?` | agent record | Register an AI environment agent |
| `bulk_create_environments` | `tenant_id, environments: list[dict]` | list of records | Parallel bulk create |
| `bulk_delete_environments` | `tenant_id, env_ids, reason?` | list of results | Soft-delete multiple environments |
| `export_csv` | `tenant_id, collection?` | CSV string | Export any collection to CSV |
| `export_json` | `tenant_id, collection?` | JSON string | Export any collection to JSON |
| `env_analytics` | `tenant_id, period` | analytics dict | Cost + drift + count aggregations |
| `dashboard_summary` | `tenant_id` | KPI dict | Full KPI dashboard snapshot |
| `health_check` | — | health dict | Service health and collection sizes |
| `list_environments` | `tenant_id?` | list | All environments |
| `list_drift_reports` | `tenant_id?` | list | All drift reports |
| `list_promotions` | `tenant_id?` | list | All promotion records |
| `list_promotion_paths` | `tenant_id?` | list | All promotion paths |
| `list_promotion_runs` | `tenant_id?` | list | All promotion runs |
| `list_secret_scopes` | `tenant_id?` | list | All secret scopes |
| `list_audit_events` | `tenant_id?` | list | All audit events |
| `list_health_checks` | `tenant_id?, env_id?` | list | Health check records |
| `list_envm_agents` | `tenant_id?` | list | Registered ENVM agents |
| `list_cost_records` | `tenant_id?, period?` | list | Cost records |

## World-Class Enhancements (v2.0)

The following 15 improvements are specified in `WORLD_CLASS_IMPROVEMENTS.md`
and represent the production-grade roadmap for this capability.

| # | Enhancement | Impact |
|---|-------------|--------|
| 1 | **Persistent Backend Abstraction** — `StoreBackend` protocol with `MemoryBackend`, `PostgresBackend` (asyncpg), `RedisBackend` (aioredis) | Zero data loss on restart; horizontal scaling |
| 2 | **Pydantic v2 Domain Models** — Replace all `dict[str, Any]` I/O with typed models (`EnvironmentRecord`, `DriftReport`, etc.) | Input validation at boundary; OpenAPI schema generation |
| 3 | **Multi-Stage Promotion Gate Checks** — `promotion_gate_check()` evaluates `dependency_scan_pass`, `drift_compliant`, `health_pass`, `compliance_pass`, `approval_present` before promotion | Prevents broken artifacts reaching production |
| 4 | **Automated Drift Remediation** — `drift_remediate(strategy: auto_revert | pin_observed | notify_only)` closes detect-to-fix loop | Reduces MTTR; satisfies SOC2 CC6.8 |
| 5 | **Secret Vault Integration Layer** — `VaultAdapter` protocol with `fetch` / `rotate`; validates vault path at injection time | Catches bad paths at write time; pluggable HashiCorp/AWS backend |
| 6 | **Environment Template Registry** — `template_register()` / `template_instantiate()` with version tracking | Standardises provisioning; enables self-service |
| 7 | **Environment Tag and Label System** — `env_tag()` / `env_list_by_tags()` with `dict[str, str]` tag store | Cost allocation by tag; topology visibility |
| 8 | **Promotion Rollback with Audit Trail** — `rollback_promotion(run_id, rolled_back_by, reason)` reverses promotion atomically | ITIL / SOC2 CC8.1 change management compliance |
| 9 | **Cost Anomaly Detection** — `cost_anomaly_detect(threshold_pct)` computes rolling mean+stddev and emits `cost_anomaly` event | Prevents billing surprises; FinOps automation |
| 10 | **RBAC Policy Enforcement** — `_check_rbac(actor_id, action, env)` enforced at service boundary for promote/deprovision/rotate/create | Service becomes self-defending |
| 11 | **Event Streaming Outbox** — `_Outbox` flushes to pluggable `EventBusAdapter` (Null + Kafka); best-effort, non-blocking | Real-time composition with `moni`, `audl`, `depl` |
| 12 | **Environment Locking and Freeze** — `env_lock()` / `env_unlock()` with `EnvironmentLockedError` on mutating ops | Enforces change-freeze windows; CAB compliance |
| 13 | **Drift History and Trend Analysis** — `drift_trend(window)` computes slope via linear regression, classifies `improving | stable | worsening` | Proactive governance; enables automated escalation |
| 14 | **Multi-Region Replication Metadata** — `env_add_replica(replica_region, replica_role)` / `env_list_replicas()` with per-replica drift/health | Models geo-distributed deployments; DR test automation |
| 15 | **Structured Capability Metrics Endpoint** — `capability_metrics(tenant_id?)` returns Prometheus-compatible `MetricsSnapshot` (promotion rate, drift, rotation compliance, health pass rate, cost variance) | SLO alerting; executive reporting |

## New Methods

### env_clone — Clone with config overlay

```python
# Clone dev to a new staging env with overridden log level
clone = await service.env_clone(
    tenant_id="tenant-acme",
    source_env_id="env-dev",
    target_name="Staging Clone",
    target_stage="staging",
    override_config={"log_level": "warn", "feature_flags": {"new_ui": True}},
)
# clone["cloned_from"] == "env-dev"
```

### env_compare — Structured diff

```python
diff = await service.env_compare(
    tenant_id="tenant-acme",
    env1_id="env-staging",
    env2_id="env-prod",
)
# diff["identical"]  -> False
# diff["differences"] -> ["configuration", "rbac_policy"]
# diff["config_diff"] -> {"log_level": {"env1": "warn", "env2": "error"}}
```

### config_drift_check — Key-level drift detection

```python
report = await service.config_drift_check(
    tenant_id="tenant-acme",
    env_id="env-prod",
    declared={"db_host": "db.internal", "pool_size": 10},
    observed={"db_host": "db.internal", "pool_size": 20, "extra_key": "oops"},
)
# report["drift_percent"] -> 66.67
# report["status"]        -> "review_required"
# report["drifted_keys"]  -> {"extra_key": {...}, "pool_size": {...}}
```

### bulk_create_environments — Parallel provisioning

```python
results = await service.bulk_create_environments(
    tenant_id="tenant-acme",
    environments=[
        {"name": "Env A", "stage": "development", "region": "eu-west-1", "cloud_provider": "gcp", "owner": "team-a"},
        {"name": "Env B", "stage": "test",        "region": "eu-west-1", "cloud_provider": "gcp", "owner": "team-b"},
        {"name": "Env C", "stage": "staging",     "region": "eu-west-1", "cloud_provider": "gcp", "owner": "team-c"},
    ],
)
# All three created concurrently via asyncio.gather
```

### compliance_check_env + dependency_scan — Pre-promotion gate

```python
compliance = await service.compliance_check_env(
    tenant_id="tenant-acme",
    env_id="env-staging",
    framework="SOC2",
)

scan = await service.dependency_scan(
    tenant_id="tenant-acme",
    env_id="env-staging",
    manifest={"fastapi": "0.111.0", "pydantic": "2.7.0", "some-lib": "1.0.0b3"},
)

if compliance["passed"] and scan["passed"]:
    await service.env_promote(
        tenant_id="tenant-acme",
        source_env_id="env-staging",
        target_stage="production",
        approved_by="release-lead",
        artifact_ref="sha256:def456",
    )
```

## AI Agent Registration

AI agents are first-class environment contributors only after registration:

```python
agent = await service.register_envm_agent(
    tenant_id="tenant-acme",
    name="Drift Reviewer",
    runtime="codex",
    role="drift_reviewer",
    scope="review drift reports and recommend remediation actions",
    contribution_disclosed=True,
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
Supported roles: `environment_reviewer`, `promotion_reviewer`, `drift_reviewer`,
`secret_scope_reviewer`, `policy_reviewer`.

## Guardrails

The deterministic rules deny or require review when:

- tenant context is missing;
- environment owner, region, cloud provider, or RBAC policy is missing;
- production change lacks approval evidence;
- promotion lacks a declared path or artifact reference;
- secret scope lacks policy, secret references, or access roles;
- drift exceeds 20% threshold without review;
- an AI ENVM agent is unregistered, unsupported, unscoped, or undisclosed;
- lifecycle state changes lack audit evidence;
- batch environment mutation does not use Bytewax;
- production deprovision is attempted without a reason string.

## Bytewax Batch Mutation

Batch environment mutation must go through the Bytewax event stream:

```python
allowed = await service.validate_batch_environment_mutation("bytewax")
blocked  = await service.validate_batch_environment_mutation("other-stream")

assert allowed["decision"] == "allow"
assert blocked["decision"]  == "deny"
```

Topic: `apg.envm.lifecycle`. State tracked: environments, promotion paths,
promotion runs, drift reports, secret scopes, ENVM agents, audit events.

## Composition

| Attribute | Value |
|-----------|-------|
| Capability ID | `envm` |
| API prefix | `/envm/api/v1` |
| Provided services | environment inventory, promotion, drift detection, secret scopes, environment policy, ENVM agents |
| Required services | `auth`, `conf`, `audl`, `depl`, `keym`, `moni` |
| UI routes | dashboard, environments, promotion, drift, secrets, agents, policies, rules, analytics, audit, settings |
| Theme | `envm_environment_ops` |
| Stream processor | `bytewax` |

## Proof Commands

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/envm/__init__.py \
    capabilities/common/envm/capability_contract.py \
    capabilities/common/envm/models.py \
    capabilities/common/envm/service.py \
    capabilities/common/envm/api.py \
    capabilities/common/envm/views.py \
    capabilities/common/envm/app.py

./.venv/bin/pytest -q capabilities/common/envm/

./.venv/bin/python -c "
import asyncio
from capabilities.common.envm import EnvmService
async def _():
    svc = EnvmService()
    await svc.env_create('tenant-proof', 'env-proof', 'Proof', 'development', 'us-east-1', 'aws', 'system')
    print(await svc.dashboard_summary('tenant-proof'))
asyncio.run(_())
"

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/envm --json
./.venv/bin/apg capabilities publish-plan capabilities/common/envm --json
```

---

© 2025 Datacraft — Nyimbi Odero — www.datacraft.co.ke
