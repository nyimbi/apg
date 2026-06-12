# APG Configuration Management Capability (`conf`)

**System-wide configuration store providing centralized, hierarchical configuration management with environment-specific overrides, governance workflows, audit evidence, and real-time updates.**

© 2025 Datacraft — www.datacraft.co.ke

---

## Overview

`conf` is the foundational capability required by all other APG capabilities. It has two primary surfaces:

- **`ConfService`** — dependency-light governance service for APG composition. Manages configuration records, change requests, deployments, drift remediations, agent registration, and batch validation through a policy-evaluated workflow. All operations produce structured audit evidence.
- **`ProductionConfigurationManager`** — full AI-native infrastructure automation layer with predictive analytics, GitOps integration, real-time collaboration, and AI model management.

---

## Features

### Core Governance (`ConfService`)
- **Hierarchical Configuration Scopes**: System → Tenant → User → Session
- **Governed Change Workflow**: `create_record` → `request_change` → `decide_change` → `deploy_change`
- **Drift Detection and Remediation**: Governed `request_drift_remediation` / `decide_drift_remediation` cycle
- **Audit Evidence on Every Operation**: `policy_decision`, `matched_rules`, `review_reasons`, `audit_evidence` stored on all objects
- **Multi-tenant Isolation**: All stores keyed by `(tenant_id, item_id)` — no cross-tenant access
- **Review Queue**: `list_pending_reviews()` surfaces all `review_required` / `pending_review` items in one call
- **Governance Summary**: `governance_summary()` returns counts across all entity types
- **Agent Registration**: AI agents (`codex`, `claude_code`, `opencode`, `pi`) registered and governed via `register_conf_agent()`
- **Batch Validation**: Bytewax stream metadata validated before acceptance via `validate_batch()`

### Production Infrastructure (`ProductionConfigurationManager`)
- **AI-Powered Optimization**: Configuration creation with predictive risk analysis
- **GitOps Integration**: Repository setup, manifest generation, CI/CD pipeline orchestration
- **Real-Time Collaboration**: Multi-user concurrent editing with conflict resolution
- **Deployment Strategies**: Rolling update, blue/green, canary via `DeploymentStrategy`
- **AI Model Configuration**: Register, deploy, and manage AI/ML model configurations as infrastructure
- **Predictive Analytics**: System-wide and per-resource insights
- **Security Integration**: Zero-trust `ConfigurationSecurityLevel` checks on every operation
- **Autonomous Drift Remediation**: AI-generated remediation plans applied without human intervention

### Agent Composition
- AI review agents (Codex, Claude Code, OpenCode, Pi) can inspect, prepare, and recommend configuration changes under human approval guardrails
- Configuration lifecycle events published on Bytewax stream metadata
- All review evidence is durable: `policy_decision`, `matched_rules`, `review_reasons`, `required_actions`

### Local GitOps Execution
When `GitRepository.url` is empty, `conf` initializes a real local Git repository, writes YAML/JSON manifests, commits changed manifest paths, and returns the actual `HEAD` SHA — no live GitHub/GitLab/cloud CI required.

---

## Quick Start

### Governance Workflow (ConfService)

```python
from capabilities.common.conf.service import ConfService

svc = ConfService()

# 1. Create a configuration record
svc.create_record(
    record_id="rec-001",
    tenant_id="acme",
    key="api.rate_limit",
    value=1000,
    environment="production",
    owner="platform-team",
)

# 2. Request a change
svc.request_change(
    change_id="chg-001",
    tenant_id="acme",
    record_id="rec-001",
    target_environment="production",
    requested_by="alice",
    summary="Increase rate limit for Q4 traffic",
    proposed_value=2000,
    validation_passed=True,
    rollback_plan="Revert to 1000 if error rate exceeds 1%",
)

# 3. Approve the change (different reviewer)
svc.decide_change("chg-001", "acme", reviewer="bob", decision="approved", notes="Load test passed")

# 4. Deploy
svc.deploy_change("dep-001", "acme", change_id="chg-001", requested_by="bob")
```

### Production Manager

```python
import asyncio
from capabilities.common.conf.service import create_configuration_manager

async def main():
    mgr = await create_configuration_manager(tenant_id="acme", apg_integrations={})
    await mgr.initialize({})

    resource = await mgr.create_configuration({
        "name": "api-gateway",
        "type": "network",
        "cloud_provider": "aws",
        "created_by": "alice",
        "configuration": {"replicas": 3, "timeout": 30},
    })
    deployment = await mgr.deploy_configuration(resource.id, "production")
    print(deployment.status)

asyncio.run(main())
```

---

## API Reference

### `ConfService` — Governance Methods

| Method | Signature | Description |
|---|---|---|
| `create_record` | `(record_id, tenant_id, key, value, environment, owner, ...)` | Create a governed configuration record |
| `list_records` | `(tenant_id=None)` | List all records, optionally scoped to tenant |
| `request_change` | `(change_id, tenant_id, record_id, target_environment, requested_by, summary, proposed_value, validation_passed, ...)` | Initiate a governed change request |
| `decide_change` | `(change_id, tenant_id, reviewer, decision, notes)` | Approve or reject a change (reviewer != requester enforced) |
| `deploy_change` | `(deployment_id, tenant_id, change_id, requested_by, strategy, ...)` | Deploy an approved change; production requires approval + rollback plan |
| `list_changes` | `(tenant_id=None)` | List all change requests |
| `list_deployments` | `(tenant_id=None)` | List all deployments |
| `request_drift_remediation` | `(remediation_id, tenant_id, record_id, detected_by, drift_summary, remediation_plan)` | Record detected drift and request remediation |
| `decide_drift_remediation` | `(remediation_id, tenant_id, reviewer, decision, notes)` | Approve or reject drift remediation |
| `list_drift_remediations` | `(tenant_id=None)` | List drift remediations |
| `register_conf_agent` | `(agent_id, tenant_id, name, runtime, role, purpose, owner, human_approval_required)` | Register AI configuration agent with governance |
| `validate_batch` | `(tenant_id, record_count, event_stream)` | Validate batch against Bytewax stream policy |
| `list_agents` | `(tenant_id=None)` | List registered agents |
| `list_batches` | `(tenant_id=None)` | List batch validation records |
| `list_audit_events` | `(tenant_id=None)` | List all audit events |
| `list_pending_reviews` | `(tenant_id=None)` | All items in `review_required` or `pending_review` status |
| `governance_summary` | `(tenant_id=None)` | Count summary across all entity types |
| `describe` | `(tenant_id="default")` | Return capability contract |
| `evaluate` | `(context)` | Evaluate capability rules against an operation context |

### `ProductionConfigurationManager` — Infrastructure Methods

| Method | Description |
|---|---|
| `initialize(apg_integrations)` | Boot all subsystems (AI engine, security, GitOps, predictive analytics) |
| `create_configuration(config_data)` | AI-optimized resource creation with security and compliance checks |
| `deploy_configuration(resource_id, target_environment)` | Deploy with AI-generated deployment plan |
| `detect_and_remediate_drift(resource_id)` | Autonomous drift detection and remediation |
| `create_intelligent_template(requirements)` | AI-generated template from business requirements |
| `enforce_policy(policy_id, resource_id)` | Real-time policy enforcement with auto-remediation |
| `natural_language_configuration(nl_request, context)` | NL → configuration conversion |
| `get_predictive_insights(resource_id=None)` | Per-resource or system-wide predictive recommendations |
| `setup_gitops_repository(name, url, branch, ...)` | Register GitOps repository |
| `create_gitops_manifest(resource_id, repository_id, environment, namespace)` | Generate GitOps manifest for a resource |
| `setup_cicd_pipeline(name, repository_id, trigger_events, ...)` | Create CI/CD pipeline |
| `trigger_deployment_pipeline(pipeline_id, commit_sha, ...)` | Fire pipeline execution |
| `create_deployment_plan(resource_id, environment, strategy, ...)` | Plan deployment with `DeploymentStrategy` (rolling/blue-green/canary) |
| `approve_and_deploy(deployment_plan_id, approved_by)` | Gate + execute a deployment plan |
| `register_ai_model(model_config)` | Register AI model as infrastructure resource |
| `deploy_ai_model(model_id, deployment_options)` | Deploy AI model via GitOps |
| `create_ml_pipeline(pipeline_config)` | Create multi-model ML pipeline configuration |
| `get_governed_metrics()` | Full metrics: AI, security, predictive analytics, and operational KPIs |
| `create_collaboration_session(resource_id, owner_id, ...)` | Start real-time collaborative editing session |
| `apply_collaborative_change(session_id, user_id, ...)` | Apply change in collaboration session |

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements across correctness, operability, and composability. See `WORLD_CLASS_IMPROVEMENTS.md` for full rationale.

| # | Enhancement | Impact |
|---|---|---|
| 1 | **Async-Native `ConfService` with `asyncio.Lock` isolation** | Eliminates data races; per-store locks avoid unnecessary serialization |
| 2 | **Pluggable `StorageBackend` Protocol** | `MemoryBackend` (current) and `PostgresBackend` (asyncpg); swap without changing service logic |
| 3 | **Streaming Audit Events via `asyncio.Queue`** | `subscribe_audit_stream()` yields events to any consumer (Bytewax, WebSocket, test harness) |
| 4 | **Tenant-Scoped Feature Flags with Hot-Reload** | `set_feature_flag` / `get_feature_flag` with `BoundedCache` TTL and forced `reload_feature_flags()` |
| 5 | **Configuration Schema Registry** | JSON Schema (draft-07) per key; `create_record` and `request_change` validate before accepting values |
| 6 | **Configuration Inheritance and Override Chain** | `resolve_config(tenant_id, key, environment)` walks: session → user → tenant → system → default; chain is tenant-configurable |
| 7 | **Cryptographic Integrity for Audit Evidence** | BLAKE2b digest over every `ConfigurationAuditEvent`; `list_audit_events()` optionally re-verifies and flags tampered entries |
| 8 | **Rollback Snapshots with Point-in-Time Recovery** | `rollback_to_version(tenant_id, record_id, version)` restores snapshot, creates synthetic change request, records full audit evidence |
| 9 | **Policy-as-Code with CEL Expression Evaluation** | `register_cel_policy(tenant_id, name, expression, action)` — zero-downtime policy changes; per-tenant rule customization without redeploy |
| 10 | **Bulk Import/Export with Diff Report** | `export_configs(tenant_id, environment, format)` (JSON/YAML); `import_configs(tenant_id, data, dry_run)` returns structured diff before applying |
| 11 | **Configuration Dependency Graph** | `register_dependency`, `validate_dependency_graph`, `get_dependency_subgraph` — broken/circular references caught at deploy time |
| 12 | **Encrypted Secret Rotation with Zero-Downtime Swap** | `rotate_secret(tenant_id, record_id, new_value, rotation_strategy)` with `immediate` and `dual_write` (TTL-expiry) strategies |
| 13 | **Environment Promotion Pipeline with Gating Rules** | `create_promotion_pipeline(stages, gating_rules)` enforces "tested in staging before production" with evidence recorded at each gate |
| 14 | **Real-Time Configuration Health Dashboard Metrics** | `compute_health_metrics(tenant_id)` returns stale records, drift rate (7d), approval latency, secret expiry count, policy violation rate |
| 15 | **Composability Contracts via APG Capability Bus** | `ConfBusAdapter` exposes operations as typed bus messages (`conf:read`, `conf:write`, `conf:approve`, `conf:drift.report`); tenant context is an unforgeable token |

---

## New Methods

### 1. Schema Validation (Enhancement #5)

Prevent type drift and silent overwrites by registering a JSON Schema per key before accepting values.

```python
svc = ConfService()

# Register schema once
svc.register_schema(
    tenant_id="acme",
    key="api.rate_limit",
    schema={"type": "integer", "minimum": 1, "maximum": 100000},
)

# create_record and request_change now validate against this schema
# — a string value raises ValueError before the record is stored
svc.create_record(
    record_id="rec-001", tenant_id="acme", key="api.rate_limit",
    value=1000, environment="production", owner="platform-team",
)
```

### 2. Rollback to Version (Enhancement #8)

Restore a configuration record to any prior version without bypassing governance.

```python
# After several deploy_change cycles, version is 4
# Restore to version 2 — creates a synthetic change request + audit trail
rollback = await svc.rollback_to_version(
    tenant_id="acme",
    record_id="rec-001",
    version=2,
)
print(rollback["status"])         # "completed"
print(rollback["audit_evidence"]) # full policy evidence attached
```

### 3. Bulk Import with Dry Run (Enhancement #10)

Migrate configurations between environments; inspect the diff before committing.

```python
export_data = await svc.export_configs(
    tenant_id="acme",
    environment="staging",
    format="yaml",
)

diff = await svc.import_configs(
    tenant_id="acme",
    data=export_data,
    dry_run=True,        # returns diff only, nothing written
)
print(diff["added"])    # list of new keys
print(diff["modified"]) # list of changed keys
print(diff["removed"])  # list of removed keys

# Apply if diff looks correct
await svc.import_configs(tenant_id="acme", data=export_data, dry_run=False)
```

### 4. Audit Stream Subscription (Enhancement #3)

Consume audit events in real time without polling.

```python
import asyncio
from capabilities.common.conf.service import ConfService

svc = ConfService()

async def audit_consumer():
    async for event in svc.subscribe_audit_stream(tenant_id="acme"):
        print(f"{event['event_type']} by {event['actor']} — {event['decision']}")
        if event["event_type"] == "configuration_change_deployed":
            await notify_ops_channel(event)
```

### 5. Configuration Health Metrics (Enhancement #14)

Surface SLO-relevant signals for dashboards and alerting.

```python
health = await svc.compute_health_metrics(tenant_id="acme")

print(health["stale_record_count"])              # records not updated in N days
print(health["drift_rate_7d"])                   # drift detections per 7 days
print(health["mean_change_approval_latency_hours"])
print(health["secret_expiry_count"])             # secrets expiring within 30 days
print(health["policy_violation_rate"])           # denied ops / total ops
```

---

## Configuration Precedence

| Priority | Source |
|---|---|
| 1 (highest) | Runtime / programmatic set |
| 2 | Database-stored (`ConfService` governed records) |
| 3 | Environment variables (`APG_DATABASE_HOST` → `apg.database.host`) |
| 4 | YAML/JSON configuration files |
| 5 (lowest) | Built-in defaults |

---

## Security Considerations

- **Secrets**: Mark with `contains_secrets=True`; enforce `secrets_encrypted=True` or the governance policy will require review
- **Production deployments**: Require explicit `approved` status and a non-empty `rollback_plan` — both enforced structurally in `deploy_change()`
- **Self-review prevention**: `decide_change` rejects a reviewer who is the same actor as the requester
- **Agent privilege**: Agents in `policy_reviewer` or `deployment_reviewer` roles are automatically placed in `pending_review` until a human approves
- **Bus composability**: `ConfBusAdapter` makes cross-tenant secret access structurally impossible — tenant context is carried as an unforgeable token at the bus layer

---

## Local GitOps Execution

```python
from capabilities.common.conf.gitops_integration import GitOpsRepository, GitRepository

repository = GitRepository(
    name="generated-app-config",
    branch="main",
    local_path="/tmp/generated-app-config",
    sync_enabled=False,     # empty url → local-only
)
gitops = GitOpsRepository(repository)

await gitops.clone_or_pull()
await gitops.write_manifest_file(
    "environments/dev/resources/api.yaml",
    {"kind": "Configuration", "spec": {"resources": {"replicas": 2}}},
)
await gitops.commit_and_push(["environments/dev/resources/api.yaml"], "Add API manifest")
commit_sha = await gitops.get_latest_commit_sha()
```

Provider-neutral pull-request evidence is stored under `.apg/pull_requests/<id>.json`. Live GitHub/GitLab adapters can replace that evidence writer without changing generated application behavior.

---

## Integration with Other Capabilities

```python
from capabilities.common.conf.service import ConfService

svc = ConfService()

# Other capabilities consume conf records directly
record = svc.list_records("acme")
db_host = next(r["value"] for r in record if r["key"] == "apg.database.host")
```

Or via the APG Capability Bus (Enhancement #15):

```python
# conf:read message carries tenant context as unforgeable token
bus.publish("conf:read", {"key": "apg.auth.jwt.secret", "tenant_token": token})
```

---

## Testing

```bash
uv run pytest -vxs tests/ci
```

All tests live under `tests/`. No mocks — use pytest fixtures with real `ConfService` instances.

---

## Development & Setup

```bash
# Install in development mode
pip install -e .

# Environment flags
export APG_ENVIRONMENT=development
export APG_CONFIG_DIR=./config
```

---

**Next**: After `conf`, proceed with [Audit Logging (`audl`)](../audl/README.md) per the development order plan.
