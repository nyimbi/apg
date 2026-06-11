# SHDN - Shutdown and Lifecycle Control

SHDN is the APG capability for governed service lifecycle control. It gives generated applications a composable runtime for registering lifecycle targets, building shutdown plans, draining services, enforcing backup and health gates, executing shutdowns, recording recovery evidence, composing AI-assisted review, and emitting Bytewax lifecycle events.

Use SHDN when an application needs safe start, drain, shutdown, restart, recovery, or retirement workflows for tenant services and their dependencies.

## What It Provides

- Tenant-scoped lifecycle target registry.
- Shutdown plan builder with rollback, restart sequence, maintenance window, and approval gates.
- Drain tracking with active-session, queue-depth evidence, and real-time progress updates.
- Backup snapshot and restore-test evidence.
- Shutdown execution with health, snapshot, actor, approval, force-review, and Bytewax stream gates.
- Recovery records with incident/change evidence and post-shutdown health checks.
- First-class SHDN agents for Codex, Claude Code, OpenCode, and Pi based review lanes.
- APG Python UI view models for dashboard, services, plans, executions, approvals, recovery, agents, policy, audit, and settings.
- Visual theme tokens for lifecycle-control screens.
- Bytewax stream metadata for lifecycle events.
- NATS-ready event publishing subjects for real-time fan-out.
- Dependency-ordered shutdown sequencing (topological sort).
- Shutdown disruption budgets (PDB equivalent).
- Immutable audit chain with SHA-256 Merkle anchoring.
- Canary shutdown validation before fleet-wide operations.
- OS signal handler binding records (SIGTERM/SIGINT).
- Capability adapter binding for hlth, moni, bkup, audl, envm composition.

## Core Runtime

```python
from capabilities.common.shdn import ShdnService
import asyncio

service = ShdnService()

target = service.register_service(
    tenant_id="tenant-a",
    target_id="billing-api",
    target_type="service",
    owner="platform-owner",
    environment="production",
    dependencies=["payments-db", "invoice-queue"],
    criticality="critical",
)

plan = service.create_shutdown_plan(
    tenant_id="tenant-a",
    name="Billing maintenance",
    owner="platform-owner",
    target_ids=[target["id"]],
    reason="Patch database driver",
    rollback_plan_ref="runbook://rollback/billing",
    restart_sequence=["payments-db", "invoice-queue", "billing-api"],
    approved_by="ops-director",
    maintenance_window_ref="window://mw-2026-05-30",
)

service.start_drain(
    tenant_id="tenant-a",
    plan_id=plan["id"],
    target_id=target["id"],
    active_sessions=0,
    queue_depth=0,
)

service.record_backup_snapshot(
    tenant_id="tenant-a",
    plan_id=plan["id"],
    target_id=target["id"],
    evidence_ref="backup://billing-api/1",
    restore_test_ref="restore-test://billing-api/1",
)

service.execute_shutdown(
    tenant_id="tenant-a",
    plan_id=plan["id"],
    target_id=target["id"],
    actor="operator-1",
    health_gate_ref="health://billing-api/pre",
)

service.record_recovery(
    tenant_id="tenant-a",
    plan_id=plan["id"],
    target_id=target["id"],
    actor="operator-1",
    evidence_ref="change://123",
    post_shutdown_health_check_ref="health://billing-api/post",
)
```

## Advanced Async Methods

### Real-Time Drain Progress

```python
drain = service.start_drain(...)
# tick updates as sessions complete
progress = await service.update_drain_progress(
    tenant_id="tenant-a",
    drain_id=drain["id"],
    active_sessions=12,
    queue_depth=3,
    actor="drain-agent",
)
# transitions to quiesced automatically when both reach zero
```

### Dependency-Ordered Shutdown

```python
order = await service.compute_shutdown_order(
    tenant_id="tenant-a",
    plan_id=plan["id"],
)
# order["order"] lists target IDs leaf-first (safest drain order)
# order["cycles"] is non-empty if a cyclic dependency is detected
```

### Shutdown Disruption Budget

```python
await service.set_shutdown_budget(
    tenant_id="tenant-a",
    target_id=target["id"],
    actor="platform-ops",
    max_simultaneous_shutdowns=1,
    window_seconds=300,
)
```

### OS Signal Handlers

```python
handlers = await service.install_signal_handlers(
    tenant_id="tenant-a",
    target_id=target["id"],
    actor="platform-ops",
    signals=["SIGTERM", "SIGINT"],
)
# handler_sequence: ["service_drain", "graceful_shutdown"]
```

### Canary Shutdown Validation

```python
canary = await service.canary_shutdown_test(
    tenant_id="tenant-a",
    target_id=target["id"],
    canary_instance_ref="instance://billing-api/pod-0",
    actor="operator-1",
    validation_ref="test://canary-drain-2026-06-11",
)
assert canary["canary_passed"]
```

### Immutable Audit Chain

```python
anchor = await service.anchor_audit_chain(tenant_id="tenant-a", actor="auditor")
# later — verify no records were tampered with
result = await service.verify_audit_chain(
    tenant_id="tenant-a",
    anchor_id=anchor["id"],
    actor="auditor",
)
assert result["valid"]
```

### Capability Adapter Binding

```python
await service.bind_capability_adapter(
    tenant_id="tenant-a",
    capability_id="hlth",
    adapter_ref="adapter://hlth-probe/v1",
    actor="platform-ops",
    adapter_config={"probe_timeout_seconds": 5},
)
```

## AI Agent Composition

SHDN treats lifecycle agents as governed composition elements.

```python
agent = service.register_shdn_agent(
    tenant_id="tenant-a",
    name="Shutdown reviewer",
    runtime="codex",
    role="shutdown_reviewer",
    scope="review critical shutdown gates before execution",
    owner="platform-owner",
)

decision = service.validate_agent_lifecycle_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    target_criticality="critical",
    human_approval_recorded=False,
)

assert decision["decision"] == "deny"
```

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `lifecycle_planner`
- `shutdown_reviewer`
- `dependency_reviewer`
- `recovery_reviewer`
- `approval_reviewer`
- `audit_reviewer`

## Rule Engine

The deterministic rule engine protects lifecycle operations:

- tenant context is mandatory;
- lifecycle targets require owners;
- shutdown plans require dependency context;
- shutdown execution requires health gate, verified backup, actor, approval, and Bytewax event routing;
- force shutdown requires review;
- recovery requires incident/change evidence and post-shutdown health evidence;
- agents require supported runtime and role;
- critical agent-driven lifecycle actions require human approval;
- batch lifecycle mutation requires Bytewax coordination.

Rules are exposed through `evaluate_capability_rules()` and `ShdnService.evaluate()`.

## UI Surfaces

`views.py` exposes route-backed models for:

- dashboard: `/shdn/dashboard`
- services: `/shdn/services`
- plans: `/shdn/plans`
- executions: `/shdn/executions`
- approvals: `/shdn/approvals`
- recovery: `/shdn/recovery`
- agents: `/shdn/agents`
- policy: `/shdn/policy`
- audit: `/shdn/audit`
- settings: `/shdn/settings`

These models are framework-neutral so APG generated Python applications can compose them into their UI shell.

## Event Stream

SHDN publishes lifecycle metadata for Bytewax / NATS:

- processor: `bytewax`
- stream: `apg.shdn.lifecycle`
- key: `tenant_id`
- NATS subjects: `apg.shdn.<event_type>.<tenant_id>`

Events:

- `target_registered`
- `plan_created`
- `drain_started`
- `drain_progress_updated`
- `snapshot_recorded`
- `shutdown_executed`
- `recovery_recorded`
- `shdn_agent_registered`
- `signal_handlers_installed`
- `shutdown_order_computed`
- `shutdown_budget_set`
- `audit_chain_anchored`
- `audit_chain_verified`
- `canary_shutdown_tested`
- `capability_bound`

## Adapter Boundaries

The package does not directly call live deployment systems, backup engines, schedulers, service meshes, health probes, ticketing tools, or audit sinks. Add those integrations as adapters around the stable service methods and stream metadata. Use `bind_capability_adapter()` to record live adapter wiring for composability tracing.

## Verification

Battery-conscious package verification:

```bash
./.venv/bin/python -m py_compile capabilities/common/shdn/__init__.py capabilities/common/shdn/capability_contract.py capabilities/common/shdn/models.py capabilities/common/shdn/lifecycle_runtime.py capabilities/common/shdn/service.py capabilities/common/shdn/api.py capabilities/common/shdn/views.py capabilities/common/shdn/app.py capabilities/common/shdn/test_capability_contract.py capabilities/common/shdn/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/shdn/test_capability_contract.py capabilities/common/shdn/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/shdn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/shdn --json
```

Run broader checks only when battery and time allow.
