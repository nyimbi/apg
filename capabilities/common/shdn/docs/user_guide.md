# Shutdown and Lifecycle Control — User Guide

**Capability ID**: `shdn` | **Domain**: `common` | **Version**: `1.1.0`

---

## Overview

SHDN gives APG-generated applications a governed, tenant-scoped runtime for the full service lifecycle: registration, planning, drain, backup snapshot, execution, recovery, and post-shutdown verification.  All state changes produce immutable audit events.  Bytewax is the stream processor; NATS is the real-time event transport.

---

## Installation

```bash
pip install apg-common-shdn
```

---

## Quick Start

```python
import asyncio
from capabilities.common.shdn import ShdnService

service = ShdnService()

# 1. Register the service
target = service.register_service(
    tenant_id="acme",
    target_id="payments-api",
    target_type="service",
    owner="platform-team",
    environment="production",
    dependencies=["payments-db"],
    criticality="critical",
    drain_timeout_seconds=120,
)

# 2. Build a shutdown plan
plan = service.create_shutdown_plan(
    tenant_id="acme",
    name="Payments patch 2026-06-11",
    owner="platform-team",
    target_ids=[target["id"]],
    reason="Security patch CVE-2026-1234",
    rollback_plan_ref="runbook://rollback/payments",
    restart_sequence=["payments-db", "payments-api"],
    approved_by="cto@acme.io",
    maintenance_window_ref="window://mw-2026-06-11",
)

# 3. Drain active sessions
service.start_drain(
    tenant_id="acme",
    plan_id=plan["id"],
    target_id=target["id"],
    active_sessions=42,
    queue_depth=7,
)

# 4. Tick drain progress updates as sessions complete
async def drain_ticker(service, tenant_id, drain_id):
    for sessions in [30, 15, 5, 0]:
        await service.update_drain_progress(
            tenant_id=tenant_id,
            drain_id=drain_id,
            active_sessions=sessions,
            queue_depth=0,
            actor="drain-agent",
        )

# 5. Record backup snapshot
service.record_backup_snapshot(
    tenant_id="acme",
    plan_id=plan["id"],
    target_id=target["id"],
    evidence_ref="backup://payments-api/20260611",
    restore_test_ref="restore-test://payments-api/20260611",
)

# 6. Execute shutdown
service.execute_shutdown(
    tenant_id="acme",
    plan_id=plan["id"],
    target_id=target["id"],
    actor="operator-1",
    health_gate_ref="health://payments-api/pre-shutdown",
)

# 7. Record recovery
service.record_recovery(
    tenant_id="acme",
    plan_id=plan["id"],
    target_id=target["id"],
    actor="operator-1",
    evidence_ref="change://JIRA-4567",
    post_shutdown_health_check_ref="health://payments-api/post-restart",
)
```

---

## Service Registration

`register_service()` creates a `ShutdownTargetRecord` that is the anchor for all subsequent lifecycle operations.

| Parameter | Required | Default | Notes |
|-----------|----------|---------|-------|
| `tenant_id` | yes | — | Tenant scope; all operations are tenant-isolated |
| `target_id` | yes | — | Human-readable service name |
| `target_type` | yes | — | One of `service`, `worker`, `database`, `queue`, `tenant_app`, `integration` |
| `owner` | yes | — | Team or person accountable for this service |
| `environment` | no | `production` | Drives approval requirements |
| `dependencies` | no | `[]` | Names of services this target depends on |
| `criticality` | no | `normal` | One of `low`, `normal`, `high`, `critical` |
| `drain_timeout_seconds` | no | `300` | Maximum seconds to wait for drain completion |
| `health_gate_ref` | no | `None` | Reference to the health probe endpoint |

---

## Shutdown Plans

Plans bind one or more targets into an orchestrated lifecycle operation.

**Required fields for production targets:**

- `rollback_plan_ref` — runbook or automation reference for rollback
- `restart_sequence` — ordered list of service names to restart
- `maintenance_window_ref` — change-management window reference
- `approved_by` — approver identity (mandatory for production/critical targets)

Plan statuses: `draft` → `approved` / `scheduled` → `executing` → `completed` | `blocked`

---

## Drain Operations

### Initial Drain

```python
service.start_drain(
    tenant_id="acme",
    plan_id=plan["id"],
    target_id=target["id"],
    active_sessions=50,
    queue_depth=10,
)
```

Status is set to `quiesced` immediately if both counts are zero, otherwise `draining`.

### Real-Time Progress Updates

```python
await service.update_drain_progress(
    tenant_id="acme",
    drain_id=drain["id"],
    active_sessions=0,
    queue_depth=0,
    actor="drain-agent",
)
# Automatically transitions drain and target to "quiesced"
```

Progress updates emit `drain_progress_updated` events so downstream load balancers can stop routing before the target goes offline.

### Queue Drain

For NATS / Bytewax-backed message queues:

```python
await service.queue_drain(
    tenant_id="acme",
    target_id=target["id"],
    queue_ref="nats://acme/payments.commands",
    actor="operator-1",
    max_drain_seconds=60,
)
```

---

## Signal Handling

Bind OS signal handlers so the deployment adapter can wire SIGTERM/SIGINT to the correct lifecycle sequence:

```python
handlers = await service.install_signal_handlers(
    tenant_id="acme",
    target_id=target["id"],
    actor="platform-ops",
    signals=["SIGTERM", "SIGINT"],
)
# Returns handler_sequence: ["service_drain", "graceful_shutdown"]
```

Supported signals: `SIGTERM`, `SIGINT`, `SIGHUP`, `SIGUSR1`, `SIGUSR2`.

---

## Dependency-Ordered Shutdown

Before executing a multi-target shutdown, compute the safe order:

```python
order = await service.compute_shutdown_order(
    tenant_id="acme",
    plan_id=plan["id"],
)

if order["has_cycles"]:
    raise RuntimeError(f"Cyclic dependency detected: {order['cycles']}")

for target_id in order["order"]:
    # drain and stop each target in computed order
    ...
```

The algorithm uses Kahn's BFS on the reverse dependency graph so leaf services (no dependents) are drained first.

---

## Shutdown Disruption Budget

Prevent runaway automation from shutting down too many instances simultaneously:

```python
await service.set_shutdown_budget(
    tenant_id="acme",
    target_id=target["id"],
    actor="platform-ops",
    max_simultaneous_shutdowns=1,
    window_seconds=300,
)
```

This is the SHDN equivalent of a Kubernetes `PodDisruptionBudget`.

---

## Canary Shutdown Validation

Validate one instance before fleet-wide shutdown:

```python
canary = await service.canary_shutdown_test(
    tenant_id="acme",
    target_id=target["id"],
    canary_instance_ref="pod://payments-api-0",
    actor="operator-1",
    validation_ref="test://canary-drain-2026-06-11",
)

if not canary["canary_passed"]:
    raise RuntimeError("Canary shutdown failed — abort fleet shutdown")
```

Gate your full shutdown plan on `canary_passed: True` to catch state-leakage bugs before they affect all instances.

---

## Emergency Stop

When immediate shutdown is required without drain/snapshot:

```python
await service.emergency_stop(
    tenant_id="acme",
    target_id=target["id"],
    actor="incident-commander",
    reason="Active security incident SEV1",
    override_ref="incident://INC-9999",
)
```

Emergency stops require both `reason` and `override_ref` as governance evidence.

---

## Backup Snapshots

Before execution, record backup evidence and restore-test confirmation:

```python
service.record_backup_snapshot(
    tenant_id="acme",
    plan_id=plan["id"],
    target_id=target["id"],
    evidence_ref="backup://payments-api/20260611",
    restore_test_ref="restore-test://payments-api/20260611",
    verified=True,
)
```

Shutdown execution is blocked unless a verified snapshot exists.

---

## Health Gates

### Pre-Shutdown Final Health Check

```python
health = await service.health_check_final(
    tenant_id="acme",
    target_id=target["id"],
    actor="operator-1",
    probe_ref="probe://payments-api/liveness",
)
assert health["healthy"]
```

### Post-Restart Automated Rollback Watch

```python
# Available via bind_capability_adapter for the hlth capability
await service.bind_capability_adapter(
    tenant_id="acme",
    capability_id="hlth",
    adapter_ref="adapter://hlth-probe/v2",
    actor="platform-ops",
    adapter_config={"probe_timeout_seconds": 5, "failure_threshold": 3},
)
```

---

## Recovery

Recovery records close the lifecycle loop after restart:

```python
service.record_recovery(
    tenant_id="acme",
    plan_id=plan["id"],
    target_id=target["id"],
    actor="operator-1",
    evidence_ref="change://JIRA-4567",          # incident or change reference
    post_shutdown_health_check_ref="health://payments-api/post",
)
```

Both `evidence_ref` and `post_shutdown_health_check_ref` are mandatory.

---

## Rollback

When shutdown cannot proceed, roll back in-flight operations:

```python
await service.rollback_inflight(
    tenant_id="acme",
    plan_id=plan["id"],
    target_id=target["id"],
    actor="operator-1",
    rollback_evidence_ref="runbook://rollback/payments/exec-1",
)
# Target state returns to "active"; plan status returns to "approved"
```

---

## Maintenance Mode

```python
# Enter
await service.maintenance_mode(
    tenant_id="acme",
    target_id=target["id"],
    actor="operator-1",
    window_ref="window://mw-2026-06-11",
    expires_at="2026-06-11T04:00:00Z",
)

# Exit
await service.maintenance_exit(
    tenant_id="acme",
    target_id=target["id"],
    actor="operator-1",
)
```

---

## Post-Shutdown Report

```python
report = await service.shutdown_report(
    tenant_id="acme",
    plan_id=plan["id"],
)
# Contains: executions, recoveries, drains, snapshots, plan status
```

---

## Immutable Audit Trail

### Anchor

```python
anchor = await service.anchor_audit_chain(
    tenant_id="acme",
    actor="compliance-auditor",
)
# chain_root is a SHA-256 hash over all audit events in chronological order
```

### Verify

```python
result = await service.verify_audit_chain(
    tenant_id="acme",
    anchor_id=anchor["id"],
    actor="compliance-auditor",
)
assert result["valid"], "AUDIT CHAIN TAMPERED"
```

This satisfies SOC 2 CC7.2 and ISO 27001 A.12.4 tamper-evidence requirements.

---

## AI Agent Composition

```python
agent = service.register_shdn_agent(
    tenant_id="acme",
    name="Shutdown reviewer",
    runtime="claude_code",
    role="shutdown_reviewer",
    scope="review critical shutdown gates",
    owner="platform-ops",
)

decision = service.validate_agent_lifecycle_action(
    tenant_id="acme",
    agent_id=agent["id"],
    target_criticality="critical",
    human_approval_recorded=True,
)
# decision["decision"] == "allow" only when human_approval_recorded=True for critical targets
```

---

## Capability Adapter Binding

Bind live adapters for the capabilities SHDN depends on:

```python
for cap, ref in [
    ("hlth", "adapter://hlth-probe/v1"),
    ("moni", "adapter://moni-metrics/v1"),
    ("bkup", "adapter://bkup-s3/v1"),
    ("audl", "adapter://audl-pg/v1"),
    ("envm", "adapter://envm-config/v1"),
]:
    await service.bind_capability_adapter(
        tenant_id="acme",
        capability_id=cap,
        adapter_ref=ref,
        actor="platform-ops",
    )
```

---

## Dashboard Summary

```python
summary = service.dashboard_summary(tenant_id="acme")
# Returns: target_count, production_target_count, active_plan_count,
#          shutdown_count, recovery_count, audit_event_count, streaming manifest
```

---

## Analytics

```python
analytics = await service.shutdown_analytics(tenant_id="acme")
# Returns: stopped/active/maintenance targets, completed/blocked plans,
#          emergency stop count, rollback count, checkpoint count
```

---

## Event Stream

All state changes emit structured events to the Bytewax stream `apg.shdn.lifecycle` and NATS subjects `apg.shdn.<event_type>.<tenant_id>`.

Key events:

| Event | Trigger |
|-------|---------|
| `target_registered` | `register_service()` |
| `plan_created` | `create_shutdown_plan()` |
| `drain_started` | `start_drain()` |
| `drain_progress_updated` | `update_drain_progress()` |
| `snapshot_recorded` | `record_backup_snapshot()` |
| `shutdown_executed` | `execute_shutdown()` |
| `recovery_recorded` | `record_recovery()` |
| `emergency_stop_executed` | `emergency_stop()` |
| `signal_handlers_installed` | `install_signal_handlers()` |
| `shutdown_order_computed` | `compute_shutdown_order()` |
| `shutdown_budget_set` | `set_shutdown_budget()` |
| `audit_chain_anchored` | `anchor_audit_chain()` |
| `audit_chain_verified` | `verify_audit_chain()` |
| `canary_shutdown_tested` | `canary_shutdown_test()` |
| `capability_bound` | `bind_capability_adapter()` |

---

## Rule Engine Reference

| Rule | Condition | Effect |
|------|-----------|--------|
| `tenant_context_required` | `tenant_context_present=False` | deny |
| `service_requires_owner` | `register_service` + no owner | deny |
| `shutdown_requires_health_gate` | `execute_shutdown` + no health gate | deny |
| `shutdown_requires_backup_snapshot` | `execute_shutdown` + no snapshot | deny |
| `shutdown_requires_bytewax_stream` | `execute_shutdown` + wrong stream | deny |
| `production_shutdown_requires_approval` | production/critical + no approval | deny |
| `force_shutdown_requires_review` | `force_shutdown=True` + no review | require_review |
| `critical_agent_shutdown_requires_human_approval` | agent action on critical + no human approval | deny |
| `batch_lifecycle_mutation_requires_bytewax` | batch + wrong stream | deny |

---

## Composability

Reference SHDN in APG source files:

```apg
use shdn;
```

SHDN requires: `moni`, `hlth`, `bkup`, `audl`, `envm`

SHDN provides: `service_lifecycle`, `shutdown_orchestration`, `restart_plans`, `backup_gates`, `operational_safety`, `shdn_agents`

---

## Further Reading

- `service.py` — Business logic (37 public methods)
- `lifecycle_runtime.py` — Data records and runtime primitives
- `capability_contract.py` — Rule engine and configuration schema
- `models.py` — Re-exports for external consumers
- `api.py` — REST endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural improvement proposals
