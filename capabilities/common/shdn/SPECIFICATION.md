# SHDN Capability Specification

## 1. Identity

- Capability ID: `shdn`
- Name: Shutdown and Lifecycle Control
- Category: common platform capability
- Runtime target: APG Python capability package
- Primary users: platform operators, service owners, release managers, incident commanders, generated application administrators

SHDN gives generated applications a governed lifecycle-control layer for service registration, shutdown planning, drain execution, backup and restore gates, shutdown execution, recovery evidence, AI-assisted lifecycle review, audit trails, and Bytewax event streams.

## 2. Scope

SHDN owns the executable lifecycle for:

- tenant-scoped lifecycle target registration;
- dependency-aware shutdown plan creation;
- drain and quiescence tracking;
- backup snapshot and restore-test evidence;
- shutdown execution with health, approval, actor, and stream gates;
- recovery evidence with post-shutdown health checks;
- governed AI agent composition for lifecycle planning and review;
- policy and audit views for generated applications.

Deployment systems, health check runners, backup engines, schedulers, service meshes, and audit sinks stay behind adapters. The package defines stable contracts and deterministic behavior that those adapters can call.

## 3. Provided Services

- `service_lifecycle`
- `shutdown_orchestration`
- `restart_plans`
- `backup_gates`
- `operational_safety`
- `shdn_agents`

## 4. Required Services

- `moni` for health and operational telemetry
- `hlth` for health-gate evidence
- `bkup` for backup and restore-test evidence
- `audl` for durable audit publication
- `envm` for environment and maintenance-window context

Optional integrations include deployment, logging, CI/CD, service mesh, scheduler, ticketing, and incident-management adapters.

## 5. Domain Model

### Shutdown Target

A shutdown target is a tenant-scoped service, worker, database, queue, tenant app, or integration under lifecycle control. It records owner, environment, criticality, dependencies, drain timeout, health gate reference, and state.

Target states:

- `running`
- `draining`
- `quiesced`
- `snapshot_ready`
- `stopped`
- `recovered`
- `failed`

### Shutdown Plan

A shutdown plan groups one or more targets with reason, owner, rollback reference, restart sequence, maintenance window, optional schedule, and production approval.

Plan states:

- `draft`
- `approved`
- `scheduled`
- `executing`
- `completed`
- `blocked`

### Drain Operation

A drain operation records active sessions, queue depth, status, start timestamp, and completion timestamp for a target in a plan.

### Backup Snapshot

A backup snapshot records backup evidence, restore-test evidence, verification state, and timestamp.

### Shutdown Execution

A shutdown execution records actor, target, plan, status, force-shutdown state, matched rules, and required actions.

### Recovery Record

A recovery record links the lifecycle operation to incident, change, or work-order evidence and post-shutdown health-check evidence.

### SHDN Agent

A SHDN agent is a first-class lifecycle composition element with tenant, name, runtime, role, scope, owner, status, and human approval policy.

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

Agents can prepare plans and review evidence, but critical lifecycle actions require human approval.

### Audit Event

An audit event records tenant, event type, subject, message, actor, severity, metadata, and timestamp.

## 6. Rule Engine

The deterministic rule engine enforces:

- tenant context on every operation;
- service owner on lifecycle target registration;
- dependency context on shutdown planning;
- health gate and backup snapshot before shutdown;
- accountable shutdown actor;
- Bytewax stream routing for shutdown execution and batch lifecycle mutation;
- production approval;
- force-shutdown review;
- incident link and post-shutdown health check on recovery;
- supported agent runtime and role;
- human approval for critical agent-driven lifecycle actions.

Rules return `allow`, `require_review`, or `deny` with required actions.

## 7. Workflows

### Target Registration

1. Register target with tenant, ID, type, owner, environment, criticality, dependencies, drain timeout, and optional health gate.
2. Enforce tenant and owner rules.
3. Emit `target_registered`.

### Plan Creation

1. Create shutdown plan with owner, targets, reason, rollback reference, restart sequence, maintenance window, schedule, and approval.
2. Require approval for production or critical targets.
3. Require dependency context for multi-target planning.
4. Emit `plan_created`.

### Drain and Snapshot

1. Start drain for a target in a plan.
2. Mark `quiesced` only when active sessions and queue depth reach zero.
3. Record backup and restore-test evidence.
4. Emit `drain_started` and `snapshot_recorded`.

### Shutdown Execution

1. Require recorded drain, quiesced state, verified snapshot, actor, health gate, approval, and Bytewax stream.
2. Mark execution `completed` or `blocked` when force-shutdown review is required.
3. Update target and plan state.
4. Emit `shutdown_executed`.

### Recovery

1. Record incident, change, or work-order evidence.
2. Record post-shutdown health-check evidence.
3. Mark target `recovered`.
4. Emit `recovery_recorded`.

### Agent Workflow

1. Register agent with supported runtime and role.
2. Evaluate critical lifecycle action requests.
3. Deny critical agent-driven action without human approval.
4. Emit `shdn_agent_registered`.

## 8. UI Contract

SHDN exposes APG Python view models for:

- `/shdn/dashboard`
- `/shdn/services`
- `/shdn/plans`
- `/shdn/executions`
- `/shdn/approvals`
- `/shdn/recovery`
- `/shdn/agents`
- `/shdn/policy`
- `/shdn/audit`
- `/shdn/settings`

Generated UIs should prioritize queues, state, blockers, required evidence, approvals, health gates, and recovery confidence.

## 9. Theming

The default theme is `shdn_lifecycle_control`. It defines compact density, lifecycle bands, gate chips, health chips, restore chips, review chips, and rule chips.

## 10. Event Stream

SHDN lifecycle events use Bytewax:

- processor: `bytewax`
- stream: `apg.shdn.lifecycle`
- key: `tenant_id`

Events:

- `target_registered`
- `plan_created`
- `drain_started`
- `snapshot_recorded`
- `shutdown_executed`
- `recovery_recorded`
- `shdn_agent_registered`

## 11. Acceptance Criteria

- Contract exposes configuration, schema, rules, UI, theme, services, dependencies, and streaming metadata.
- Service executes registration, planning, drain, snapshot, shutdown, recovery, agent, batch-validation, and audit lifecycles without external dependencies.
- Rules deny invalid tenant, owner, dependency, health, snapshot, actor, stream, approval, recovery, agent, and critical-human-approval states.
- API helpers expose the executable lifecycle.
- View models expose dashboard, service console, plan builder, execution monitor, approvals, recovery, agents, policy, audit, and settings.
- Generated package artifacts reflect the current contract.
- Focused package verification passes.
