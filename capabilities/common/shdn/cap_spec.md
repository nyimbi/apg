# Shutdown and Lifecycle Control Capability Specification

- Capability Name: Shutdown and Lifecycle Control
- Capability ID: `shdn`
- Category: common
- Version: 1.0.0

## Purpose

SHDN provides executable APG lifecycle control for generated applications. It coordinates service lifecycle targets, shutdown planning, drain and quiescence, backup and restore gates, shutdown execution, recovery evidence, governed AI agents, audit trails, and Bytewax lifecycle events.

## Provided Services

- `service_lifecycle`
- `shutdown_orchestration`
- `restart_plans`
- `backup_gates`
- `operational_safety`
- `shdn_agents`

## Required Services

- `moni`
- `hlth`
- `bkup`
- `audl`
- `envm`

## Current Runtime

The package exposes `ShdnService`, API helpers, UI view models, deterministic rules, visual theme metadata, and package publication evidence.

The service can:

- register lifecycle targets;
- create shutdown plans with rollback, restart sequence, maintenance window, and approval gates;
- drain targets and track quiescence;
- record backup snapshot and restore-test evidence;
- execute shutdowns with health, snapshot, actor, approval, force-review, and Bytewax stream guardrails;
- record recovery evidence;
- register governed SHDN agents;
- validate critical agent-driven lifecycle actions;
- validate batch lifecycle mutation stream routing;
- expose audit events and dashboard summaries.

## Rules

- `tenant_context_required`
- `service_requires_owner`
- `service_requires_dependency_map`
- `shutdown_requires_health_gate`
- `shutdown_requires_backup_snapshot`
- `shutdown_requires_actor`
- `shutdown_requires_bytewax_stream`
- `production_shutdown_requires_approval`
- `force_shutdown_requires_review`
- `recovery_requires_post_health_check`
- `recovery_requires_incident_link`
- `shdn_agent_runtime_supported`
- `shdn_agent_role_supported`
- `critical_agent_shutdown_requires_human_approval`
- `batch_lifecycle_mutation_requires_bytewax`

## UI

SHDN exposes route-backed APG Python view models for dashboard, service console, plan builder, execution monitor, approvals, recovery center, agent workbench, policy center, audit, and settings.

## Theme

SHDN uses the `shdn_lifecycle_control` theme with compact density, lifecycle bands, gate chips, health chips, restore chips, review chips, and rule chips.

## Event Stream

SHDN lifecycle events are described by the Bytewax stream manifest:

- processor: `bytewax`
- stream: `apg.shdn.lifecycle`
- key: `tenant_id`

## Detailed Packet

See `SPECIFICATION.md`, `PLAN.md`, and `README.md` for the full lifecycle packet, implementation plan, usage examples, and focused verification commands.
