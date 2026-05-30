# Tenants Legacy Capability Specification

- Capability Name: Tenants Legacy
- Capability ID: `tens`
- Category: common
- Version: 1.0.0

## Purpose

TENS provides executable APG legacy-tenant compatibility and migration governance. It coordinates legacy tenant registration, APG tenant mapping, access-boundary evidence, migration approval and completion, deprecation planning, governed AI agents, audit trails, and Bytewax lifecycle events.

## Provided Services

- `legacy_tenant_registry`
- `tenant_mapping`
- `migration_controls`
- `access_boundaries`
- `deprecation_governance`
- `tens_agents`

## Required Services

- `mten`
- `auth`
- `audl`
- `idfd`
- `usrm`

## Current Runtime

The package exposes `TensService`, API helpers, UI view models, deterministic rules, visual theme metadata, and package publication evidence.

The service can:

- register legacy tenants with lineage and compatibility scope;
- map legacy tenant IDs to APG tenant IDs;
- validate access boundaries, role mappings, isolation evidence, and privileged access review;
- create and complete migration plans;
- record deprecation plans;
- register governed TENS agents;
- validate privileged agent-driven tenant actions;
- validate batch tenant mapping stream routing;
- expose audit events and dashboard summaries.

## Rules

- `tenant_context_required`
- `legacy_tenant_requires_owner`
- `legacy_tenant_requires_source_system`
- `legacy_tenant_requires_compatibility_scope`
- `mapping_requires_validation`
- `mapping_requires_bytewax_stream`
- `migration_requires_approval`
- `migration_requires_rollback_plan`
- `migration_completion_requires_post_validation`
- `migration_completion_requires_bytewax_stream`
- `access_boundary_required`
- `role_mapping_required`
- `isolation_validation_required`
- `privileged_access_review_required`
- `stale_legacy_tenant_requires_review`
- `tens_agent_runtime_supported`
- `tens_agent_role_supported`
- `privileged_agent_mapping_requires_human_approval`
- `batch_tenant_mapping_requires_bytewax`

## UI

TENS exposes route-backed APG Python view models for dashboard, legacy tenant registry, mapping workbench, migration queue, boundary review, deprecation planning, agent workbench, policy center, audit, and settings.

## Theme

TENS uses the `tens_legacy_tenant_migration` theme with compact density, legacy status pills, migration bands, validation chips, approval chips, isolation chips, review lanes, and guardrail chips.

## Event Stream

TENS lifecycle events are described by the Bytewax stream manifest:

- processor: `bytewax`
- stream: `apg.tens.lifecycle`
- key: `tenant_id`

## Detailed Packet

See `SPECIFICATION.md`, `PLAN.md`, and `README.md` for the full lifecycle packet, implementation plan, usage examples, and focused verification commands.
