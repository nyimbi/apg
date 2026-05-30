# PLFD Platform Foundation Specification

## Purpose

PLFD is APG's common platform foundation capability. It lets generated and
composed applications register foundation services, map dependencies, attach
required baselines, assess readiness, approve platform changes, and operate
foundation governance workflows through APG UI and API surfaces.

The capability is designed for executable applications first. It provides a
dependency-light runtime and explicit adapter boundaries so production systems
can connect real configuration stores, tenant registries, identity providers,
audit stores, monitoring systems, health checks, security review systems,
plugin registries, and Bytewax workers later.

## Capability Identity

- Capability id: `plfd`
- Display name: `Platform Foundation`
- Category: `common`
- Runtime target: `python`
- Primary service: `service.PlfdService`
- UI prefix: `/plfd`
- API prefix: `/plfd/api/v1`
- Event stream processor: `bytewax`

## Provided Services

- `foundation_registry`
- `dependency_posture`
- `configuration_baselines`
- `readiness_gates`
- `platform_governance`
- `plfd_agents`

## Required Capabilities

- `conf` for configuration policy.
- `mten` for tenant baseline requirements.
- `auth` for identity, permissions, and RBAC.
- `audl` for durable audit evidence.

Optional adapters include `moni`, `hlth`, `regy`, `secu`, and `plgn`.

## Domain Model

`FoundationService`

- Tenant-local service id, name, owner, tier, dependency declarations,
  readiness score, configuration baseline status, health, monitoring state,
  rollback plan, change window, lifecycle status, metadata, and timestamps.

`FoundationDependency`

- Dependency edge with tenant, source service, target service, health, required
  flag, evidence reference, and creation time.

`FoundationBaseline`

- Baseline record with tenant, service id, baseline type, evidence reference,
  approver, status, and creation time.

`ReadinessAssessment`

- Readiness result with score, dependency health, baseline completeness,
  monitoring, rollback, change-window posture, status, issues, and creation
  time.

`PlatformChange`

- Platform change with owner, affected capability count, dependency health,
  approval state, broad review, security review, change window, rollback plan,
  lifecycle status, and approval time.

`PlfdAuditEvent`

- Governance record for platform foundation lifecycle actions.

`PlfdAgent`

- Registered AI foundation agent with tenant, runtime, role, explicit scope,
  registration status, contribution disclosure, and activity state.

## Rule Engine

The deterministic rule engine must enforce:

- tenant context on every foundation operation;
- owner identity for foundation services;
- tier classification for foundation services;
- readiness score for foundation services;
- dependency health evidence;
- baseline evidence;
- baseline approver identity;
- healthy dependencies for platform change approval;
- configuration baseline presence;
- owner identity for platform changes;
- affected capability scope for platform changes;
- platform approval;
- security review;
- change window;
- rollback plan;
- Bytewax event stream for platform change lifecycle events;
- broad review for broad platform changes;
- registered AI foundation agent;
- supported AI-agent runtime;
- supported AI-agent role;
- explicit AI-agent scope;
- AI contribution disclosure;
- audit evidence for lifecycle state changes;
- Bytewax event stream for batch foundation mutation.

## UI Contract

The capability exposes these APG Python UI routes:

- `/plfd/dashboard`
- `/plfd/services`
- `/plfd/dependencies`
- `/plfd/baselines`
- `/plfd/readiness`
- `/plfd/changes`
- `/plfd/agents`
- `/plfd/governance`
- `/plfd/audit`
- `/plfd/settings`

View models must expose foundation summaries, services, dependency maps,
baselines, readiness assessments, changes, foundation agents, rules, audit
events, theme data, and Bytewax stream metadata.

## Theme

The default theme is `plfd_platform_foundation`. Theme components cover
foundation cards, dependency maps, baseline managers, change queues, agent
panels, and audit timelines.

## Event Stream

Lifecycle telemetry is described by:

- processor: `bytewax`
- topic: `apg.plfd.lifecycle`
- state: services, dependencies, baselines, readiness assessments, changes,
  PLFD agents, audit events
- events: service registered, dependency recorded, baseline attached,
  readiness assessed, change proposed, change approved, agent registered
- guardrail: `batch_foundation_mutation_requires_bytewax`

The package records stream metadata and guardrails. Live Bytewax topology is
an application deployment concern.

## Acceptance Criteria

- The package has local README, specification, plan, contract, runtime, view,
  API, test, package-manifest, semantic-model, and release-report artifacts.
- The contract exposes provides/requires, configuration schema, rules, routes,
  theme, and Bytewax stream metadata.
- The service supports foundation services, dependencies, baselines, readiness
  assessments, platform changes, AI-agent registration, audit events,
  tenant-local IDs, and Bytewax batch mutation validation.
- Focused tests prove the main lifecycle, guardrails, tenant isolation,
  generated evidence, and docs.
- Compile, focused pytest, implementation-audit, publish-plan, marker scan,
  and diff checks pass before commit.
