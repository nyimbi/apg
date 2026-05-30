# EDGE Edge Computing Specification

## Purpose

The EDGE capability (`edge`) lets generated APG applications compose
tenant-scoped edge nodes, fleets, signed workloads, deployments, offline
execution, synchronization, resource pressure monitoring, audit evidence, visual
route metadata, theme metadata, Bytewax stream governance, and AI-agent
assistance into ERP, IoT, manufacturing, logistics, retail, and field-service
applications.

This package owns the executable contract, deterministic guardrails,
dependency-light service, API helpers, view models, UI route metadata, theme
metadata, Bytewax stream declaration, generated semantic evidence, and focused
proof commands. Physical devices, container runtimes, model runtimes, remote
update systems, telemetry stores, and stream-worker deployments remain adapter
concerns.

## Users And Jobs

- Edge operators register nodes with owner, attestation, location policy,
  health, secure transport, capacity, and capabilities.
- Fleet managers group nodes under a policy version and monitor membership.
- Release managers register signed workloads and deploy them to healthy nodes
  with enough capacity.
- Operations teams synchronize state after offline execution and resolve
  conflicts.
- Security reviewers inspect attestation, secure transport, signed artifacts,
  agent scope, and audit evidence.
- Platform engineers bind device enrollment, distribution, cache, monitoring,
  audit, identity, and Bytewax workers.
- AI agents assist with fleet optimization, node health review, workload
  placement, offline sync review, and security review under explicit
  registration and disclosure.

## Capability Boundary

`edge` provides:

- edge node lifecycle;
- edge fleet lifecycle;
- signed workload lifecycle;
- deployment placement;
- offline execution and synchronization metadata;
- resource pressure and audit summaries;
- AI edge-agent registration and policy enforcement;
- Bytewax stream metadata for batch edge mutation.

`edge` requires:

- `auth` for identity and permission context;
- `conf` for tenant configuration;
- `audl` for durable audit evidence;
- `dist` for artifact distribution;
- `cach` for offline cache policy;
- `moni` for operational monitoring.

## Lifecycle

Node lifecycle:

1. A node is registered with tenant, owner, node type, location, location
   policy, attestation, health, secure transport, capacity, and capabilities.
2. Missing owner, attestation, location policy, or secure transport denies the
   registration.
3. Node membership in a fleet records audit evidence.

Fleet lifecycle:

1. A fleet is created with tenant, owner, name, and policy version.
2. Nodes can be attached only within the same tenant.
3. Membership changes record audit evidence.

Workload lifecycle:

1. A workload is registered with owner, version, signed artifact digest,
   deployment policy, and resource quota.
2. Unsigned artifacts or missing quota deny registration.
3. Deployment checks node health, secure transport, and capacity before
   reserving resources.

Sync lifecycle:

1. A sync session references node, workload, conflict policy, cache policy,
   offline hours, event count, and conflicts.
2. Missing conflict or cache policy denies synchronization.
3. Long offline windows require review.
4. Conflict status, replay status, and review status are reflected in view
   models.

AI-agent lifecycle:

1. Agent is registered with runtime, role, scope, tenant, and disclosure.
2. Runtime must be one of `codex`, `claude_code`, `opencode`, or `pi`.
3. Role must be one of the configured EDGE roles.
4. Agent contributions are audit-visible and cannot bypass policy decisions.

## Rule Engine

Rules must deny or require review for:

- missing tenant context;
- missing node owner;
- missing node attestation;
- missing node location policy;
- insecure edge transport;
- missing fleet owner or policy version;
- missing workload owner, artifact signature, or resource quota;
- missing sync conflict policy or cache policy;
- long offline windows without review;
- unregistered, unsupported, unscoped, or undisclosed AI agents;
- lifecycle state changes without audit evidence;
- batch edge mutations that do not use Bytewax.

## UI And Theme

The APG Python UI contract exposes dashboard, nodes, fleets, workloads,
deployments, sync, agents, rules, analytics, audit, and settings routes. The
theme uses compact operational density with distinct treatments for node maps,
fleet panels, workload tables, deployment tables, sync timelines, edge-agent
scope, stream health, and audit digests.

## Streaming

Batch edge mutation must use Bytewax. The stream topic is
`apg.edge.lifecycle`, and state covers nodes, fleets, workloads, deployments,
sync sessions, edge agents, and audit events. Live Bytewax topology deployment
is an adapter concern, but the package declares and enforces the guardrail.

## Adapter Boundaries

Adapters must handle:

- physical device enrollment and attestation providers;
- container or process runtime execution;
- model runtime execution;
- artifact distribution through `dist`;
- offline cache storage through `cach`;
- authentication and permission checks through `auth`;
- audit durability through `audl`;
- monitoring and alerting through `moni`;
- Bytewax lifecycle topology and operational monitoring.

## Acceptance Gates

- Contract validates through the APG capability registry.
- Configuration schema includes nodes, fleets, workloads, sync, edge agents,
  governance, observability, adapters, UI, and theme.
- Rules cover node, fleet, workload, sync, agent, audit, and Bytewax guardrails.
- Service can register nodes, create fleets, register workloads, deploy
  workloads, sync state, review offline windows, register agents, summarize
  state, and validate batch mutation streams.
- API helpers and view models expose the same lifecycle surfaces.
- Generated semantic evidence exposes provides/requires, routes, rules, theme,
  and streaming.
- README, specification, plan, progress log, focused tests, implementation
  audit, publish plan, and stale-marker scan are current.
