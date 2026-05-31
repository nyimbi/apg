# CACH Capability Specification

## Purpose

CACH provides APG applications with a tenant-scoped caching control plane. It
defines how generated applications register namespaces, admit cache entries,
enforce freshness and encryption rules, warm predictable data, handle memory
pressure, and expose operable UI surfaces without binding every application to a
specific cache backend.

The capability is deliberately split into two layers:

- **Capability control plane**: dependency-light records, deterministic rules,
  API helpers, view models, semantic-model publication, and audit evidence that
  generated APG applications can compose immediately.
- **Runtime adapters**: Redis, in-process memory, edge caches, CDN caches,
  distributed stores, AI optimizers, APG monitoring, APG audit logging, and APG
  security integrations. These adapters must honor the control-plane decisions.

## Capability Outcomes

CACH must let a generated application:

1. Create tenant-scoped cache namespaces with explicit owners, classification,
   TTL, tier, quota, freshness, and encryption posture.
2. Evaluate every read, write, delete, warm, promotion, and eviction operation
   through deterministic guardrails before the runtime cache backend is touched.
3. Store cache-entry metadata that records key, namespace, tenant, tier, TTL,
   freshness, classification, encryption status, producer, size, access count,
   and lifecycle status.
4. Deny cross-tenant access unless an explicit future exchange capability is
   added.
5. Require encryption for sensitive, restricted, regulated, and credential-like
   entries.
6. Require namespace registration before writes or warming.
7. Deny stale reads for critical data unless stale-while-revalidate is explicitly
   enabled and the namespace policy permits it.
8. Require review for TTLs, warming batches, memory pressure, and eviction
   actions that exceed tenant or namespace limits.
9. Record warming plans with source evidence, batch size, requester, reviewer,
   and decision state.
10. Record eviction/capacity review decisions with independent reviewer notes.
11. Provide generated-application view models for dashboard, namespace
   inventory, entry explorer, policy manager, warming console, eviction review,
   tier topology, adapter health, cache-agent roster, lifecycle batches, audit
   timeline, and settings.
12. Publish a current semantic model and release report from the live capability
   contract rather than stale embedded JSON.
13. Register cache agents as first-class APG actors for namespace policy,
   warming, eviction, freshness, tier optimization, adapter health, and
   lifecycle audit decisions.
14. Fail closed when cache agents omit supported runtime, supported role, owner,
   purpose, scope, contribution disclosure, or privileged-role human approval.
15. Validate cache lifecycle batches through a Bytewax-first stream manifest
   before generated applications compose cache policy, warming, agent, or
   eviction mutations.

## Functional Scope

### Namespace Lifecycle

Namespaces are the operational boundary for cache behavior. Each namespace must
store:

- `tenant_id`
- `namespace`
- `owner`
- `data_classification`
- `default_ttl_seconds`
- `max_ttl_seconds`
- `max_entries`
- `default_tier`
- `allowed_tiers`
- `encryption_required`
- `critical_reads_require_freshness`
- `stale_while_revalidate_allowed`
- `source_registration_required`
- `status`

Valid namespace statuses are `active`, `disabled`, and `retiring`.

### Entry Lifecycle

Entry admission must capture:

- tenant, namespace, key, producer, value reference, tier, TTL, size, and
  lifecycle status
- classification and encryption evidence
- idempotent admission decision and matched rules
- timestamps for creation, expiry, last access, and invalidation

Valid entry statuses are `active`, `pending_review`, `expired`, `invalidated`,
`refresh_required`, and `denied`.

### Warming Lifecycle

Warming requests must include a namespace, source name, source registration
evidence, key count, requester, reason, and optional schedule. Large or
unregistered warming plans must fail closed or require independent review.

### Eviction and Memory Pressure

CACH must represent memory-pressure and eviction reviews as first-class records.
Eviction plans above configured thresholds require an independent reviewer and
review notes before they can be approved.

### Cache Agents

Cache agents are first-class APG actors. The initial supported runtimes are
Codex, Claude Code, opencode, and Pi. Each agent must declare tenant, agent ID,
name, runtime, role, scope, owner, purpose, contribution disclosure, and human
approval posture.

Privileged roles require human approval:

- warming reviewer
- eviction reviewer
- tier optimization reviewer
- adapter health reviewer

### Lifecycle Batches

Cache lifecycle batches represent groups of namespace, entry, warming, eviction,
and agent mutations. They must be non-empty and declare Bytewax as the lifecycle
processor. Live workers remain adapter work; this packet records the contract
and validation evidence.

### Rules

The rule engine is deterministic. It evaluates simple context dictionaries and
returns `allow`, `deny`, or `require_review` with matched rule evidence.

Baseline rules:

- tenant context is required
- writes require a namespace
- disabled namespaces block writes and warming
- sensitive/restricted/regulated entries require encryption
- cross-tenant access is denied
- critical stale reads require refresh
- TTL above namespace limit requires review
- warming requires a registered source
- warming batch above configured limit requires review
- memory pressure without an eviction plan requires review
- eviction review requires an independent reviewer
- review notes are required for capacity and eviction decisions
- cache-agent runtime and role must be supported
- cache-agent scope, owner, purpose, and contribution disclosure are required
- privileged cache-agent roles require human approval
- lifecycle batches require Bytewax stream processing

### UI and Theming

The capability must expose a compact, operations-focused UI contract. UI routes
are metadata only in this packet; generated applications can render them in their
chosen shell.

Required screens:

- dashboard
- namespaces
- entries
- policies
- warming
- eviction reviews
- hierarchy/tier topology
- analytics
- security
- adapters
- cache agents
- lifecycle batches
- audit
- settings

Theme tokens must be restrained and suitable for operational software. Theme
components should map to recognizable cache operations: hit-rate card, namespace
policy trace, tier topology, warming timeline, eviction queue, adapter panel,
entry freshness badge, and audit timeline.

## Integration Boundaries

CACH depends conceptually on:

- `conf` for tenant defaults and environment configuration
- `auth` for user and permission context
- `audl` for immutable audit evidence
- `mten` for tenant isolation when the runtime adapter binds tenant services
- `moni` for metrics, health, traces, and alerts when adapter telemetry is bound
- `mqeb` for optional event publication after Bytewax lifecycle validation

The dependency-light packet must not require those capabilities at import time.
Adapters must be allowed to bind them at runtime.

Runtime cache adapters may include memory, Redis-compatible stores, edge caches,
CDNs, browser caches, object-store metadata caches, or application-local query
caches. The control plane must remain backend-neutral.

## Non-Goals

- CACH does not implement every production distributed-cache algorithm in the
  control-plane packet.
- CACH does not promise fixed latency, throughput, or benchmark claims without a
  named backend and environment.
- CACH does not bypass APG authorization, tenant isolation, encryption, or audit
  capabilities when those adapters are present.
- CACH does not require optional compression, Flask-AppBuilder, Redis, AI, or
  dashboard dependencies merely to publish its capability contract.

## Acceptance Criteria

The CACH packet is serviceable when:

- `SPECIFICATION.md`, `PLAN.md`, and `README.md` explain the capability, usage,
  extension points, and boundaries.
- `capability_contract.py` exposes configuration, rules, UI routes, and theme
  components that cover the lifecycle above.
- `service.py` includes a dependency-light lifecycle/governance service that can
  create namespaces, admit/read/delete entries, request warming, and decide
  eviction reviews, register cache agents, and validate lifecycle batches.
- `api.py` exposes simple callable helpers over the lifecycle service.
- `view_models.py` exposes generated-application view models.
- `app.py`, `semantic_model.json`, and `release_report.json` are derived from
  current contract evidence.
- Focused package tests prove the rule engine, lifecycle service, view models,
  semantic model, and publish-plan path.
