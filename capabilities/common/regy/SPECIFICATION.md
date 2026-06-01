# REGY Capability Specification

## Purpose

REGY is APG's API and service registry capability. It gives generated APG
applications a governed control plane for service registration, instance
registration, service discovery, health evidence, API version governance,
gateway publication, retirement, and audit evidence.

REGY must be useful without booting a service mesh, Kubernetes cluster,
external registry, cache backend, Bytewax worker, gateway, monitoring backend,
APG auth stack, or external AI-agent runtime. Those systems remain adapter
boundaries. The dependency-light lifecycle service must still make registry
decisions executable and auditable.

## Scope

REGY owns:

- tenant-scoped service registration;
- service instance registration and health evidence;
- service discovery and healthy-instance selection;
- service contract and API version governance;
- production registration, compatibility, discovery-limit, owner-transfer, and
  retirement review records;
- gateway publication eligibility;
- first-class registry-agent registrations for AI and automation tools that
  participate in service registration, contract review, discovery review,
  health review, gateway sync, owner transfer, retirement review, or catalog
  stewardship;
- Bytewax lifecycle batch validation for registry mutation streams;
- registry audit events;
- durable review evidence for policy decisions, matched rules, review reasons,
  required actions, pending review queues, and denial records;
- generated application API helpers, UI view models, theme tokens, and package
  evidence.

REGY integrates with:

- `auth` for service authentication and registry permissions;
- `conf` for dynamic configuration and discovery metadata;
- `moni` for metrics, health, traces, and SLOs;
- `audl` for audit trails;
- `apig` for gateway publication and route synchronization;
- `cach` for discovery and health caches;
- Bytewax-backed streams for registry lifecycle events.
- external AI and automation runtimes such as Codex, Claude Code, OpenCode,
  Pi, and future tools through governed adapter boundaries.

## Functional Requirements

### Service Lifecycle

REGY must register services with tenant context, owner, service type,
environment, API version, contract schema reference, health endpoint, routing
metadata, labels, and audit evidence. Duplicate service names are blocked
within a tenant. Production services require production review evidence and
trace propagation evidence.

### Instance Lifecycle

REGY must register service instances with service reference, endpoint, allowed
region, health probe, load-balancing weight, health state, and audit evidence.
Instances with missing endpoints, missing health probes, disallowed regions, or
non-positive weights are denied.

### Discovery Lifecycle

REGY must discover services in tenant scope, optionally filtered by name and
healthy instances. Cross-tenant discovery is denied by default. Large discovery
result limits require review evidence.

### Version and Contract Lifecycle

REGY must record API/service versions with schema references. Breaking changes
require compatibility review. Deprecated versions require migration notes and a
future end-of-life date.

### Gateway Publication Lifecycle

REGY must publish only registered services with at least one healthy instance
and routing metadata. Live gateway side effects are adapter-owned and must
honor the generated-app guardrail decision.

### Retirement Lifecycle

REGY must retire services only after impact review and gateway unpublish
evidence are recorded. Retirement produces audit evidence and keeps historical
registry records visible.

### Registry-Agent Lifecycle

REGY must model AI and automation agents as first-class registry participants.
Agent registration requires a supported runtime, supported role, bounded scope,
accountable owner, documented purpose, and machine-contribution disclosure.
Privileged roles such as registration reviewer, contract reviewer, discovery
reviewer, health reviewer, gateway sync reviewer, owner transfer reviewer, and
retirement reviewer require human approval evidence before they can mutate
registry state.

Supported runtimes are adapter identifiers, not embedded SDK commitments:
`codex`, `claude_code`, `opencode`, and `pi`. Future runtimes can be added by
extending the contract and adapter policy while preserving the same guardrail
shape.

### Bytewax Lifecycle Batches

REGY must validate lifecycle mutation batches before adapter side effects.
Accepted lifecycle batches must use Bytewax as the required processor. Non-
Bytewax batches are recorded as denied evidence and blocked.

### Durable Review Evidence

REGY must persist policy decisions on all generated-app registry lifecycle
records. Services, instances, versions, gateway publications, reviews,
registry agents, lifecycle batches, and audit events carry `policy_decision`,
`matched_rules`, `review_reasons`, and `review_evidence` fields. Generated
applications must be able to compose a single pending-review queue from those
records, and denied non-Bytewax lifecycle batches must remain visible as
auditable evidence after the blocking exception.

### UI and Theme

REGY must expose generated UI models for dashboard, services, registration,
instances, discovery, health, versions, contract reviews, gateway sync,
retirements, audit, registry-agent roster, lifecycle-batch monitor, and
settings. Theme metadata must include compact registry components for catalog
rows, registration forms, discovery results, instance tables, health timelines,
version matrices, contract reviews, gateway sync, retirement reviews, audit
timelines, registry-agent rosters, and Bytewax lifecycle panels.

## Guardrails

REGY decisions must return `allow`, `deny`, or `require_review`, with matched
rules and required actions. Guardrails must cover tenant context, service owner,
health endpoint, API version, contract schema, duplicate names, production
registration review, instance endpoint, health probe, region, weight,
cross-tenant discovery, high discovery limits, gateway publication,
healthy-instance requirements, routing metadata, breaking changes,
deprecation, manual health overrides, owner transfer, retirement impact, gateway
unpublish, production tracing, registry-agent runtime, registry-agent role,
agent scope, agent owner, agent purpose, contribution disclosure, human
approval for privileged agent roles, and Bytewax lifecycle processing.

## Adapter Boundaries

The dependency-light control plane must not execute live registry, gateway,
service mesh, monitor, audit sink, cache, external AI runtime, or Bytewax
operations. Those adapters must treat REGY decisions as the source of truth
before performing side effects.

## Acceptance Criteria

- Root `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe current REGY
  behavior and adapter boundaries.
- Contract exposes configuration, rules, adapters, UI, theme, and package
  evidence for service, instance, discovery, version, publication, retirement,
  registry-agent, lifecycle-batch, and audit workflows.
- Contract exposes `review_evidence` metadata, and API/view-model surfaces
  expose pending review queues for generated applications.
- Generated apps can use a dependency-light service for registry lifecycle
  workflows without optional production dependencies.
- Focused tests cover positive and negative guardrail paths.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` derive from the current contract.
- Focused compile, tests, implementation audit, publish-plan, stale marker
  scan, and diff checks pass.
