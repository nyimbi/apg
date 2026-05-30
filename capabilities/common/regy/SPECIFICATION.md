# REGY Capability Specification

## Purpose

REGY is APG's API and service registry capability. It gives generated APG
applications a governed control plane for service registration, instance
registration, service discovery, health evidence, API version governance,
gateway publication, retirement, and audit evidence.

REGY must be useful without booting a service mesh, Kubernetes cluster,
external registry, cache backend, Bytewax worker, gateway, monitoring backend,
or APG auth stack. Those systems remain adapter boundaries. The
dependency-light lifecycle service must still make registry decisions
executable and auditable.

## Scope

REGY owns:

- tenant-scoped service registration;
- service instance registration and health evidence;
- service discovery and healthy-instance selection;
- service contract and API version governance;
- production registration, compatibility, discovery-limit, owner-transfer, and
  retirement review records;
- gateway publication eligibility;
- registry audit events;
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

### UI and Theme

REGY must expose generated UI models for dashboard, services, registration,
instances, discovery, health, versions, contract reviews, gateway sync,
retirements, audit, and settings. Theme metadata must include compact registry
components for catalog rows, registration forms, discovery results, instance
tables, health timelines, version matrices, contract reviews, gateway sync,
retirement reviews, and audit timelines.

## Guardrails

REGY decisions must return `allow`, `deny`, or `require_review`, with matched
rules and required actions. Guardrails must cover tenant context, service owner,
health endpoint, API version, contract schema, duplicate names, production
registration review, instance endpoint, health probe, region, weight,
cross-tenant discovery, high discovery limits, gateway publication,
healthy-instance requirements, routing metadata, breaking changes,
deprecation, manual health overrides, owner transfer, retirement impact, gateway
unpublish, and production tracing.

## Adapter Boundaries

The dependency-light control plane must not execute live registry, gateway,
service mesh, monitor, audit sink, cache, or Bytewax operations. Those adapters
must treat REGY decisions as the source of truth before performing side effects.

## Acceptance Criteria

- Root `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe current REGY
  behavior and adapter boundaries.
- Contract exposes configuration, rules, adapters, UI, theme, and package
  evidence for service, instance, discovery, version, publication, retirement,
  and audit workflows.
- Generated apps can use a dependency-light service for registry lifecycle
  workflows without optional production dependencies.
- Focused tests cover positive and negative guardrail paths.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` derive from the current contract.
- Focused compile, tests, implementation audit, publish-plan, stale marker
  scan, and diff checks pass.
