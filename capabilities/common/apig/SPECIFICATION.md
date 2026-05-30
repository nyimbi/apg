# APIG Capability Specification

## Purpose

APIG is APG's API gateway and management capability. It gives generated APG
applications a governed control plane for upstream services, consumers, routes,
traffic policies, security policies, edge filters, quota reviews, canary traffic
shifts, deployments, retirement, and audit evidence.

APIG must be useful without booting a reverse proxy, Kubernetes cluster,
service mesh, WebAssembly runtime, Redis cache, AI provider, or external API
manager. Those systems remain adapter boundaries. The package-backed lifecycle
service must still make route publication decisions executable and auditable.

## Scope

APIG owns:

- tenant-scoped upstream service registration;
- API consumer registration and credential governance;
- route request, review, activation, and retirement;
- auth, threat, mTLS, quota, rate-limit, canary, edge-filter, and observability
  guardrails;
- high-quota and canary review decisions;
- deployment gate records for gateway rollout;
- generated application API helpers, UI view models, theme tokens, and package
  evidence.

APIG integrates with:

- `auth`/`auth_rbac` for authentication and authorization;
- `conf`/registry for service discovery and gateway configuration;
- `moni` for metrics, traces, access logs, health, and SLOs;
- `audl`/audit compliance for decision trails;
- `mqeb` or Bytewax-backed streams for gateway events;
- `keym` for API keys, certificates, and signing material;
- `cach` for gateway/cache policy adapters;
- edge runtime adapters for WebAssembly filters and regional deployments.

## Functional Requirements

### Upstream Lifecycle

APIG must register upstream services with tenant context, owner, HTTPS base URL
by default, health status, discovery labels, and audit evidence.

### Consumer Lifecycle

APIG must register API consumers with owner, identity provider, credential
rotation evidence, RBAC approval for restricted routes, and audit evidence.

### Route Lifecycle

APIG must request, review, activate, list, and retire routes. Route requests
must include path, methods, upstream reference, owner, exposure, auth policy,
threat policy, mTLS evidence, rate-limit evidence, requested quota, optional
consumer reference, optional WebAssembly filter evidence, lineage/trace
settings, and rollback plan.

### Traffic and Release Lifecycle

APIG must govern high-quota requests, canary traffic shifts, regional edge
deployments, rollback plans, and route retirement impact reviews.

### Policy Lifecycle

APIG must record policy changes and require review/audit evidence before
activation for security-sensitive route or traffic policy changes.

### UI and Theme

APIG must expose generated UI models for dashboard, routes, upstreams,
consumers, traffic, security, edge filters, quota reviews, canary releases,
deployments, analytics, audit, and settings. Theme metadata must include
components for route status, upstream health, traffic policies, security
posture, edge filters, quota reviews, canary releases, deployment gates, and
audit timelines.

## Guardrails

APIG decisions must return `allow`, `deny`, or `require_review`, with matched
rules and required actions. Guardrails must cover tenant context, upstream
ownership, HTTPS requirements, upstream health, consumer ownership, credential
rotation, RBAC approval, route ownership, absolute route paths, registered
upstreams, HTTP method evidence, public auth, unsafe method threat policy,
mTLS, rate limits, high quota review, signed edge filters, canary review,
canary percentage limits, rollback plans, deployment observability, policy
review, and retirement impact review.

## Adapter Boundaries

The dependency-light control plane must not execute live proxy operations.
Reverse proxies, ingress controllers, service meshes, WAFs, certificate
managers, external API managers, WebAssembly engines, metrics backends, audit
sinks, cache stores, Bytewax flows, and AI optimization providers are adapters
that must honor APIG decisions.

## Acceptance Criteria

- Root `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe current APIG
  behavior and adapter boundaries.
- Contract exposes configuration, rules, adapters, UI, theme, and package
  evidence for upstream, consumer, route, traffic, policy, deployment, and audit
  workflows.
- Generated apps can use a dependency-light service for gateway lifecycle
  workflows without optional production dependencies.
- Focused tests cover positive and negative guardrail paths.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` derive from the current contract.
- Focused compile, tests, implementation audit, publish-plan, stale marker
  scan, and diff checks pass.
