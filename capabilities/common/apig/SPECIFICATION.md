# APIG Capability Specification

## Identity

- Capability ID: `apig`
- Display name: APG Intelligent Gateway
- Category: `common`
- Owner: APG Platform Team
- Runtime shell: `apg_python`
- Theme: `apig_gateway_console`

## Purpose

APIG is the tenant-scoped API gateway control plane for APG applications. It
registers upstream services, governs route publication, enforces auth and threat
policies, controls signed edge filters, records high-quota reviews, and exposes
gateway state through API helpers and view models.

The package must remain usable without a running reverse proxy, service mesh,
Kubernetes cluster, Redis, WebAssembly runtime, or AI provider. Those systems
remain adapter boundaries. Local package proof focuses on deterministic
governance, route lifecycle, tenant isolation, and composition behavior.

## Users And Outcomes

- Platform teams can register upstream services with owners and health state.
- API owners can request routes and see why guardrails block publication.
- Security reviewers can require auth, threat policies, and signed edge
  filters before traffic is exposed.
- Operations reviewers can approve high-RPS quotas before activation.
- Generated APG applications can compose APIG with AUTH, MONI, MQEB, CONF,
  AUDL, WFLO, NTFY, and AICR without binding to one gateway implementation.

## Domain Model

APIG owns these package-level records:

- `GatewayUpstreamRecord`: tenant-scoped upstream service registration.
- `GatewayRouteRecord`: governed route request and activation state.
- `GatewayQuotaReview`: high-quota approval request for route activation.
- `GatewayAuditEvent`: tenant-scoped evidence event for gateway lifecycle
  decisions.

All mutable package-level state must be tenant-qualified so duplicate IDs in
different tenants cannot collide.

## Lifecycle

The focused lifecycle is:

1. Register a tenant-owned upstream service.
2. Request a route against the registered upstream.
3. Deny public routes without auth policy.
4. Deny unsafe methods without threat policy.
5. Deny unsigned WebAssembly edge filters.
6. Create a pending quota review for high-RPS routes.
7. Approve or reject the quota review with reviewer evidence.
8. Activate only routes that pass guardrails and have required quota approval.
9. Emit audit evidence for registration, route request, review decision, and
   activation.

## Rules And Guardrails

The contract rules are executable guardrails:

- `tenant_context_required`: operations require tenant context.
- `route_requires_registered_service`: routes require registered upstreams.
- `public_route_requires_auth_policy`: public routes require auth policy.
- `unsafe_method_requires_threat_policy`: unsafe methods require threat policy.
- `wasm_filter_requires_signature`: edge filters require signature
  verification.
- `high_quota_requires_review`: high-RPS routes require review.

Service methods must enforce these rules and expose the same decisions through
API helpers and view models.

## UI And Theme

APIG exposes route and view-model surfaces for:

- dashboard summary;
- route designer;
- traffic console;
- security policy console;
- upstream services;
- edge filters;
- analytics;
- settings.

The `apig_gateway_console` theme must provide semantic tokens and component
metadata for route status, traffic policy, topology maps, and edge-filter
signature traces.

## Adapter Boundaries

These integrations remain replaceable:

- reverse proxy, service mesh, and ingress controllers;
- Kubernetes and edge deployment systems;
- WebAssembly runtimes and filter registries;
- traffic analytics and metrics backends;
- audit/SIEM exporters;
- AI route-optimization providers.

Local package tests must not require those systems.

## Acceptance Gates

Focused APIG proof:

```bash
./.venv/bin/pytest -q capabilities/common/apig/test_capability_contract.py capabilities/common/apig/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/apig --json
./.venv/bin/apg capabilities publish-plan capabilities/common/apig --json
git diff --check -- capabilities/common/apig
```
