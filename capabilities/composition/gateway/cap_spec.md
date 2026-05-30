# APG API Service Mesh Capability Specification

## Capability Metadata

- **Capability Code:** COMPOSITION_GATEWAY
- **Capability Name:** API Service Mesh
- **Version:** 2.1.0
- **Category:** Composition
- **Runtime Target:** python
- **Primary Stream Processor:** Bytewax

## Purpose

API Service Mesh is the APG composition-layer gateway for services, routes, policies, traffic shifts, certificate references, health evidence, and gateway review agents. It gives generated applications a shared operational plane for routing and protecting capability APIs without embedding gateway logic inside each capability.

## Scope

The capability owns these executable surfaces:

- Service registration with tenant, owner, capability binding, endpoint inventory, and health-check path.
- Route lifecycle with public-route policy, approval, TLS, and Bytewax lifecycle routing.
- Gateway policies for rate limits and circuit breakers.
- Canary and weighted traffic-shift records with evidence and Bytewax routing.
- Certificate reference lifecycle with accountable owner and secret reference.
- Health evidence capture for registered services.
- First-class gateway agents for Codex, Claude Code, OpenCode, and Pi.
- UI contracts for dashboard, services, routes, policies, traffic, certificates, agents, and settings.

## Lifecycle

1. Register a service with an owner, capability id, endpoint list, and health-check path.
2. Attach gateway policy for public services, including rate limits and circuit breakers.
3. Register certificate references for public TLS routes.
4. Create routes with route policy, approval, TLS, and Bytewax event routing when public.
5. Shift traffic with canary evidence when using canary rollout.
6. Record health evidence as gateway observations arrive.
7. Register gateway agents for mesh, route, policy, traffic, certificate, and incident review lanes.
8. Audit every state change and expose operational UI models.

## Guardrails

- Tenant context is mandatory.
- Gateway writes require policy attachment.
- Services require owners, endpoints, and health checks.
- Public routes require route policy, approval, and TLS.
- Route lifecycle events require Bytewax.
- Canary traffic shifts require evidence and review.
- Traffic-shift events require Bytewax.
- Public services require rate limits and circuit breakers.
- Certificates require owners and secret references.
- Batch route changes require Bytewax.
- Gateway agents require supported runtime and role.
- Privileged gateway actions proposed by agents require human approval.

## UI Contract

The capability exposes these APG routes:

- `/composition-gateway/dashboard`
- `/composition-gateway/services`
- `/composition-gateway/routes`
- `/composition-gateway/policies`
- `/composition-gateway/traffic`
- `/composition-gateway/certificates`
- `/composition-gateway/agents`
- `/composition-gateway/settings`

## Event Stream

- **Processor:** `bytewax`
- **Stream:** `apg.composition.gateway.lifecycle`
- **Key:** `tenant_id`
- **Events:** service registered, route created, policy attached, traffic shifted, certificate registered, health recorded, gateway agent registered.

## Integration Requirements

- Requires `auth`, `audl`, `ntfy`, `registry`, `composition_access`, and `composition_events`.
- Provides service mesh registry, route lifecycle, traffic management, gateway policy enforcement, certificate lifecycle, mesh health observability, and gateway agents.
- Uses APG Python runtime surfaces: `service.py`, `api.py`, `views.py`, and `app.py`.
