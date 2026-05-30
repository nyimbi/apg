# API Service Mesh Development Plan

## Slice Goal

Deliver a coherent lifecycle and guardrail packet for `composition_gateway` so the capability is executable, documented, testable, theme-aware, AI-agent aware, and Bytewax-aligned.

## Implementation Steps

1. Replace the generic contract with a domain-specific contract for services, routes, traffic, security, gateway agents, governance, UI, theme, adapters, and Bytewax lifecycle streaming.
2. Preserve the existing service-mesh internals and add a dependency-light APG lifecycle facade for focused package tests and generated application composition.
3. Replace package API, views, app, and registration entrypoints with dependency-light APG surfaces.
4. Refresh package evidence from the active contract.
5. Add specification and plan documents.
6. Expand focused tests around contract shape, rules, service lifecycle, guardrail failures, API/view surfaces, app self-test, and semantic metadata.
7. Run focused verification and commit the coherent packet.

## Review Checklist

- Tenant context is enforced.
- Service owner, endpoints, and health checks are required.
- Public routes require policy, approval, and TLS.
- Public services require rate limits and circuit breakers.
- Route and traffic lifecycle events require Bytewax.
- Certificates require owners and secret references.
- Gateway agents support Codex, Claude Code, OpenCode, and Pi.
- Package imports remain dependency-light.

## Deferred Work

- Bind to live ingress controllers, service discovery stores, certificate managers, and traffic proxies.
- Deploy live Bytewax topology for gateway lifecycle events.
- Render browser UI screens and run visual checks.
- Run gateway performance and failover checks after battery constraints are lifted.
