# API Service Mesh Specification

## Intent

API Service Mesh makes routing, policy, traffic control, certificates, health evidence, and gateway-agent review composable APG primitives. It lets generated applications expose services through a shared gateway lifecycle instead of repeating gateway logic in every capability.

## Functional Requirements

- Register tenant-scoped services with owners, endpoints, health checks, and capability bindings.
- Create routes for registered services with path and method matching.
- Require policy, approval, and TLS for public routes.
- Attach gateway policies with rate limit and circuit-breaker evidence for public services.
- Shift traffic with weighted target maps and canary evidence when canary routing is used.
- Register certificate references with owner, domain, expiry, and secret-reference attribution.
- Record service health observations.
- Register first-class gateway agents for Codex, Claude Code, OpenCode, and Pi.
- Expose dashboard, service registry, route console, policy center, traffic console, certificate console, agent workbench, and settings UI models.

## Rule Engine

The deterministic rule engine enforces tenant context, write policy attachment, service ownership, endpoint presence, health checks, public-route policy, public-route approval, public-route TLS, Bytewax route events, canary evidence, Bytewax traffic-shift events, public-service rate limits, public-service circuit breakers, certificate ownership, secret references, batch route Bytewax coordination, agent runtime and role support, and human approval for privileged agent actions.

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with configuration, schema, rules, UI, theme, and Bytewax streaming metadata.
- Package import exposes `CompositionGatewayService`, `ASMService`, contract helpers, and registration metadata without web-framework imports.
- Service supports service, route, policy, traffic, certificate, health, agent, batch, dashboard, audit, and compatibility record operations.
- API helpers and view models expose the same lifecycle surfaces.
- Semantic model includes `gateway_agents`, required dependencies, route metadata, rules, theme, and Bytewax stream metadata.
- Focused tests cover lifecycle success paths and guardrail failures.
