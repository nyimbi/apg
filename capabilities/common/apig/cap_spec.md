# API Gateway & Management Capability Summary

- **Capability name**: API Gateway & Management
- **Capability ID**: `apig`
- **Category**: common
- **Version**: 1.0.0

## Purpose

APIG provides a governed API gateway control plane for APG applications. It
models upstream services, API consumers, routes, traffic policies, security
controls, edge filters, quota reviews, canary releases, deployment gates, route
retirement, governed gateway agents, Bytewax lifecycle batches, and audit
events.

The package separates generated-app lifecycle behavior from production runtime
adapters:

- `gateway_runtime.ApigService` is dependency-light and used for package
  composition and focused proof.
- production gateway modules bind reverse proxies, service meshes, edge
  runtimes, metrics, audit, cache, APG service clients, and AI/optimization
  systems.

## Provided Services

- `apig_operations`
- `api_gateway`
- `traffic_management`
- `gateway_agent_composition`
- upstream lifecycle governance
- consumer lifecycle governance
- route publication and retirement workflow
- quota review and canary traffic-shift governance
- gateway policy and deployment gate evaluation
- provider-neutral AI/automation agent participation for Codex, Claude Code,
  OpenCode, Pi, and future runtimes
- Bytewax lifecycle-batch validation
- generated-application UI route and theme metadata

## Required Services

- tenant context
- `auth` for identity and authorization
- `moni` for metrics, traces, logs, and health evidence
- `mqeb` for event-publication integration where configured
- `conf` for discovery/configuration
- adapter-bound `keym`, `audl`, and `cach` services when production
  deployments bind credentials, audit sinks, or cache stores
- Bytewax-backed event streaming for gateway lifecycle batches

## Configuration

Configuration is defined by `capability_contract.py` and includes upstreams,
consumers, routes, traffic, security, edge, canary, deployments, governance,
observability, adapters, agents, streaming, UI, and theme sections.

## Rules

APIG ships deterministic guardrails for tenant context, upstream ownership,
HTTPS, health checks, consumer ownership, credential rotation, RBAC approval,
route ownership, absolute paths, registered upstreams, HTTP methods, public auth,
external mTLS, unsafe methods, route rate limits, signed edge filters, quota
review, canary review, canary percentage limits, rollback plans, deployment
regions, observability, production approval, policy review, and route retirement
impact review. It also enforces supported gateway-agent runtime and role,
agent scope, owner, purpose, contribution disclosure, privileged-role human
approval, and Bytewax-only lifecycle-batch routing.

## UI

Generated UI models live in `view_models.py` and cover dashboard, route
designer, upstream manager, consumer manager, traffic console, security policy
console, edge filter manager, quota review queue, canary release console,
deployment gates, analytics, gateway-agent roster, lifecycle-batch monitor,
audit timeline, and settings.

## Runtime Boundary

Focused package tests do not configure a live proxy, deploy Kubernetes
resources, execute WebAssembly, publish Bytewax flows, or run performance
benchmarks. Production adapters must honor APIG lifecycle decisions before
performing side effects. AI/automation tools are also adapters: the lifecycle
packet records their runtime, role, scope, owner, purpose, and approval posture,
but it does not embed vendor-specific clients.
