# API Gateway & Management Capability Summary

- **Capability name**: API Gateway & Management
- **Capability ID**: `apig`
- **Category**: common
- **Version**: 1.0.0

## Purpose

APIG provides a governed API gateway control plane for APG applications. It
models upstream services, API consumers, routes, traffic policies, security
controls, edge filters, quota reviews, canary releases, deployment gates, route
retirement, and audit events.

The package separates generated-app lifecycle behavior from production runtime
adapters:

- `gateway_runtime.ApigService` is dependency-light and used for package
  composition and focused proof.
- production gateway modules bind reverse proxies, service meshes, edge
  runtimes, metrics, audit, cache, APG service clients, and AI/optimization
  systems.

## Provided Services

- `apig_operations`
- upstream lifecycle governance
- consumer lifecycle governance
- route publication and retirement workflow
- quota review and canary traffic-shift governance
- gateway policy and deployment gate evaluation
- generated-application UI route and theme metadata

## Required Services

- tenant context
- `auth` or `auth_rbac` for identity and authorization
- `conf` or registry service for discovery/configuration
- `keym` for API keys, certificates, and signing material
- `moni` for metrics, traces, logs, and health evidence
- `audl` for audit trails
- Bytewax-backed event streaming for future gateway lifecycle events

## Configuration

Configuration is defined by `capability_contract.py` and includes upstreams,
consumers, routes, traffic, security, edge, canary, deployments, governance,
observability, adapters, UI, and theme sections.

## Rules

APIG ships deterministic guardrails for tenant context, upstream ownership,
HTTPS, health checks, consumer ownership, credential rotation, RBAC approval,
route ownership, absolute paths, registered upstreams, HTTP methods, public auth,
external mTLS, unsafe methods, route rate limits, signed edge filters, quota
review, canary review, canary percentage limits, rollback plans, deployment
regions, observability, production approval, policy review, and route retirement
impact review.

## UI

Generated UI models live in `view_models.py` and cover dashboard, route
designer, upstream manager, consumer manager, traffic console, security policy
console, edge filter manager, quota review queue, canary release console,
deployment gates, analytics, audit timeline, and settings.

## Runtime Boundary

Focused package tests do not configure a live proxy, deploy Kubernetes
resources, execute WebAssembly, publish Bytewax flows, or run performance
benchmarks. Production adapters must honor APIG lifecycle decisions before
performing side effects.
