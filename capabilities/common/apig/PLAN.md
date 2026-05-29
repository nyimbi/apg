# APIG Capability Development Plan

## Current State

APIG has a large production gateway implementation plus an executable contract,
API helper shell, view shell, packaging evidence, and contract tests. The
package-level API/view facade currently needs a dependency-light service that
can execute route lifecycle governance without booting the production gateway
runtime.

## Packet 1: Governed Route Publication

Deliver a focused lifecycle packet:

- add package-level upstream, route, quota review, and audit event records;
- add a dependency-light `ApigService` control-plane facade;
- register tenant-owned upstream services;
- request routes and enforce registered upstream, auth policy, threat policy,
  signed edge-filter, and tenant guardrails;
- create and decide high-quota reviews;
- activate only approved high-quota routes;
- expose API helper and view-model surfaces for routes, upstreams, reviews,
  traffic, security, edge filters, analytics, and audit evidence;
- replace stale generated-package test naming with package contract tests;
- update package documentation and progress evidence.

## Implementation Steps

1. Extend `models.py` with `GatewayUpstreamRecord`, `GatewayRouteRecord`,
   `GatewayQuotaReview`, and `GatewayAuditEvent`.
2. Add `gateway_runtime.py` with the dependency-light `ApigService` facade.
3. Update `api.py` and `views.py` to use the dependency-light facade.
4. Update registration metadata with quota review and audit endpoint surfaces.
5. Extend package contract tests with positive high-quota review activation and
   negative public-auth, unsafe-method, unsigned-filter, tenant-mismatch, and
   duplicate-ID isolation coverage.
6. Rename generated-package tests to package contract naming.
7. Update `cap_spec.md` with current executable lifecycle and proof commands.
8. Run focused package proof, implementation audit, publish-plan, and diff
   checks.

## Review Checklist

- Upstream, route, quota review, and audit state is tenant-qualified.
- Public routes require auth policy.
- Unsafe methods require threat policy.
- WebAssembly filters require verified signatures.
- High quota routes require approved reviews before activation.
- Tenant mismatches are blocked.
- API helpers expose the same behavior as service methods.
- View models expose route, upstream, review, traffic, security, edge, rule,
  theme, and audit-event state.
- Production gateway, service mesh, edge runtime, monitoring, and AI systems
  remain adapter boundaries.
