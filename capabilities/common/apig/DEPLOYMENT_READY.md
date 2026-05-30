# APIG Deployment Readiness Notes

APIG is ready for generated-application composition as a governed control-plane
packet. Runtime deployment readiness depends on the adapter checks below.

## Package-Ready Surface

- Upstream, consumer, route, traffic, policy, deployment, retirement, and audit
  lifecycle records.
- Deterministic guardrails for API gateway control decisions.
- Generated API helpers and UI view models.
- Contract-derived semantic and release evidence.

## Deployment Gates Still Required

- Bind APIG to a concrete reverse proxy, ingress controller, or service mesh.
- Bind service discovery, auth/RBAC, key/certificate management, metrics,
  audit, cache, and event streaming adapters.
- Validate WebAssembly filter loading and signature verification with the
  selected runtime.
- Run rendered UI checks.
- Run live route, quota, canary, rollback, and retirement tests.
- Run performance, resilience, and failover tests in a dedicated environment.
