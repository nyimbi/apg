# DVRL Deployment Guide

The current DVRL packet is ready for generated-application composition and
focused package verification. Production deployment requires binding and
testing the adapter-backed runtime services listed below.

## Required Runtime Bindings

- Tenant and actor context provider.
- RBAC provider for restricted query authorization.
- Credential vault for source secrets.
- Metadata catalog for schemas, classification, and lineage.
- Cache store for query-result cache entries.
- Audit sink for source, query, policy, and retirement decisions.
- Connector registry for physical database, warehouse, API, file, stream, and
  Singer source adapters.
- Query planner and execution engine.
- Bytewax runtime for streaming flow execution.

## Deployment Checklist

- Configure DVRL contract defaults for the tenant.
- Confirm allowed source types and source registration approval policy.
- Confirm query timeout, row-limit, cost threshold, cross-source join threshold,
  and cache TTL values.
- Bind the adapter services.
- Run focused DVRL package checks.
- Run live connector smoke checks with non-production data.
- Run rendered UI checks for the generated DVRL screens.
- Run load and performance tests in a dedicated verification window.

## Focused Package Checks

```bash
./.venv/bin/pytest -q capabilities/common/dvrl/test_capability_contract.py capabilities/common/dvrl/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/dvrl --json
./.venv/bin/apg capabilities publish-plan capabilities/common/dvrl --json
```

## Non-Goals for the Packet

This packet does not certify live production performance, external connector
availability, cache persistence, metadata synchronization, audit persistence,
or rendered browser behavior.
