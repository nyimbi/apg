# APIG Capability Build Plan

## Current State

APIG already has a large production-oriented gateway runtime, a dependency-light
`ApigService`, API helpers, view helpers, packaging evidence, and focused
contract tests. The package still needs the richer lifecycle packet shape used
for the current APG capability build-out.

Known gaps:

- APIG has the route, consumer, traffic, deployment, and policy lifecycle
  packet, but it needs the current first-class agent and Bytewax lifecycle-batch
  guardrail shape used by the completed MDM, META, ETLP, and DVRL packets.
- `app.py` and package evidence need to expose contract-level
  `provides`/`requires`, agents, streaming, and composition metadata instead of
  static `apig_operations` evidence.

## Build Sequence

1. Documentation baseline
   - Add root `README.md`.
   - Replace `SPECIFICATION.md`, `PLAN.md`, and `cap_spec.md` with current
     executable scope, adapter boundaries, and focused proof commands.
   - Clean primary package overclaims.

2. Contract expansion
   - Add upstream, consumer, route, policy, quota, canary, deployment, adapter,
     UI, and theme configuration.
   - Expand deterministic guardrails to at least 20 route and gateway lifecycle
     rules.
   - Add first-class gateway-agent runtime/role metadata for Codex, Claude
     Code, OpenCode, Pi, and future provider-neutral adapters.
   - Add Bytewax streaming metadata for APIG lifecycle batches.

3. Dependency-light lifecycle service
   - Extend `ApigService` with consumer, policy, canary, deployment, and
     retirement records/workflows.
   - Add gateway-agent records, registration guardrails, lifecycle-batch
     records, Bytewax validation, and dashboard counts.
   - Keep production gateway/service-mesh/edge runtime code behind adapter
     boundaries.

4. API and UI
   - Add generated-app helpers for consumer, policy, deployment, traffic shift,
     and retirement workflows.
   - Add generated-app helpers and view models for gateway agents and lifecycle
     batches.
   - Add `view_models.py` and keep `views.py` as a compatibility re-export.

5. Package evidence and tests
   - Replace static app semantic JSON with contract-derived evidence.
   - Refresh `semantic_model.json`, `release_report.json`, and
     `package_manifest.json`.
   - Expand focused tests for guardrails, lifecycle behavior, UI models, package
     contract shape, and app evidence.

6. Verification and commit
   - Run focused compile, APIG package tests, implementation audit, publish
     plan, stale marker scan, and diff checks.
   - Commit and push the coherent APIG packet.

## Battery-Conscious Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/apig/__init__.py capabilities/common/apig/capability_contract.py capabilities/common/apig/models.py capabilities/common/apig/gateway_runtime.py capabilities/common/apig/api.py capabilities/common/apig/view_models.py capabilities/common/apig/views.py capabilities/common/apig/app.py capabilities/common/apig/test_capability_contract.py capabilities/common/apig/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/apig/test_capability_contract.py capabilities/common/apig/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/apig --json
./.venv/bin/apg capabilities publish-plan capabilities/common/apig --json
```

Full repository tests, live proxy execution, Kubernetes deployment, service
mesh behavior, WebAssembly runtime execution, rendered UI, Bytewax flow
execution, and performance benchmarks are deferred.
