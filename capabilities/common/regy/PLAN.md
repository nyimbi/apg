# REGY Capability Build Plan

## Current State

REGY already has a production-oriented `ServiceRegistryService`, Flask REST
API, Flask-AppBuilder views, Pydantic domain models, package evidence, and
focused contract tests. The package still needed the richer lifecycle packet
shape used for the current APG capability build-out.

Known gaps addressed by this packet:

- No root `README.md`.
- No root `SPECIFICATION.md` or `PLAN.md`.
- Primary docs contained older market and speculative technology claims.
- Contract had only a narrow 6-rule, 8-route surface.
- `app.py` embedded static semantic JSON.
- There was no dependency-light generated-app registry runtime.
- UI helpers were tied to Flask-AppBuilder rather than the generated-app
  `view_models.py` convention.
- Package evidence and tests did not cover instances, discovery review,
  gateway publication, retirement, adapter metadata, or Bytewax streaming.
- Registry agents were not first-class citizens with supported runtimes, role
  guardrails, ownership, scope, purpose, contribution disclosure, approval
  state, UI, or audit evidence.
- Lifecycle mutation batches did not have executable Bytewax validation.

## Build Sequence

1. Documentation baseline
   - Add root `README.md`.
   - Add root `SPECIFICATION.md` and `PLAN.md`.
   - Replace primary package docs with current executable scope, adapter
     boundaries, and focused proof commands.

2. Contract expansion
   - Add service, instance, contract, discovery, health, routing, governance,
     observability, adapter, UI, and theme configuration.
   - Add first-class registry-agent and Bytewax lifecycle-batch configuration.
   - Expand deterministic guardrails to at least 33 registry lifecycle rules.

3. Dependency-light lifecycle service
   - Add `registry_runtime.RegistryService` for service registration, instance
     registration, discovery, version governance, gateway publication, health
     overrides, retirement, summaries, and audit events.
   - Add registry-agent records, lifecycle-batch records, guardrail-backed
     registration, Bytewax validation, summaries, and audit events.
   - Keep production registry, service-mesh, gateway, cache, and monitoring
     behavior behind adapter boundaries.

4. API and UI
   - Keep the existing Flask REST API intact.
   - Add generated-app API helper functions for services, agents, and lifecycle
     batches.
   - Add `view_models.py` surfaces for generated application UIs, including
     agent roster and lifecycle-batch monitor views.

5. Package evidence and tests
   - Replace static app semantic JSON with contract-derived evidence.
   - Refresh `semantic_model.json`, `release_report.json`, and
     `package_manifest.json`.
   - Expand focused tests for guardrails, lifecycle behavior, UI models,
     package contract shape, and app evidence.

6. Verification and commit
   - Run focused compile, REGY package tests, implementation audit,
     publish-plan, stale marker scan, and diff checks.
   - Commit and push the coherent REGY packet.

## Battery-Conscious Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/regy/__init__.py capabilities/common/regy/capability_contract.py capabilities/common/regy/models.py capabilities/common/regy/registry_runtime.py capabilities/common/regy/api.py capabilities/common/regy/view_models.py capabilities/common/regy/app.py capabilities/common/regy/test_capability_contract.py capabilities/common/regy/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/regy/test_capability_contract.py capabilities/common/regy/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/regy --json
./.venv/bin/apg capabilities publish-plan capabilities/common/regy --json
```

Full repository tests, live gateway execution, Kubernetes deployment, service
mesh behavior, Bytewax flow execution, rendered Flask-AppBuilder UI, and
performance benchmarks are deferred.
