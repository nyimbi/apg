# CONN Capability Build Plan

## Current State

CONN already has production-oriented connection management modules, Singer
runtime scaffolding, local tap packages, FastAPI routes, Flask-AppBuilder views,
React frontend code, package evidence, and tests. The package still needed the
richer lifecycle packet shape used for the current APG capability build-out.

Known gaps addressed by this packet:

- No root `SPECIFICATION.md` or `PLAN.md`.
- Root `README.md` and primary docs contained broad market and completion
  claims rather than current executable scope.
- Contract had only a narrow 3-rule, 7-route surface.
- `app.py` embedded static semantic JSON.
- There was no dependency-light generated-app connector lifecycle runtime.
- UI helpers were not available as generated-app `view_models.py`.
- Package evidence and tests did not cover connector marketplace review,
  credential vault, activation, flow, sync, schedule, replay, retirement,
  adapter metadata, or Bytewax streaming.

## Build Sequence

1. Documentation baseline
   - Replace root `README.md`.
   - Add root `SPECIFICATION.md` and `PLAN.md`.
   - Replace primary package docs with current executable scope, adapter
     boundaries, and focused proof commands.

2. Contract expansion
   - Add connector, connection, flow, sync, security, quality, governance,
     observability, adapter, UI, and theme configuration.
   - Expand deterministic guardrails to at least 25 lifecycle rules.

3. Dependency-light lifecycle service
   - Add `conn_runtime.ConnService` for connector, connection, test,
     activation, flow, sync, schedule, replay, owner transfer, retirement,
     review, summary, and audit records.
   - Keep Singer execution, service calls, credentials, monitoring, lineage,
     quality engines, and Bytewax behavior behind adapter boundaries.

4. API and UI
   - Keep the existing FastAPI runtime intact.
   - Add generated-app API helper functions.
   - Add `view_models.py` for generated application UIs.

5. Package evidence and tests
   - Replace static app semantic JSON with contract-derived evidence.
   - Refresh `semantic_model.json`, `release_report.json`, and
     `package_manifest.json`.
   - Replace baseline-package test naming with package-contract naming.
   - Expand focused tests for guardrails, lifecycle behavior, UI models,
     package contract shape, and app evidence.

6. Verification and commit
   - Run focused compile, CONN package tests, implementation audit,
     publish-plan, stale marker scan, and diff checks.
   - Commit and push the coherent CONN packet.

## Battery-Conscious Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/conn/__init__.py capabilities/common/conn/capability_contract.py capabilities/common/conn/models.py capabilities/common/conn/conn_runtime.py capabilities/common/conn/api.py capabilities/common/conn/view_models.py capabilities/common/conn/app.py capabilities/common/conn/tests/test_capability_contract.py capabilities/common/conn/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/conn/tests/test_capability_contract.py capabilities/common/conn/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/conn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/conn --json
```

Full repository tests, live Singer tap execution, external SaaS/database calls,
credential vault access, Bytewax flow execution, rendered frontend/browser
behavior, and performance benchmarks are deferred.
