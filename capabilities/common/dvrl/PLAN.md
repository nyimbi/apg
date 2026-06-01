# DVRL Capability Build Plan

## Current State

DVRL already includes a substantial production-oriented service, connector
support, adapters, API/views, integration modules, executable contract, package
entrypoint, semantic evidence, and tests. It still needs the same coherent
generated-application lifecycle packet used for the completed common
capabilities.

Known packet gaps:

- Root `SPECIFICATION.md` and `PLAN.md` were missing.
- Primary docs and work reports contained overclaiming production and market
  claims.
- Package test used legacy generated-package naming.
- `app.py` embedded static semantic JSON.
- Generated applications did not have a dependency-light lifecycle service.
- Generated applications now need first-class virtualization-agent composition
  and explicit Bytewax lifecycle-batch guardrails aligned with MDM, META, and
  ETLP.
- Generated applications need durable review evidence on every reviewable or
  denied lifecycle record.

## Build Sequence

1. Documentation baseline
   - Add root `SPECIFICATION.md` and `PLAN.md`.
   - Replace claim-heavy root docs with practical behavior and adapter
     boundaries.

2. Contract expansion
   - Add source, schema, query, cache, policy, adapter, audit, UI, and theme
     configuration.
   - Expand deterministic guardrails for source ownership, supported source
     type, vaulted credentials, encrypted connection, source approval, schema
     refresh, virtual table ownership, classification, read-only queries,
   parameterization, RBAC, cache sensitivity, lineage, cost, joins, result
   limits, cache TTL, policy review, and retirement impact review.
   - Add first-class virtualization-agent runtime/role metadata for Codex,
     Claude Code, OpenCode, Pi, and future provider-neutral adapters.
   - Add Bytewax streaming metadata for lifecycle batches.

3. Generated-app lifecycle service
   - Add dependency-light records and methods for source, schema, virtual
     table, query, cache, policy, source retirement, and audit workflows.
   - Add virtualization-agent records, registration guardrails, lifecycle-batch
     records, Bytewax validation, and dashboard counts.
   - Persist `policy_decision`, `matched_rules`, `review_reasons`, and
     `review_evidence` across reviewable lifecycle records and audit events.
   - Preserve denied non-Bytewax lifecycle-batch evidence before raising
     `PermissionError`.
   - Keep the production `DVRLService` intact as the runtime surface.

4. API and view models
   - Add generated-app helper functions.
   - Add view models for dashboard, source manager, schema browser, virtual
     table catalog, query workbench, federation map, cache console, policies,
     metrics, adapter health, agent roster, lifecycle batches, audit, and
     settings.

5. Package evidence and tests
   - Replace static `app.py` semantic JSON with contract-derived evidence.
   - Refresh `semantic_model.json`, `package_manifest.json`, and
     `release_report.json`.
   - Rename the package test and expand focused contract/lifecycle coverage.
   - Assert pending-review queues and review evidence appear in service,
     API/view models, registration metadata, semantic model, and release
     evidence.

6. Verification and commit
   - Run focused compile, package tests, implementation audit, publish plan,
     stale marker search, and diff checks.
   - Commit and push the coherent DVRL packet.

## Battery-Conscious Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/dvrl/__init__.py capabilities/common/dvrl/capability_contract.py capabilities/common/dvrl/service.py capabilities/common/dvrl/api.py capabilities/common/dvrl/view_models.py capabilities/common/dvrl/app.py capabilities/common/dvrl/test_capability_contract.py capabilities/common/dvrl/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/dvrl/test_capability_contract.py capabilities/common/dvrl/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/dvrl --json
./.venv/bin/apg capabilities publish-plan capabilities/common/dvrl --json
```

Full repository tests, live source connectors, query execution, rendered UI,
Bytewax runtime flows, and performance benchmarks are deferred.
