# ETLP Capability Build Plan

## Current State

ETLP already has a production-oriented async service, Pydantic models, FastAPI
controller, field mapper, executable capability contract, app package evidence,
and focused tests. It still needs a coherent generated-application lifecycle
packet comparable to the completed KEYM, MQEB, CACH, MONI, HLTH, MDM, and META
packets.

Known issues at packet start:

- Root README, specification, and plan were missing.
- Primary docs contained overclaiming benchmark and final-validation language.
- Package import reported an API setup error from a missing
  `ETLPAPIController.get_pipeline_logs` handler.
- The package test still used stale generated-package naming.
- `app.py` embeds static semantic JSON instead of deriving evidence from the
  contract.
- Generated applications do not yet have a dependency-light lifecycle service.

## Build Sequence

1. Documentation baseline
   - Add root `README.md`, `SPECIFICATION.md`, and `PLAN.md`.
   - Replace primary overclaiming package docs with practical usage and scope.
   - Retain future-looking ideas only as explicit backlog, not as implemented
     claims.

2. Import and package hygiene
   - Fix API route setup errors.
   - Rename the legacy generated-package test file to
     `tests/test_package_contract.py`.
   - Update package description language.
   - Remove stale generated/static package markers.

3. Contract expansion
   - Add lifecycle configuration for datasources, mappings, execution,
     scheduling, retry, replay, backfill, publishing, adapters, audit, and UI.
   - Add deterministic rules for datasource approval, schedule review, replay
     range, retry count, embedded secrets, destructive delete, publish quality,
     lineage emission, cost review, and tenant context.
   - Expand UI routes and theme components for the full lifecycle.

4. Generated-app lifecycle service
   - Add dependency-light records for pipelines, datasources, transformations,
     mappings, executions, quality results, publish reviews, schedules, replay
     requests, audit events, and adapter health.
   - Add helper methods for register, approve, schedule, execute, pause,
     resume, cancel, retry, replay, backfill, assess quality, publish, and
     retire.
   - Keep `ETLPService` as the production runtime and make adapter boundaries
     explicit.

5. API and view-model surface
   - Add generated-app helper functions to `api.py`.
   - Add or refine view models for dashboard, pipelines, designer, field
     mapper, executions, quality, datasources, schedules, publish review,
     lineage, adapter health, audit, and settings.

6. Package evidence
   - Replace static semantic JSON with contract-derived evidence.
   - Refresh `semantic_model.json`, `package_manifest.json`, and
     `release_report.json`.
   - Ensure publish-plan output shows side-effect-free, domain-specific ETLP
     evidence.

7. Focused review and verification
   - Run `py_compile` for ETLP package files.
   - Run focused ETLP package tests only.
   - Run implementation audit and publish plan.
   - Run stale marker search over primary ETLP package files.
   - Run `git diff --check`.
   - Document known gaps and not-run checks in `docs/progress_log.md`.

## Battery-Conscious Verification

Use focused checks during implementation:

```bash
./.venv/bin/python -m py_compile capabilities/common/etlp/__init__.py capabilities/common/etlp/capability_contract.py capabilities/common/etlp/models.py capabilities/common/etlp/service.py capabilities/common/etlp/api.py capabilities/common/etlp/field_mapper.py capabilities/common/etlp/views.py capabilities/common/etlp/app.py capabilities/common/etlp/test_capability_contract.py capabilities/common/etlp/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/etlp/test_capability_contract.py capabilities/common/etlp/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/etlp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/etlp --json
```

Do not run full repository tests until a larger verification window is
available.
